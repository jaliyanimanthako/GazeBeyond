"""One recording's SLAM semidense points as a 3D scene: an interactive HTML viewer (orbit / zoom, the head trajectory,
a time slider that moves the head and its gaze ray through the scene) and a binary .ply for MeshLab / CloudCompare.

    # AEA (lab server layout, as build_dataset.py aea-server)
    python scripts/render_scene.py aea-server --rec loc1_script2_seq1_rec1 --slam-root /data/SS/Aria_data/mps_slam \
        --gaze-root /data/SS/Aria_data/gaze --cpf-root data/aea_cpf --out figs/scene
    # HD-EPIC: one recording, or with --kitchen every session of its kitchen (one world frame)
    python scripts/render_scene.py hdepic --rec P01-20240202-110250 \
        --slam-root /data/disk2/hdepic_dinithi/HD-EPIC/SLAM-and-Gaze --src /data/disk2/hdepic/HD-EPIC/raw --out figs/scene
    python scripts/render_scene.py hdepic --list --slam-root ...       # the recordings there

Writes <out>/<rec>.html (self-contained apart from three.js from a CDN; open it in a browser) and <out>/<rec>.ply.
Points are kept when MPS is confident in them (inverse-distance std and distance std below --max-inv-dist-std /
--max-dist-std, Project Aria's defaults) and coloured by height above the floor (along gravity).
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.aria import T_DEVICE_CPF_NOMINAL, _read_csv, list_hdepic, read_gaze_pose   # noqa: E402

POINT_COLS = ["px_world", "py_world", "pz_world", "inv_dist_std", "dist_std"]


def sources(a):
    """(gaze_src, trajectory_src, [points_src, ...], T_device_cpf, gaze_rows), the same files as the dataset readers
    (read_aea_server, read_hdepic), without the (large) observation tables."""
    if a.cmd == "aea-server":
        T = T_DEVICE_CPF_NOMINAL
        cpf = a.cpf_root and os.path.join(a.cpf_root, f"{a.rec}.json")
        if cpf and os.path.isfile(cpf):
            T = np.array(json.load(open(cpf))["T_device_cpf"])
        else:
            print("no per-recording T_device_cpf (--cpf-root): the nominal one, gaze rays off by a degree or so")
        f = lambda suffix: os.path.join(a.slam_root, f"{a.rec}_{suffix}")     # noqa: E731
        return (os.path.join(a.gaze_root, a.rec, "mps", "eye_gaze", "general_eye_gaze.csv"),
                (f("mps_slam_trajectories.zip"), "closed_loop_trajectory.csv"),
                [(f("mps_slam_points.zip"), "semidense_points.csv.gz")], T, None)
    recs = list_hdepic(a.slam_root)
    hits = [(P, z, i) for r, P, z, i in recs if r == a.rec]
    if not hits:
        raise SystemExit(f"{a.rec} not in {a.slam_root} (--list shows the recordings)")
    P, z, i = hits[0]
    T = T_DEVICE_CPF_NOMINAL.copy()
    cal = os.path.join(a.src, "HD-EPIC_INTERMEDIATE", P, a.rec, "device_calibration.json")
    if os.path.isfile(cal):
        T[:3] = np.array(json.load(open(cal))["origin_transforms"]["T_device_cpf"], float)[:3, :4]
    # one zip per session (SLAM/multi/<i>.zip), all sessions of a kitchen in the same world frame
    pts = [(zk, f"{ik}/slam/semidense_points.csv.gz") for _, Pk, zk, ik in recs if Pk == P] if a.kitchen \
        else [(z, f"{i}/slam/semidense_points.csv.gz")]
    return ((os.path.join(a.src, "HD-EPIC_GAZE", P, "GAZE_HAND.zip"), f"GAZE_HAND/mps_{a.rec}_vrs.zip",
             f"mps_{a.rec}_vrs/eye_gaze/general_eye_gaze.csv"),
            (z, f"{i}/slam/closed_loop_trajectory.csv"), pts, T, None)


def load_points(srcs, max_inv, max_dist):
    out = []
    for s in srcs:
        Pt = _read_csv(s, usecols=POINT_COLS)
        keep = (Pt["inv_dist_std"].to_numpy() < max_inv) & (Pt["dist_std"].to_numpy() < max_dist)
        out.append(Pt[POINT_COLS[:3]].to_numpy(np.float32)[keep])
        print(f"{s[-1]}: {keep.sum():,} of {len(Pt):,} points kept")
    return np.concatenate(out)


def height_colors(h):
    """viridis-like ramp over the 2nd-98th percentile of height."""
    lo, hi = np.percentile(h, [2, 98])
    t = np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)[:, None]
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    x = t * (len(stops) - 1)
    k = np.minimum(x.astype(int), len(stops) - 2)
    return (stops[k[:, 0]] * (1 - (x - k)) + stops[k[:, 0] + 1] * (x - k)).astype(np.uint8)


def write_ply(path, xyz, rgb):
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["r"], v["g"], v["b"] = rgb.T
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                 % len(v)).encode())
        f.write(v.tobytes())


def b64(x):
    return base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#111418; --panel:#1b1f25ee; --text:#e6e8eb; --muted:#9aa3ad; --accent:#ff6b4a; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--text); font:13px system-ui,sans-serif; overflow:hidden; }
  #ui { position:absolute; left:12px; right:12px; bottom:12px; background:var(--panel); border-radius:8px; padding:10px 12px;
        display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; }
  #ui label { color:var(--muted); display:flex; gap:6px; align-items:center; }
  #time { flex:1 1 240px; }
  #info { position:absolute; left:12px; top:10px; color:var(--muted); }
  button { background:#2a3038; color:var(--text); border:0; border-radius:5px; padding:5px 10px; cursor:pointer; }
</style></head><body>
<div id="info">__TITLE__ &middot; __NPTS__ points &middot; drag to orbit, scroll to zoom, right-drag to pan</div>
<div id="ui">
  <button id="play">Play</button>
  <input id="time" type="range" min="0" max="__NFRAMES_1__" value="0" step="1">
  <span id="clock">0.0 s</span>
  <label>point size <input id="size" type="range" min="0.5" max="6" step="0.5" value="1.5"></label>
  <label>gaze ray (m) <input id="ray" type="range" min="0.5" max="5" step="0.25" value="__RAY__"></label>
  <label><input id="follow" type="checkbox"> follow head</label>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const dec = (s, T) => { const b = Uint8Array.from(atob(s), c => c.charCodeAt(0)); return new T(b.buffer); };
const P = dec("__XYZ__", Float32Array), C = dec("__RGB__", Uint8Array);
const H = dec("__HEAD__", Float32Array), G = dec("__GAZE__", Float32Array), TS = dec("__T__", Float32Array);
const up = new THREE.Vector3(...__UP__);
const N = TS.length;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x111418);
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 500);
camera.up.copy(up);
const controls = new OrbitControls(camera, renderer.domElement);

const pg = new THREE.BufferGeometry();
pg.setAttribute("position", new THREE.BufferAttribute(P, 3));
pg.setAttribute("color", new THREE.BufferAttribute(C, 3, true));
const pm = new THREE.PointsMaterial({ size: 1.5, sizeAttenuation: false, vertexColors: true });
scene.add(new THREE.Points(pg, pm));

const tg = new THREE.BufferGeometry();
tg.setAttribute("position", new THREE.BufferAttribute(H, 3));
const tc = new Float32Array(N * 3);
for (let i = 0; i < N; i++) { const c = new THREE.Color().setHSL(0.6 - 0.6 * i / N, 0.8, 0.6); tc.set([c.r, c.g, c.b], 3 * i); }
tg.setAttribute("color", new THREE.BufferAttribute(tc, 3));
scene.add(new THREE.Line(tg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7 })));

const head = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
scene.add(head);
const rg = new THREE.BufferGeometry(); rg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
const ray = new THREE.Line(rg, new THREE.LineBasicMaterial({ color: 0xff6b4a }));
scene.add(ray);
const tip = new THREE.Mesh(new THREE.SphereGeometry(0.04, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff6b4a }));
scene.add(tip);

const box = new THREE.Box3().setFromBufferAttribute(pg.getAttribute("position"));
const centre = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()).length();
controls.target.copy(centre);
camera.position.copy(centre).add(new THREE.Vector3(0.5, 0.5, 0.5).normalize().multiplyScalar(size * 0.6)).addScaledVector(up, size * 0.3);
controls.update();

const $ = id => document.getElementById(id);
let k = 0, playing = false, last = 0;
function setFrame(i) {
  k = Math.max(0, Math.min(N - 1, i)); $("time").value = k; $("clock").textContent = TS[k].toFixed(1) + " s";
  const h = new THREE.Vector3(H[3*k], H[3*k+1], H[3*k+2]), g = new THREE.Vector3(G[3*k], G[3*k+1], G[3*k+2]);
  const e = h.clone().addScaledVector(g, +$("ray").value);
  head.position.copy(h); tip.position.copy(e);
  rg.attributes.position.array.set([h.x, h.y, h.z, e.x, e.y, e.z]); rg.attributes.position.needsUpdate = true;
  if ($("follow").checked) { const d = h.clone().sub(controls.target); controls.target.copy(h); camera.position.add(d); }
}
$("time").oninput = e => setFrame(+e.target.value);
$("ray").oninput = () => setFrame(k);
$("size").oninput = e => { pm.size = +e.target.value; };
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "Pause" : "Play"; last = performance.now(); };
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
setFrame(0);
renderer.setAnimationLoop(now => {
  if (playing) {           // real time: advance by the elapsed seconds on the recording's clock
    const t = TS[k] + (now - last) / 1000; last = now;
    let j = k; while (j < N - 1 && TS[j + 1] <= t) j++;
    if (j === N - 1) { playing = false; $("play").textContent = "Play"; }
    if (j !== k) setFrame(j);
  }
  controls.update(); renderer.render(scene, camera);
});
</script></body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["aea-server", "hdepic"])
    ap.add_argument("--rec")
    ap.add_argument("--list", action="store_true", help="hdepic: list the recordings with SLAM and exit")
    ap.add_argument("--slam-root", required=True, help="aea-server: folder with <rec>_mps_slam_*.zip; hdepic: SLAM-and-Gaze/")
    ap.add_argument("--gaze-root", help="aea-server: folder with <rec>/mps/eye_gaze/")
    ap.add_argument("--cpf-root", help="aea-server: folder with <rec>.json from scripts/derive_device_cpf.py")
    ap.add_argument("--src", help="hdepic: the HD-EPIC raw/ folder (eye gaze, calibration)")
    ap.add_argument("--kitchen", action="store_true", help="hdepic: points of every session of the recording's kitchen")
    ap.add_argument("--t0", type=float, default=0.0, help="trajectory from t0 s ...")
    ap.add_argument("--t1", type=float, help="... to t1 s (default: the end)")
    ap.add_argument("--max-points", type=int, default=1_000_000, help="random subset for the HTML (the .ply keeps all)")
    ap.add_argument("--max-inv-dist-std", type=float, default=0.001)
    ap.add_argument("--max-dist-std", type=float, default=0.05)
    ap.add_argument("--ray", type=float, default=1.5, help="initial gaze ray length (m)")
    ap.add_argument("--out", default="figs/scene")
    a = ap.parse_args()
    if a.list:
        for r, P, _, i in list_hdepic(a.slam_root):
            print(P, r, f"session {i}")
        sys.exit()
    if not a.rec:
        ap.error("--rec is needed")

    gaze_src, traj_src, pts_src, T, rows = sources(a)
    xyz = load_points(pts_src, a.max_inv_dist_std, a.max_dist_std)
    t, gaze, T_world_cpf, ok, gravity = read_gaze_pose(gaze_src, traj_src, T, rows)
    t = (t - t[0]) / 1e6
    ok &= np.isfinite(gaze).all(1) & (t >= a.t0) & (t <= (a.t1 if a.t1 is not None else np.inf))
    step = max(1, int(round(0.1 / np.median(np.diff(t)))))   # 10 Hz is plenty for the viewer
    k = np.flatnonzero(ok)[::step]
    head = T_world_cpf[k, :3, 3]
    gaze_w = np.einsum("nij,nj->ni", T_world_cpf[k, :3, :3], gaze[k])

    up = -gravity / np.linalg.norm(gravity) if gravity is not None else np.array([0, 0, 1.0])
    origin = np.median(head, 0)                               # small coordinates for float32 in the browser
    xyz = xyz - origin.astype(np.float32)
    rgb = height_colors(xyz @ up.astype(np.float32))
    a.out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(a.out, exist_ok=True)
    name = a.rec + ("_kitchen" if a.kitchen else "")
    write_ply(os.path.join(a.out, f"{name}.ply"), xyz, rgb)

    sub = np.random.default_rng(0).permutation(len(xyz))[:a.max_points]
    html = (HTML.replace("__TITLE__", name).replace("__NPTS__", f"{len(sub):,}")
            .replace("__NFRAMES_1__", str(len(k) - 1)).replace("__RAY__", str(a.ray)).replace("__UP__", json.dumps(up.tolist()))
            .replace("__XYZ__", b64(xyz[sub])).replace("__RGB__", b64(rgb[sub]))
            .replace("__HEAD__", b64((head - origin).astype(np.float32))).replace("__GAZE__", b64(gaze_w.astype(np.float32)))
            .replace("__T__", b64(t[k].astype(np.float32))))
    path = os.path.join(a.out, f"{name}.html")
    open(path, "w").write(html)
    print(f"{path} ({os.path.getsize(path) / 1e6:.0f} MB, {len(sub):,} points, {len(k):,} frames over {t[k[-1]] - t[k[0]]:.0f} s)\n"
          f"{os.path.join(a.out, name + '.ply')} ({len(xyz):,} points)")
