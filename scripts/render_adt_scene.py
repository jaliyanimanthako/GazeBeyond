"""An Aria Digital Twin (ADT) sequence as a clean 3D scene for figures: every object of the apartment as its
ground-truth 3D box (name, category), the wearer's head path, the fixations (ground-truth 3D gaze points, exact depth
from the twin) and, at a chosen moment, the current gaze ray and the next attention shift (ray to its landing, the
landed-on object highlighted). Needs only the lab's ADT download (main_groundtruth), no SLAM points.

    python scripts/render_adt_scene.py --seq Apartment_release_clean_seq131_M1292 --inspect      # what the zip holds
    python scripts/render_adt_scene.py --seq Apartment_release_clean_seq131_M1292 --list           # teaser moments
    python scripts/render_adt_scene.py --seq Apartment_release_clean_seq131_M1292 --out ~/GazeBeyond/figs/adt_scene

Writes <out>/<seq>.html: open it in a browser, scrub to a moment, frame the view, "Save PNG" (3x resolution, optionally
transparent) for the teaser. Shifts as in the paper (approximately): fixations from the world gaze speed (< 75 deg/s for
>= 100 ms), a shift = consecutive fixations >= 30 deg apart; "out of view": the landing is more than --fov-half deg from
the head's forward axis 0.3 s before the shift. --list ranks shifts that land out of view on a named object; --t picks
the moment the viewer opens at (default: the top one). Dynamic objects are drawn where they are at that moment.
"""
import argparse
import base64
import glob
import io
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from egospan.data.aria import T_DEVICE_CPF_ADT, _read_csv, read_gaze_pose   # noqa: E402
from egospan.data.streams import quat_to_rotmat                           # noqa: E402

SKIP_CATEGORIES = {"human", "person"}


def col(df, prefix):
    """the column whose name starts with `prefix` (ADT puts units in the names, e.g. t_wo_x[m])."""
    hits = [c for c in df.columns if c.startswith(prefix)]
    if not hits:
        raise KeyError(f"no column {prefix}* in {list(df.columns)}")
    return df[hits[0]].to_numpy()


def angle(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def read_objects(gt, t_ns):
    """[(name, category, motion, R_world_obj (3, 3), centre (3,), size (3,))] at device time t_ns."""
    zf = zipfile.ZipFile(gt)
    inst = json.loads(zf.read("instances.json"))
    S = _read_csv((gt, "scene_objects.csv"))
    B = _read_csv((gt, "3d_bounding_box.csv"))
    box = {}
    for u, x0, x1, y0, y1, z0, z1 in zip(col(B, "object_uid"), col(B, "p_local_obj_xmin"), col(B, "p_local_obj_xmax"),
                                         col(B, "p_local_obj_ymin"), col(B, "p_local_obj_ymax"),
                                         col(B, "p_local_obj_zmin"), col(B, "p_local_obj_zmax")):
        box.setdefault(int(u), np.array([[x0, y0, z0], [x1, y1, z1]], float))
    uid, ts = col(S, "object_uid").astype(np.int64), col(S, "timestamp").astype(np.int64)
    t = np.stack([col(S, "t_wo_x"), col(S, "t_wo_y"), col(S, "t_wo_z")], 1).astype(float)
    q = np.stack([col(S, "q_wo_x"), col(S, "q_wo_y"), col(S, "q_wo_z"), col(S, "q_wo_w")], 1).astype(float)
    order = np.argsort(uid, kind="stable")
    us, first = np.unique(uid[order], return_index=True)
    out = []
    for u, rows in zip(us, np.split(order, first[1:])):
        r = rows[np.argmin(np.abs(ts[rows] - t_ns))]           # static objects have a single row
        info = inst.get(str(u), {})
        cat = str(info.get("category", "")).lower()
        if int(u) not in box or info.get("instance_type", "object") != "object" or cat in SKIP_CATEGORIES:
            continue
        R = quat_to_rotmat(q[r])
        lo, hi = box[int(u)]
        out.append((info.get("instance_name", str(u)), cat or "other", info.get("motion_type", ""),
                    R, t[r] + R @ ((lo + hi) / 2), hi - lo))
    return out


def fixations(t, wdir, p, valid, max_speed=75.0, min_dur=0.1):
    """[(t_start, t_end, mean world gaze direction, median 3D point or None)] from the world gaze speed."""
    dt = np.diff(t)
    slow = (angle(wdir[1:], wdir[:-1]) / np.maximum(dt, 1e-3) < max_speed) & valid[1:] & valid[:-1] & (dt < 0.15)
    cuts = np.r_[0, np.flatnonzero(~slow) + 1, len(t)]
    F = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 2 or t[b - 1] - t[a] < min_dur:
            continue
        k = np.arange(a, b)
        d = wdir[k].mean(0)
        hit = np.isfinite(p[k]).all(1)
        F.append((t[a], t[b - 1], d / np.linalg.norm(d), np.median(p[k][hit], 0) if hit.any() else None))
    return F


def landed_object(x, objects, margin=0.05):
    """index of the object whose box contains x (the smallest one), else the nearest box within 0.3 m, else None."""
    best, best_d = None, np.inf
    for i, (_, _, _, R, c, s) in enumerate(objects):
        d = np.abs(R.T @ (x - c)) - s / 2                      # > 0 outside along an axis
        out = np.linalg.norm(np.maximum(d, 0))
        score = out if out > margin else -1 / max(np.prod(s), 1e-6)
        if score < best_d and out < 0.3:
            best, best_d = i, score
    return best


def align_up(up):
    """rotation taking `up` to +y (three.js' up)."""
    u = up / np.linalg.norm(up)
    y = np.array([0, 1.0, 0])
    v, c = np.cross(u, y), float(u @ y)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V * (1 / (1 + c))


PALETTE = ["#8fb3d9", "#e0a96d", "#9ccf9a", "#d99ab8", "#b9a6e0", "#e6d27a", "#7ec4c1", "#d98f8f", "#a8b5a0", "#c9b38f"]

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#f4f2ee; --panel:#ffffffee; --text:#22262b; --muted:#6b737c; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--text); font:13px system-ui,sans-serif; overflow:hidden; }
  #ui { position:absolute; left:12px; right:12px; bottom:12px; background:var(--panel); border-radius:8px; padding:10px 12px;
        display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; box-shadow:0 1px 6px #0002; }
  #ui label { color:var(--muted); display:flex; gap:6px; align-items:center; }
  #time { flex:1 1 240px; }
  #info { position:absolute; left:12px; top:10px; color:var(--muted); }
  #hover { position:absolute; right:12px; top:10px; font-weight:600; }
  button { background:#22262b; color:#fff; border:0; border-radius:5px; padding:5px 10px; cursor:pointer; }
  select { font:inherit; }
</style></head><body>
<div id="info">__TITLE__ &middot; __NOBJ__ &middot; drag to orbit, scroll to zoom, right-drag to pan</div>
<div id="hover"></div>
<div id="ui">
  <button id="play">Play</button>
  <input id="time" type="range" min="0" max="__NFRAMES_1__" value="__K0__" step="1">
  <span id="clock"></span>
  <label>shift <select id="shift"></select></label>
  <label>history (s) <input id="hist" type="range" min="0" max="60" step="1" value="20"></label>
  <label><input id="labels" type="checkbox"> object names</label>
  <label><input id="path" type="checkbox" checked> head path</label>
  <label id="cloudui">point size <input id="psize" type="range" min="0" max="5" step="0.5" value="1.5"></label>
  <label><input id="transp" type="checkbox"> transparent PNG</label>
  <button id="save">Save PNG</button>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
const D = __DATA__;
const N = D.t.length, $ = id => document.getElementById(id);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const labelR = new CSS2DRenderer(); labelR.setSize(innerWidth, innerHeight);
Object.assign(labelR.domElement.style, { position: "absolute", top: "0", pointerEvents: "none" });
document.body.appendChild(labelR.domElement);
const BG = new THREE.Color(0xf4f2ee);
const scene = new THREE.Scene(); scene.background = BG;
scene.add(new THREE.HemisphereLight(0xffffff, 0xb8b0a0, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(3, 8, 4); scene.add(sun);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.02, 200);
const controls = new OrbitControls(camera, renderer.domElement);

// floor
const g = new THREE.GridHelper(D.floorSize, Math.round(D.floorSize * 2), 0xc9c3b8, 0xdedad2);
g.position.set(D.floorCentre[0], D.floor, D.floorCentre[1]); scene.add(g);

// scene points (HD-EPIC: the kitchen's SLAM point cloud)
let cloud = null;
if (D.pts) {
  const dec = (s, T) => { const b = Uint8Array.from(atob(s), c => c.charCodeAt(0)); return new T(b.buffer); };
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(dec(D.pts, Float32Array), 3));
  pg.setAttribute("color", new THREE.BufferAttribute(dec(D.rgb, Uint8Array), 3, true));
  cloud = new THREE.Points(pg, new THREE.PointsMaterial({ size: 1.5, sizeAttenuation: false, vertexColors: true }));
  scene.add(cloud);
} else { $("cloudui").style.display = "none"; }

// objects: translucent boxes with edges, coloured by category; or, with positions only, labelled dots
const boxes = [], labels = [];
D.objects.forEach((o, i) => {
  if (D.markers) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.035, 16, 12), new THREE.MeshLambertMaterial({ color: 0x3b4148 }));
    m.position.set(...o.c); m.userData = { i }; scene.add(m); boxes.push(m);
    const div = document.createElement("div"); div.textContent = o.name;
    Object.assign(div.style, { font: "11px system-ui", color: "#22262b", background: "#ffffffcc", padding: "1px 4px", borderRadius: "3px", transform: "translateY(-14px)" });
    const l = new CSS2DObject(div); l.position.set(...o.c); l.visible = false; scene.add(l); labels.push(l);
    return;
  }
  const geo = new THREE.BoxGeometry(...o.size);
  const m = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: o.color, transparent: true, opacity: 0.35, depthWrite: false }));
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: o.color, transparent: true, opacity: 0.9 }));
  const M = new THREE.Matrix4().set(...o.R.slice(0, 3), o.c[0], ...o.R.slice(3, 6), o.c[1], ...o.R.slice(6, 9), o.c[2], 0, 0, 0, 1);
  for (const x of [m, e]) { x.applyMatrix4(M); scene.add(x); }
  m.userData = { i, edge: e }; boxes.push(m);
  const div = document.createElement("div"); div.textContent = o.name;
  Object.assign(div.style, { font: "11px system-ui", color: "#22262b", background: "#ffffffcc", padding: "1px 4px", borderRadius: "3px" });
  const l = new CSS2DObject(div); l.position.set(...o.c); l.visible = false; scene.add(l); labels.push(l);
});

// head path
const hp = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(D.head.flat(), 3));
const path = new THREE.Line(hp, new THREE.LineBasicMaterial({ color: 0x8a8f96, transparent: true, opacity: 0.5 }));
scene.add(path);

// fixations: the attention history (older = smaller, fainter)
const F = D.fix.filter(f => f.p);
const fg = new THREE.BufferGeometry();
fg.setAttribute("position", new THREE.Float32BufferAttribute(F.flatMap(f => f.p), 3));
fg.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(F.length * 3), 3));
const fixPts = new THREE.Points(fg, new THREE.PointsMaterial({ size: 0.05, vertexColors: true, transparent: true }));
scene.add(fixPts);

// the wearer: head, view cone, gaze ray, the next shift
const ORANGE = 0xf26b38, TEAL = 0x13a89e;
const head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 24, 16), new THREE.MeshLambertMaterial({ color: 0x3b4148 }));
scene.add(head);
const coneLen = 0.45, coneR = coneLen * Math.tan(D.fovHalf * Math.PI / 180);
const cone = new THREE.Mesh(new THREE.ConeGeometry(coneR, coneLen, 32, 1, true),
  new THREE.MeshBasicMaterial({ color: 0x3b4148, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
cone.geometry.translate(0, -coneLen / 2, 0); cone.geometry.rotateX(-Math.PI / 2);   // apex at the head, opening along +z
scene.add(cone);
const mkLine = (color, w) => { const l = new THREE.Line(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(3 * 40), 3)),
  new THREE.LineBasicMaterial({ color, linewidth: w })); scene.add(l); return l; };
const gazeRay = mkLine(ORANGE), shiftArc = mkLine(TEAL);
const dot = (color, r) => { const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshBasicMaterial({ color })); scene.add(m); return m; };
const gazeDot = dot(ORANGE, 0.035), landDot = dot(TEAL, 0.045);
const ring = new THREE.Mesh(new THREE.TorusGeometry(0.09, 0.012, 8, 32), new THREE.MeshBasicMaterial({ color: TEAL }));
scene.add(ring);

D.shifts.forEach((s, i) => { const o = document.createElement("option"); o.value = i;
  o.textContent = `${s.t.toFixed(1)} s, ${s.size.toFixed(0)}°${s.out ? ", out of view" : ""}${s.obj != null ? " → " + D.objects[s.obj].name : ""}`; $("shift").appendChild(o); });

const setLine = (l, pts) => { const a = l.geometry.attributes.position.array; a.fill(0);
  pts.forEach((p, i) => a.set(p, 3 * i)); for (let i = pts.length; i < 40; i++) a.set(pts[pts.length - 1], 3 * i);
  l.geometry.attributes.position.needsUpdate = true; };
let hl = null;
function setFrame(k) {
  k = Math.max(0, Math.min(N - 1, k)); $("time").value = k;
  const t = D.t[k], h = new THREE.Vector3(...D.head[k]), fw = new THREE.Vector3(...D.fwd[k]), gd = new THREE.Vector3(...D.gaze[k]);
  $("clock").textContent = t.toFixed(1) + " s";
  head.position.copy(h); cone.position.copy(h); cone.lookAt(h.clone().add(fw));
  const gp = D.gp[k] ? new THREE.Vector3(...D.gp[k]) : h.clone().addScaledVector(gd, 1.5);
  setLine(gazeRay, [h.toArray(), gp.toArray()]); gazeDot.position.copy(gp);
  // history: fixations that ended in the last `hist` seconds
  const H = +$("hist").value, c = fg.attributes.color.array;    // fixations are in time order: draw [i0, i1)
  let i0 = F.findIndex(f => t - f.t1 < H), i1 = F.findIndex(f => f.t0 > t);
  if (i0 < 0) i0 = F.length; if (i1 < 0) i1 = F.length;
  for (let i = i0; i < i1; i++) { const w = 1 - Math.max(t - F[i].t1, 0) / Math.max(H, 1e-3);   // 1 = now, 0 = oldest
    c.set([0.95 - 0.35 * (1 - w), 0.42 + 0.4 * (1 - w), 0.22 + 0.55 * (1 - w)], 3 * i); }
  fg.setDrawRange(i0, Math.max(i1 - i0, 0)); fg.attributes.color.needsUpdate = true;
  // the next shift within 2 s: an arc from the current gaze to its landing
  const s = D.shifts.find(s => s.t >= t && s.t - t <= 2);
  if (hl) { if (D.markers) { hl.material.color.set(0x3b4148); hl.scale.setScalar(1); labels[hl.userData.i].visible = $("labels").checked; }
            else { hl.material.opacity = 0.35; hl.userData.edge.material.color.set(D.objects[hl.userData.i].color); } hl = null; }
  [shiftArc, landDot, ring].forEach(x => x.visible = !!s);
  if (s) {
    const L = new THREE.Vector3(...s.p), mid = gp.clone().add(L).multiplyScalar(0.5);
    mid.y += 0.25 * gp.distanceTo(L);
    const curve = new THREE.QuadraticBezierCurve3(gp, mid, L);
    setLine(shiftArc, curve.getPoints(39).map(v => v.toArray()));
    landDot.position.copy(L); ring.position.copy(L); ring.lookAt(h);
    if (s.obj != null) { hl = boxes[s.obj];
      if (D.markers) { hl.material.color.set(TEAL); hl.scale.setScalar(1.6); labels[s.obj].visible = true; }
      else { hl.material.opacity = 0.7; hl.userData.edge.material.color.set(TEAL); } }
  }
}
$("time").oninput = e => setFrame(+e.target.value);
$("hist").oninput = () => setFrame(+$("time").value);
$("shift").onchange = e => { const s = D.shifts[+e.target.value]; setFrame(D.t.findIndex(x => x >= s.t - 0.3)); };
$("labels").onchange = e => labels.forEach(l => l.visible = e.target.checked);
$("path").onchange = e => { path.visible = e.target.checked; };
$("psize").oninput = e => { if (cloud) { cloud.material.size = +e.target.value; cloud.visible = +e.target.value > 0; } };
let playing = false, last = 0;
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "Pause" : "Play"; last = performance.now(); };
$("save").onclick = () => {
  const pr = renderer.getPixelRatio(); renderer.setPixelRatio(3);
  if ($("transp").checked) scene.background = null;
  renderer.render(scene, camera);
  const a = document.createElement("a"); a.download = "__TITLE__.png"; a.href = renderer.domElement.toDataURL("image/png"); a.click();
  scene.background = BG; renderer.setPixelRatio(pr);
};
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
renderer.domElement.addEventListener("pointermove", e => {
  mouse.set(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1); ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(boxes)[0]; $("hover").textContent = hit ? D.objects[hit.object.userData.i].name + " (" + D.objects[hit.object.userData.i].cat + ")" : "";
});
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); labelR.setSize(innerWidth, innerHeight); });

const k0 = +$("time").value, h0 = new THREE.Vector3(...D.head[k0]);       // over the wearer's shoulder, from behind
controls.target.copy(h0).add(new THREE.Vector3(...D.fwd[k0]).setY(0).normalize().multiplyScalar(1.0)).setY(h0.y - 0.4);
camera.position.copy(h0).add(new THREE.Vector3(...D.fwd[k0]).setY(0).normalize().multiplyScalar(-2.2)).setY(h0.y + 1.4);
controls.update(); setFrame(k0);
const s0 = D.shifts.findIndex(s => s.t >= D.t[k0]); if (s0 >= 0) $("shift").value = s0;
renderer.setAnimationLoop(now => {
  if (playing) { const t = D.t[+$("time").value] + (now - last) / 1000; last = now;
    let j = +$("time").value; while (j < N - 1 && D.t[j + 1] <= t) j++; if (j === N - 1) $("play").click(); setFrame(j); }
  controls.update(); renderer.render(scene, camera); labelR.render(scene, camera);
});
</script></body></html>
"""


def find_shifts(t, eye, fwd, wdir, p, ok, fov_half):
    """fixations and the shifts between them (>= 30 deg, landing with a 3D point); out of view: the landing more than
    fov_half from the head's forward axis at the WHERE query, 0.3 s before the eye leaves."""
    F = fixations(t, wdir, p, ok)
    shifts = []
    for (a0, a1, da, _), (b0, b1, db, pb) in zip(F[:-1], F[1:]):
        size = float(angle(da, db))
        if size < 30 or b0 - a1 > 0.5 or pb is None:
            continue
        q = min(np.searchsorted(t, a1 - 0.3), len(t) - 1)
        out = bool(angle(fwd[q], pb - eye[q]) > fov_half)
        shifts.append(dict(t=float(a1), size=size, out=out, p=pb, dur=float(b1 - b0)))
    print(f"{len(F)} fixations, {len(shifts)} shifts >= 30 deg ({sum(s['out'] for s in shifts)} landing out of view)")
    return F, shifts


def rank_shifts(shifts, objects, show=15):
    """teaser moments first: out of view, on a named object, a clear (long) landing fixation, a big shift."""
    for s in shifts:
        s["obj"] = landed_object(s["p"], objects)
    rank = sorted(shifts, key=lambda s: -(2 * s["out"] + (s["obj"] is not None) + min(s["dur"], 1) + s["size"] / 90))
    if show:
        print(f"\n{'t (s)':>7} {'size':>6} {'out':>4} {'landing fix':>11}  landed on")
        for s in rank[:show]:
            print(f"{s['t']:7.1f} {s['size']:5.0f}° {'yes' if s['out'] else '':>4} {s['dur']:10.2f}s  "
                  f"{objects[s['obj']][0] if s['obj'] is not None else '-'}")
    return rank


def write_page(path, title, t, eye, fwd, wdir, p, ok, F, shifts, objects, t_open, fov_half, up=None, points=None,
               marker_objects=False, step=3):
    """the viewer: world frame -> three.js (y up, metres, origin at the median head position). up: the world's up
    direction (None: the axis the head moves least along); points: optional (xyz (n, 3), rgb (n, 3) uint8) scene
    points; marker_objects: draw the objects as labelled dots (positions only) instead of boxes; step: viewer frames
    every step-th gaze sample."""
    if up is None:
        up = np.linalg.eigh(np.cov((eye[ok] - eye[ok].mean(0)).T))[1][:, 0]
        ref = points[0] if points is not None else np.array([o[4] for o in objects])
        if (np.median(eye[ok], 0) - np.median(ref, 0)) @ up < 0:
            up = -up
    A = align_up(up)
    o0 = np.median(eye[ok], 0)
    W = lambda x: ((np.asarray(x) - o0) @ A.T).round(4)                 # noqa: E731
    k = np.flatnonzero(ok)[::step]
    if points is not None:
        cy = (points[0] - o0) @ A.T
    else:
        cy = W(np.array([c + R @ (s / 2 * np.array(sg)) for _, _, _, R, c, s in objects
                         for sg in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)]))
    floor = float(np.percentile(cy[:, 1], 1))
    lo, hi = np.percentile(cy[:, [0, 2]], 1, 0), np.percentile(cy[:, [0, 2]], 99, 0)
    cats = sorted({o[1] for o in objects})
    data = dict(
        t=t[k].round(3).tolist(), head=W(eye[k]).tolist(), fwd=(fwd[k] @ A.T).round(4).tolist(),
        gaze=(wdir[k] @ A.T).round(4).tolist(),
        gp=[W(x).tolist() if np.isfinite(x).all() else None for x in p[k]],
        fix=[dict(t0=round(f0, 3), t1=round(f1, 3), p=W(fp).tolist() if fp is not None else None) for f0, f1, _, fp in F],
        shifts=[dict(t=s["t"], size=s["size"], out=s["out"], p=W(s["p"]).tolist(), obj=s["obj"]) for s in shifts],
        objects=[dict(name=n, cat=c, R=(A @ R).round(5).flatten().tolist(), c=W(cc).tolist(), size=sz.round(4).tolist(),
                      color=PALETTE[cats.index(c) % len(PALETTE)]) for n, c, _, R, cc, sz in objects],
        markers=marker_objects, floor=floor, floorCentre=((lo + hi) / 2).round(3).tolist(),
        floorSize=float(np.ceil((hi - lo).max() + 2)), fovHalf=fov_half, pts=None, rgb=None)
    if points is not None:
        data["pts"] = base64.b64encode(cy.astype(np.float32).tobytes()).decode()
        data["rgb"] = base64.b64encode(np.ascontiguousarray(points[1], np.uint8).tobytes()).decode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    k0 = int(np.argmin(np.abs(t[k] - t_open)))
    n = f"{len(objects)} objects" + (f" &middot; {len(cy):,} points" if points is not None else "")
    open(path, "w").write(HTML.replace("__TITLE__", title).replace("__NOBJ__", n)
                          .replace("__NFRAMES_1__", str(len(k) - 1)).replace("__K0__", str(k0))
                          .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"\n{path} ({os.path.getsize(path) / 1e6:.1f} MB, {len(objects)} objects, opens at {t_open:.1f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="ADT sequence folder name")
    ap.add_argument("--src", default="/data/disk2/adt")
    ap.add_argument("--inspect", action="store_true", help="print the ground-truth zip's files and columns and exit")
    ap.add_argument("--list", action="store_true", help="print the best teaser moments and exit")
    ap.add_argument("--t", type=float, help="the moment to open at (s from the start); default: the top --list shift")
    ap.add_argument("--fov-half", type=float, default=45.0, help="half-angle of the view cone (deg)")
    ap.add_argument("--out", default="~/GazeBeyond/figs/adt_scene")
    a = ap.parse_args()
    hits = glob.glob(os.path.join(a.src, a.seq, "*_main_groundtruth.zip"))
    if not hits:
        raise SystemExit(f"no *_main_groundtruth.zip in {os.path.join(a.src, a.seq)} (ls {a.src} | head)")
    gt = hits[0]
    if a.inspect:
        with zipfile.ZipFile(gt) as zf:
            for n in zf.namelist():
                head = zf.open(n).readline()[:300].decode(errors="replace").strip() if n.endswith(".csv") else ""
                print(f"{zf.getinfo(n).file_size / 1e6:8.1f} MB  {n}  {head}")
            js = [n for n in zf.namelist() if n.endswith("instances.json")]
            if js:
                inst = json.loads(zf.read(js[0]))
                print("instances.json, first entry:", json.dumps(next(iter(inst.items())))[:400])
        sys.exit()

    # gaze, head pose, 3D gaze points on the 30 Hz ground-truth timeline
    tus, g_cpf, T_wc, ok, gravity = read_gaze_pose((gt, "eyegaze.csv"), (gt, "aria_trajectory.csv"), T_DEVICE_CPF_ADT)
    depth = _read_csv((gt, "eyegaze.csv"), usecols=["depth_m"])["depth_m"].to_numpy()
    t = (tus - tus[0]) / 1e6
    eye, wdir, fwd = T_wc[:, :3, 3], np.einsum("nij,nj->ni", T_wc[:, :3, :3], g_cpf), T_wc[:, :3, 2]
    ok &= np.isfinite(wdir).all(1)
    p = np.where((ok & (depth > 0))[:, None], eye + depth[:, None] * wdir, np.nan)
    F, shifts = find_shifts(t, eye, fwd, wdir, p, ok, a.fov_half)

    t_ns = lambda s: int(tus[0] * 1000 + s * 1e9)                       # noqa: E731
    objects = read_objects(gt, t_ns(a.t if a.t is not None else t[-1] / 2))
    rank = rank_shifts(shifts, objects)
    if a.list:
        sys.exit()
    t_open = a.t
    if t_open is None:
        t_open = rank[0]["t"] - 0.3 if rank else t[-1] / 2
        objects = read_objects(gt, t_ns(t_open))                        # dynamic objects at that moment
        rank_shifts(shifts, objects, show=0)
    write_page(os.path.join(os.path.abspath(os.path.expanduser(a.out)), f"{a.seq}.html"), a.seq, t, eye, fwd, wdir, p,
               ok, F, shifts, objects, t_open, a.fov_half, up=-gravity if gravity is not None else None)
