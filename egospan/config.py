"""Experiment configuration: YAML file -> nested dataclasses, with `section.key=value` overrides from the CLI."""
import dataclasses
import json
from dataclasses import dataclass, field

import yaml

from .data.build import PackConfig, WindowConfig
from .lift import LEVELS, LiftConfig


@dataclass
class BuildConfig:
    """How the benchmark is curated from raw recordings (scripts/build_dataset.py)."""
    lift: LiftConfig = field(default_factory=LiftConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    pack: PackConfig = field(default_factory=PackConfig)


@dataclass
class DataConfig:
    root: str = "data/fovs_aria"            # directory with all.npz + splits.npz
    fold: int | None = None                 # location-wise CV fold (None = the default split)
    horizon: int | None = None              # future units in the target union (None = all; Fig. 4-b varies it)
    levels: list = field(default_factory=lambda: list(LEVELS))   # one level -> single-task model
    drop_span_input: bool = False           # "w/o previous span" ablation
    gaze_paths: bool = False                # add the past gaze / head path branch (<root>/gaze_all.npz from
                                            # scripts/build_gaze_paths.py, or our builds' stored poses and gaze)
    video_features: bool = False            # lifted camera (V-JEPA) features, <root>/video_all.npz from
                                            # scripts/attach_video_features.py (anchor-CPF grids only)
    crop_front: bool = False                # model sees / predicts only the front half (R x R x R/2); val and test
                                            # are still scored on the full grid (back half predicted empty).
                                            # Anchor-CPF grids only (converted lab data)
    train_categories: list | None = None    # keep only these categories in train/val (Fig. 4-a cross-category)
    test_categories: list | None = None     # ... and these in test (None = all)
    augment: bool = True
    max_shift: int = 2
    num_workers: int = 4


@dataclass
class ModelConfig:
    name: str = "egospanlift"               # egospanlift | flat_unet
    f_maps: list = field(default_factory=lambda: [128, 256, 512, 1024])
    layer_order: str = "cbrd"               # Conv-BatchNorm-ReLU-Dropout (pytorch-3dunet notation)
    num_groups: int = 8
    dropout: float = 0.1
    upsample: str = "nearest"               # nearest | trilinear | deconv
    token_pool: str = "avg"                 # avg | max  [unspecified in the paper]
    tf_layers: int = 4                      # [unspecified in the paper]
    tf_heads: int = 8                       # [unspecified in the paper]
    tf_ff_mult: int = 4
    tf_dropout: float = 0.1
    head: str = "union"                     # union | learned | none (w/o global embedding)
    skip: str = "head"                      # head | last | concat
    video_channels: int = 16                # camera features: 1x1x1-conv width (as the lab's `vid`)
    cue_hidden: int = 256                   # gaze-path branch: GRU width and depth (used with data.gaze_paths)
    cue_layers: int = 2
    bottom: str = "add"                     # add | conv: how the embedding joins the deepest skip
    span_attention: str = "none"            # none | attn | bias: cells of encoder level sa_level attend to each other
    sa_level: int = 2                       # (bias: + learned bias from each cell's past spans); our addition
    sa_dim: int = 256
    sa_layers: int = 2
    sa_heads: int = 4
    sa_ff_mult: int = 4
    sa_dropout: float = 0.1
    copy_path: str = "none"                 # none | add | gate: last observed second's spans straight to the output


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    optimizer: str = "adam"                 # adam | adamw
    weight_decay: float = 0.0
    scheduler: str = "none"                 # none | onecycle | cosine
    loss: str = "dice"                      # dice | bce | dice+bce | dice+dist
    dist_weight: float = 0.1                # dice+dist: weight of the distance term (distances in voxels)
    dist_levels: list = field(default_factory=lambda: ["central", "foveal"])
    dist_mode: str = "oneway"               # oneway (are guesses close) | bidirectional (also: is the target covered)
    dist_tau: float = 1.0                   # bidirectional: softness (voxels) of the nearest-predicted-mass search
    amp: bool = True                        # bf16 autocast on CUDA
    grad_clip: float | None = None
    init: str | None = None                 # start from a trained run (its folder or a .pt file): every weight whose
                                            # name and shape match is loaded (fine-tuning, e.g. on attention shifts)
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs/fovs_aria/egospanlift_s0"
    select_threshold: float = 0.5           # model selection: best val mean IoU at this threshold
    log_every: int = 50                     # used only when progress=false
    progress: bool = True                   # tqdm bars for train / val / test; false for nohup logs


@dataclass
class EvalConfig:
    threshold: str | float = 0.5            # a number, or "tune" (per-level on val)
    batch_size: int = 32
    saliency: bool = True
    distance: bool = True                   # foveal distance in cm (R^3 scoring only)
    space: str = "cartesian"                # spherical benchmarks: score on the paper's R^3 voxels (cartesian, mapped
                                            # back) or on the spherical cells themselves (native)


@dataclass
class Config:
    build: BuildConfig = field(default_factory=BuildConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _from_dict(cls, d):
    if d is None:
        return cls()
    kw = {}
    hints = {f.name: f for f in dataclasses.fields(cls)}
    for k, v in d.items():
        if k not in hints:
            raise KeyError(f"unknown config key {cls.__name__}.{k}")
        sub = hints[k].default_factory if hints[k].default_factory is not dataclasses.MISSING else None
        if sub is not None and dataclasses.is_dataclass(sub()) and isinstance(v, dict):
            kw[k] = _from_dict(type(sub()), v)
        elif isinstance(v, list) and isinstance(getattr(cls(), k), tuple):
            kw[k] = tuple(v)
        else:
            kw[k] = v
    return cls(**kw)


def _parse_value(s):
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def apply_overrides(d, overrides):
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return d


def load_config(path=None, overrides=None):
    d = {}
    if path:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    return _from_dict(Config, apply_overrides(d, overrides))


def to_dict(cfg):
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
