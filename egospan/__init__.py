"""Gaze Beyond the Frame: forecasting egocentric 3D visual span (Yun et al., NeurIPS 2025) - unofficial codebase.

    egospan.lift        EgoSpanLift (Sec. 3): SLAM keypoints + pose + gaze -> multi-level volumetric spans
    egospan.data        AEA / Ego-Exo4D readers, benchmark building and packing, torch dataset, augmentation
    egospan.models      3D U-Net + causal transformer forecaster (Sec. 4), flat 3D U-Net baseline
    egospan.losses      Dice (Eq. 4) and BCE
    egospan.metrics     IoU / F1 / precision / recall, foveal distance, CC / AUC
    egospan.engine      training and evaluation loops
    egospan.baselines   global prior, persistence, 2D gaze + pose forecast lifting
    egospan.project2d   3D forecast -> 2D gaze anticipation (Sec. 5.3)
"""
from .lift import CHANNELS, LEVELS, EgoSpanLift, LiftConfig

__version__ = "0.1.0"
__all__ = ["CHANNELS", "LEVELS", "EgoSpanLift", "LiftConfig"]
