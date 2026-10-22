from .forecaster import FlatUNet3D, SpanForecaster, count_parameters

MODELS = {"egospanlift": SpanForecaster, "flat_unet": FlatUNet3D}


def build_model(mcfg, n_out, num_steps, in_channels=5, cue_dim=0, video_dim=0, copy_bins=1, copy_channels=None):
    """mcfg: egospan.config.ModelConfig; cue_dim > 0 adds the gaze-path branch (SpanForecaster only)."""
    if mcfg.name not in MODELS:
        raise ValueError(f"model {mcfg.name!r}; choose from {sorted(MODELS)}")
    kw = dict(in_channels=in_channels, n_out=n_out, num_steps=num_steps, f_maps=tuple(mcfg.f_maps),
              layer_order=mcfg.layer_order, num_groups=mcfg.num_groups, dropout=mcfg.dropout, upsample=mcfg.upsample)
    if mcfg.name == "egospanlift":
        kw.update(token_pool=mcfg.token_pool, tf_layers=mcfg.tf_layers, tf_heads=mcfg.tf_heads,
                  tf_ff_mult=mcfg.tf_ff_mult, tf_dropout=mcfg.tf_dropout, head=mcfg.head, skip=mcfg.skip,
                  bottom=mcfg.bottom, cue_dim=cue_dim, cue_hidden=mcfg.cue_hidden, cue_layers=mcfg.cue_layers,
                  video_dim=video_dim, video_channels=mcfg.video_channels, span_attention=mcfg.span_attention,
                  sa_level=mcfg.sa_level, sa_dim=mcfg.sa_dim, sa_layers=mcfg.sa_layers, sa_heads=mcfg.sa_heads,
                  sa_ff_mult=mcfg.sa_ff_mult, sa_dropout=mcfg.sa_dropout, copy_path=mcfg.copy_path,
                  copy_bins=copy_bins, copy_channels=copy_channels)
    elif cue_dim or video_dim or mcfg.span_attention != "none" or mcfg.copy_path != "none":
        raise ValueError(f"model {mcfg.name!r} has no gaze-path / camera / span-attention branch; use egospanlift")
    return MODELS[mcfg.name](**kw)


__all__ = ["SpanForecaster", "FlatUNet3D", "build_model", "count_parameters"]
