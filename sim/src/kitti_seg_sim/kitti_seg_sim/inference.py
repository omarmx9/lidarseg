"""Thin wrapper around the trained Cylinder3D model for live inference."""
import torch

# PyTorch 2.6+ defaults torch.load to weights_only=True, which rejects the
# mmengine ConfigDict stored in our own checkpoint. It is a trusted local file,
# so restore the legacy full-unpickle behaviour.
_orig_load = torch.load


def _full_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _full_load


class SegModel:
    """Loads the checkpoint once; predicts per-point labels for a .bin scan."""

    def __init__(self, config, checkpoint, device="cuda:0"):
        from mmdet3d.apis import init_model, inference_segmentor
        self._infer = inference_segmentor
        self.model = init_model(config, checkpoint, device=device)

    def predict(self, bin_path):
        result, _ = self._infer(self.model, bin_path)
        return result.pred_pts_seg.pts_semantic_mask.cpu().numpy().astype("int64")
