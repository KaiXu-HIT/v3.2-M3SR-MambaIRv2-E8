"""TorchScript adapter for the SAME estimator used for the Depth cache.

The exported module must include its original resize/normalize/inference/
restore-to-LR transforms and accept RGB FP32 [0,1] BCHW; return [B,1,H,W].
Do not export a different estimator solely to obtain a cheaper cost number.
For an existing eager estimator use the equivalent factory interface instead.
"""
from pathlib import Path
import torch


class ScriptedDepth:
    def __init__(self, checkpoint, device):
        self.device = device
        self.name = Path(checkpoint).stem
        self.model = torch.jit.load(checkpoint, map_location=device).float().eval()

    def preprocess(self, rgb):
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(
            self.device, dtype=torch.float32) / 255.

    def predict(self, value):
        return self.model(value)


def build(checkpoint, device):
    return ScriptedDepth(checkpoint, device)
