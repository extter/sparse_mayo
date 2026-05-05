
import torch
import torch.nn as nn



class MixedLoss(nn.Module):
    """L1 + SSIM-proxy (MS-SSIM via pytorch_msssim if available, else L1 only)."""
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        try:
            from pytorch_msssim import SSIM
            self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
            self.use_ssim = True
            print("[Loss] Using L1 + SSIM")
        except ImportError:
            self.use_ssim = False
            print("[Loss] pytorch_msssim not found — using L1 only")
 
    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        if self.use_ssim:
            ssim_val = self.ssim_loss(pred, target)
            return self.alpha * l1 + (1 - self.alpha) * (1 - ssim_val)
        return l1