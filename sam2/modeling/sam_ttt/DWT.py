import torch
import torch.nn as nn

def resize_tensor(tensor, size):
    """
    使用插值方法将张量调整大小
    """
    resized_tensor = nn.functional.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
    return resized_tensor

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x01 x02是低频信号; x1 x2 x3 x4是高频信号
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh

class extract_high_frequency(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(extract_high_frequency, self).__init__()
        self.dwt = DWT()

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        # 将高频部分调整为 64x64，以匹配 SAM2 的特征图大小
        high_frequency = resize_tensor(hh, (64, 64))
        return high_frequency
