from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm

# ----------- Quantization Helpers -----------

def block_quantize_4bit(x, group_size=16):
    assert x.dim() == 1 and x.numel() % group_size == 0
    x = x.view(-1, group_size)
    norm = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + norm) / (2 * norm)
    xq = (x_norm * 15).round().clamp(0, 15).to(torch.uint8)
    packed = (xq[:, 0::2] & 0xF) | ((xq[:, 1::2] & 0xF) << 4)
    return packed.contiguous(), norm.to(torch.float16)

def block_dequantize_4bit(xq, norm):
    norm = norm.to(torch.float32)
    xq_out = torch.empty(xq.shape[0], xq.shape[1] * 2, dtype=torch.float32, device=xq.device)
    xq_out[:, 0::2] = (xq & 0xF).to(torch.float32)
    xq_out[:, 1::2] = ((xq >> 4) & 0xF).to(torch.float32)
    x_norm = xq_out / 15
    return (x_norm * 2 * norm) - norm

def block_quantize_2bit(x, group_size=16):
    assert x.dim() == 1 and x.numel() % group_size == 0
    x = x.view(-1, group_size)
    norm = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + norm) / (2 * norm)
    xq = (x_norm * 3).round().clamp(0, 3).to(torch.uint8)
    packed = (
        (xq[:, 0::4] << 0) |
        (xq[:, 1::4] << 2) |
        (xq[:, 2::4] << 4) |
        (xq[:, 3::4] << 6)
    )
    return packed.contiguous(), norm.to(torch.float16)

def block_dequantize_2bit(xq, norm):
    norm = norm.to(torch.float32)
    xq_out = torch.empty(xq.shape[0], xq.shape[1] * 4, dtype=torch.float32, device=xq.device)
    for i in range(4):
        xq_out[:, i::4] = ((xq >> (2 * i)) & 0x03).to(torch.float32)
    x_norm = xq_out / 3
    return (x_norm * 2 * norm) - norm

# ----------- Linear Layers -----------

class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, group_size=16):
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None
        self.register_buffer("weight_q4", torch.zeros(out_features * in_features // group_size, group_size // 2, dtype=torch.uint8), persistent=False)
        self.register_buffer("weight_norm", torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16), persistent=False)
        self._register_load_state_dict_pre_hook(self._load, with_module=True)

    @staticmethod
    def _load(module, state_dict, prefix, *_):
        if f"{prefix}weight" in state_dict:
            w = state_dict.pop(f"{prefix}weight")
            w_flat = w.contiguous().view(-1)
            w_q, w_n = block_quantize_4bit(w_flat, module._group_size)
            module.weight_q4.copy_(w_q)
            module.weight_norm.copy_(w_n)

    def forward(self, x):
        with torch.no_grad():
            w = block_dequantize_4bit(self.weight_q4, self.weight_norm).view(self._shape)
        return torch.nn.functional.linear(x, w, self.bias)

class Linear2Bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, group_size=16):
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None
        self.register_buffer("weight_q2", torch.zeros(out_features * in_features // group_size, group_size // 4, dtype=torch.uint8), persistent=False)
        self.register_buffer("weight_norm", torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16), persistent=False)
        self._register_load_state_dict_pre_hook(self._load, with_module=True)

    @staticmethod
    def _load(module, state_dict, prefix, *_):
        if f"{prefix}weight" in state_dict:
            w = state_dict.pop(f"{prefix}weight")
            w_flat = w.contiguous().view(-1)
            w_q, w_n = block_quantize_2bit(w_flat, module._group_size)
            module.weight_q2.copy_(w_q)
            module.weight_norm.copy_(w_n)

    def forward(self, x):
        with torch.no_grad():
            w = block_dequantize_2bit(self.weight_q2, self.weight_norm).view(self._shape)
        return torch.nn.functional.linear(x, w, self.bias)

# ----------- Model -----------

class LowerPrecisionBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, use_2bit=False):
            super().__init__()
            Linear = Linear2Bit if use_2bit else Linear4Bit
            self.model = torch.nn.Sequential(
                Linear(channels, channels),
                torch.nn.ReLU(),
                Linear(channels, channels),
                torch.nn.ReLU(),
                Linear(channels, channels),
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, use_2bit=True),  # Only last block is 2-bit
        )

    def forward(self, x):
        return self.model(x)

def load(path: Path | None):
    net = LowerPrecisionBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net