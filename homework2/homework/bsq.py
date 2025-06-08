import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        self.encoder = torch.nn.Linear(embedding_dim, codebook_bits)
        self.decoder = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Linear projection to codebook space
        x_proj = self.encoder(x)                          # shape: (B, h, w, codebook_bits)
        x_proj = torch.nn.functional.normalize(x_proj, dim=-1)  # L2 normalization
        return diff_sign(x_proj)                          # shape: binary (-1/+1)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)  # shape: (B, h, w, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1

class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)
        self.codebook_bits = codebook_bits

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, W, 3) → (B, h, w)
        z = self.encoder(x)                     # (B, h, w, latent_dim)
        return self.bsq.encode_index(z)         # (B, h, w) token indices

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        # (B, h, w) → (B, H, W, 3)
        z = self.bsq.decode_index(x)            # (B, h, w, latent_dim)
        return self.decoder(z)                  # (B, H, W, 3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.bsq.encode(self.encoder(x))  # (B, h, w, codebook_bits)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.bsq.decode(x))  # (B, H, W, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encoder(x)
        b = self.bsq.encode(z)
        x_hat = self.decoder(self.bsq.decode(b))

        # Codebook usage stats
        idx = self.bsq._code_to_index(b)
        cnt = torch.bincount(idx.flatten(), minlength=2 ** self.codebook_bits)
        usage_stats = {
            "cb_0": (cnt == 0).float().mean().detach(),
            "cb_1": (cnt <= 1).float().mean().detach(),
            "cb_5": (cnt <= 5).float().mean().detach(),
        }

        return x_hat, usage_stats
