# reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
from typing import Union, Tuple

import torch
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        self.heads = heads
        inner_dim = heads * head_dim
        self.scale = head_dim**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.to_qkv(x)
        q, k, v = x.chunk(3, dim=-1)
        q = q * self.scale
        # split heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        # stable softmax
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True).detach()
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.dropout(attn_scores)
        # [b, h, seq_len, head_dim]
        out = torch.einsum("b h i j, b h j d -> b h i d", attn_scores, v)
        # [b, seq_len, inner_dim]
        out = rearrange(out, "b h n d -> b n (h d)")
        # [b, seq_len, dim]
        out = self.to_out(out)
        # post norm
        return self.norm(out)


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.mlp(x)
        # post norm
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, *, dim, num_layers, heads, head_dim, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads, head_dim, dropout),
                        MLP(dim, mlp_dim, dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        num_classes: int,
        dim: int,
        num_layers: int,
        heads: int = 8,
        head_dim: int = 64,
        mlp_dim: int,
        pool: str = "cls",
        channels: int = 3,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        """
        Args:
            image_size: the size of the image
            patch_size: the size of the patch
            num_classes: the number of classes for the classification head
            dim: the hidden dimension of the transformer
            num_layers: the number of layers of the transformer blocks
            heads: the number of attention heads
            head_dim: the dimension of each attention head
            mlp_dim: the dimension of the MLP
            pool: the pooling operation to use, either "cls" or "mean"
            channels: the number of channels in the image
            dropout: the dropout rate for the attention
            emb_dropout: the dropout rate for the embedding
        """
        super().__init__()
        assert (
            image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0
        ), "Image dimensions must be divisible by the patch size"
        assert pool in {"cls", "mean"}, "Pool must be either cls or mean"
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.num_classes = num_classes
        self.dim = dim
        self.num_layers = num_layers
        self.heads = heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.pool = pool
        self.channels = channels
        self.attn_dropout = dropout
        self.emb_dropout = emb_dropout
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, dim))

        num_patches = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )
        patch_dim = channels * patch_size[0] * patch_size[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size[0],
                p2=patch_size[1],
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim=dim,
            num_layers=num_layers,
            heads=heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.pool = pool
        self.classification_head = nn.Linear(dim, num_classes)
        
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input image of shape (batch_size, channels, height, width)
        Returns:
            the tensor of shape (batch_size, num_patches, patch_dim)
        """
        assert x.shape[2] % self.patch_size[0] == 0 and x.shape[3] % self.patch_size[1] == 0
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // self.patch_size[0], self.patch_size[0], w // self.patch_size[1], self.patch_size[1])
        # axis reordering
        x = torch.einsum("b c h p w q -> b h w c p q", x)
        x = x.reshape(b, h // self.patch_size[0], w // self.patch_size[1], -1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input image of shape (batch_size, channels, height, width)
        Returns:
            the output of the model
        """
        # [b, num_patches, dim]
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # [b, 1, dim]
        cls_embedding = repeat(self.cls_embedding, "1 1 d -> b 1 d", b=b)
        # [b, n + 1, dim]
        x = torch.cat((cls_embedding, x), dim=1)
        # [b, n + 1, dim]
        x += self.pos_embedding[:, : n + 1]
        x = self.dropout(x)
        # [b, n + 1, dim]
        x = self.transformer(x)
        # [b, dim]
        x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)
        # [b, num_classes]
        return self.classification_head(x)


if __name__ == "__main__":
    vit = ViT(
        image_size=(224, 224),
        patch_size=(16, 16),
        num_classes=1000,
        dim=64,
        num_layers=2,
        heads=16,
        head_dim=64,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1,
        pool="mean",
    )
    x = torch.randn(2, 3, 224, 224)
    out = vit(x)
    print(out.shape)
    x_patched = vit.patchify(x)
    print(x_patched.shape)
