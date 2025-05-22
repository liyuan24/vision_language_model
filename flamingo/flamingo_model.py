# reference: https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py

import torch
import torch.nn as nn
from einops import repeat, rearrange


class PerceiverAttention(nn.Module):
    def __init__(self, *, latent_dim: int, heads: int = 8, head_dim: int = 64):
        """
        Perceiver Attention where latents are queries and media features are keys and values

        Args:
            latent_dim: dimension of latents
            heads: number of attention heads
            head_dim: dimension of each attention head
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.heads = heads
        self.head_dim = head_dim

        attention_dim = heads * head_dim

        self.to_q = nn.Linear(latent_dim, attention_dim, bias=False)
        self.to_kv = nn.Linear(latent_dim, attention_dim * 2, bias=False)
        self.to_out = nn.Linear(attention_dim, latent_dim)

        self.latent_norm = nn.LayerNorm(latent_dim)
        self.media_norm = nn.LayerNorm(latent_dim)

    def forward(self, latents, x):
        """
        Forward pass of the Perceiver Attention

        Args:
            latents: the latents which are queries and have much smaller sequence length than x [B, t, num_latents, latent_dim]
            x: the media features which are keys and values [B, t, N, latent_dim]
        """

        b, t = x.shape[:2]
        scale = self.head_dim**-0.5

        # pre-norm
        latents = self.latent_norm(latents)
        x = self.media_norm(x)

        # [b, 1, num_latents, heads * head_dim]
        q = self.to_q(latents)
        # [b, t, N + num_latents, latent_dim]
        kv_input = torch.cat((x, latents), dim=-2)
        # [b, t + 1, N, heads * head_dim]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # split heads
        q = rearrange(q, "b t n (h d) -> b h t n d", h=self.heads)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=self.heads)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=self.heads)

        q = q * scale

        # calculate attention scores
        # [b, heads, t, num_latents, N]
        attn_scores = torch.einsum("... i d, ... j d -> ... i j", q, k)

        # stable softmax
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True).detach()
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # weighted sum
        # [b, heads, t, num_latents, head_dim]
        out = torch.einsum("... i j, ... j d -> ... i d", attn_scores, v)

        # merge heads
        out = rearrange(out, "b h t n d -> b t n (h d)", h=self.heads)

        # project to the latent dimension
        # [b, t, num_latents, latent_dim]
        out = self.to_out(out)

        return out


class MLP(nn.Module):
    def __init__(self, dim: int, ff_mult: int = 4):
        super().__init__()

        inner_dim = int(dim * ff_mult)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        """
        Forward pass of the Perceiver MLP

        Args:
            x: the input tensor [B, t, num_latents, latent_dim]
        """

        return self.mlp(x)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        num_latents: int,
        latent_dim: int,
        depth: int,
        heads: int = 8,
        head_dim: int = 64,
        max_num_media_embeddings: int = 45,
        ff_mult: int = 4,
    ):
        """
        Perceiver Resampler

        Args:
            num_latents: number of latent embeddings
            latent_dim: dimension of latent embeddings
            depth: number of layers of Perceiver
            heads: number of cross attention heads
            head_dim: dimension of each cross attention head
            max_num_media_embeddings: maximum number of media embeddings
            ff_mult: the multiplier to calculate the hidden dimension of the MLP
        """
        super().__init__()

        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.max_num_media_embeddings = max_num_media_embeddings
        self.ff_mult = ff_mult

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.media_pos_embeddings = nn.Parameter(
            torch.randn(max_num_media_embeddings, 1, latent_dim)
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            latent_dim=latent_dim, heads=heads, head_dim=head_dim
                        ),
                        MLP(dim=latent_dim, ff_mult=ff_mult),
                    ]
                )
            )

        # post-norm
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        Forward pass of the Perceiver Resampler

        The input tensor will be cross-attended with the latent embeddings to reduce the sequence length of the input tensor.

        Args:
            x: the media features generated by the vision encoder which has the shape
               [B, t, N, latent_dim]

        Returns:
            the compressed media features with the shape [B, T_img, num_latents, latent_dim]
        """

        assert (
            x.ndim == 4
        ), f"The input tensor must have 4 dimensions(B, T_img, N, latent_dim), but got {x.ndim}"
        assert (
            x.shape[1] <= self.max_num_media_embeddings
        ), f"The number of media embeddings must be less than or equal to {self.max_num_media_embeddings}, but got {x.shape[1]}"

        B, t, N, latent_dim = x.shape

        x = x + self.media_pos_embeddings[:t]
        latents = repeat(self.latents, "n d -> b t n d", b=B, t=t)

        for cross_attn, mlp in self.layers:
            latents = latents + cross_attn(latents, x)
            latents = latents + mlp(latents)

        # post-norm
        latents = self.norm(latents)

        return latents


class MaskedCrossAttention(nn.Module):
    """
    The features after the Perceiver Resampler are cross-attended with the text features in the Language Model
    The input of the Language Model is like
      <image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|>
    <image> is a media token and indicates the position of the media in the input text
    The text tokens can only attend to its previous medias. And in Flamingo paper, the text tokens can only attend to the immediate previous media.

    So this is a masked cross-attention.
    """

    def __init__(
        self,
        *,
        dim,
        heads: int = 8,
        head_dim: int = 64,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.only_attend_immediate_media = only_attend_immediate_media

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.text_norm = nn.LayerNorm(dim)

    def forward(
        self, text: torch.Tensor, media: torch.Tensor, media_locations: torch.Tensor
    ):
        """
        Args:
            text: the text features [B, num_text_tokens, dim]
            media: the media features [B, t, num_latents, dim]
            media_locations: the locations of the media in the text [B, num_text_tokens] boolean tensor
        Returns:
            the attended text features [B, num_text_tokens, dim]
        """
        b, t, num_latents = media.shape[:3]
        # pre-norm
        text = self.text_norm(text)

        q = self.to_q(text)
        # flatten the num_media and spatial dimensions of the media
        # so that the media dimension matches the text tensor dimension
        media = rearrange(media, "b t n d -> b (t n) d")
        k, v = self.to_kv(media).chunk(2, dim=-1)

        # split heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        # calculate attention scores
        # [B, heads, num_text_tokens, num_latents * t]
        attn_scores = torch.einsum("... i d, ... j d -> ... i j", q, k)

        # construct the cross-attention mask
        if media_locations is not None:
            # media_locations is a boolean tensor of shape [B, num_text_tokens]
            # it is True for the positions of the media in the text
            # only text tokens after the media can attend to that media
            # [b, num_text_tokens]
            text_times = torch.cumsum(media_locations, dim=-1)
            # [t]
            media_times = torch.arange(t) + 1
            # [1, 1, 1, t * num_latents]
            media_times = repeat(media_times, "j -> 1 1 1 (j m)", m=num_latents)
            # [b, 1, num_text_tokens, 1]
            text_times_expanded = repeat(text_times, "b j -> b 1 j 1")
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            # [b, 1, num_text_tokens, t * num_latents], mask is shared across heads
            mask = mask_op(text_times_expanded, media_times)
            attn_scores = attn_scores.masked_fill(
                ~mask, torch.finfo(attn_scores.dtype).min
            )

        # stable softmax
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True).detach()
        # [b, heads, num_text_tokens, t * num_latents]
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # if only_attend_immediate_media is True, when media is not at the first position, the attention
        # scores for the text tokens before the first media will be 1 / (t * num_latents)
        # since after mask op, they are all -inf
        # so we need to set the attention scores for the text tokens before the first media to 0
        if media_locations is not None and self.only_attend_immediate_media:
            # [b, num_text_tokens]
            text_without_media_mask = text_times == 0
            print(f"text times shape: {text_times.shape}")
            # [b, 1, num_text_tokens, 1]
            text_without_media_mask = repeat(text_without_media_mask, "b n -> b 1 n 1")
            # [b, heads, num_text_tokens, t * num_latents]
            attn_scores = attn_scores.masked_fill(text_without_media_mask, 0.0)

        out = torch.einsum("... i j, ... j d -> ... i d", attn_scores, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class GatedMaskedCrossAttentionBlock(nn.Module):
    """
    Masked cross-attention + MLP
    """

    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        head_dim: int = 64,
        only_attend_immediate_media: bool = True,
        ff_mult: int = 4,
    ):
        super().__init__()

        self.masked_cross_attention = MaskedCrossAttention(
            dim=dim,
            heads=heads,
            head_dim=head_dim,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor(0.0))

        self.mlp = MLP(dim=dim, ff_mult=ff_mult)
        self.mlp_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, x: torch.Tensor, media: torch.Tensor, media_locations: torch.Tensor
    ):
        """
        Args:
            text: the text features [B, num_text_tokens, dim]
            media: the media features [B, t, num_latents, dim]
            media_locations: the locations of the media in the text [B, num_text_tokens] boolean tensor
        Returns:
            the attended text features [B, num_text_tokens, dim]
        """
        out = (
            self.masked_cross_attention(text, media, media_locations)
            * self.attn_gate.tanh()
            + text
        )
        out = self.mlp(out) * self.mlp_gate.tanh() + out
        return out


if __name__ == "__main__":
    perceiver_resampler = PerceiverResampler(num_latents=10, latent_dim=10, depth=2)
    x = torch.randn(3, 5, 100, 10)
    # expected output: [3, 5, 10, 10]
    print(perceiver_resampler(x).shape)

    masked_cross_attention = MaskedCrossAttention(
        dim=16, heads=2, head_dim=64, only_attend_immediate_media=True
    )
    text = torch.randn(2, 8, 16)
    media = torch.randn(2, 3, 3, 16)
    media_locations = torch.tensor(
        [
            [False, False, True, False, True, False, True, False],
            [True, False, False, True, False, True, False, False],
        ]
    )
    print(masked_cross_attention(text, media, media_locations).shape)

    gated_masked_cross_attention_block = GatedMaskedCrossAttentionBlock(
        dim=16, heads=2, head_dim=64, only_attend_immediate_media=True, ff_mult=4
    )
    print(gated_masked_cross_attention_block(text, media, media_locations).shape)
