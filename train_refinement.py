import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from heapq import heappush, heappop
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional

# Quadformer / saliency quadtree tokenization in this file follows the same
# high-level idea as the official implementation in:
# https://github.com/TomerRonen34/mixed-resolution-vit
# That repo ships batchified GPU variants (dict-lookup, tensor-lookup, Z-curve);
# here we use a heap-based splitter for clarity. Default spatial size matches
# recon + uncertainty maps at 320×320.
#
# `BatchedQuadtreeTokenizer` matches the same split rule but computes patch
# scores for the whole batch via avg_pool2d pyramids on device (see mixed-resolution-vit
# style batchified scoring); the heap still runs per image in Python.

# -----------------------------------------------------------------------------
# Minimal placeholder for data consistency.
# For the Quadformer/TransUNet-style refinement, you can train the residual
# predictor first; DC can be enabled later.
# -----------------------------------------------------------------------------
class DataConsistency(nn.Module):
    def forward(
        self,
        x_corrected: torch.Tensor,
        k_sampled: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return x_corrected

# TokenMeta is a dataclass that holds metadata for a token, including its top-left row and column position, cell span, patch size, and score.
@dataclass
class TokenMeta:
    tl_row:     int     # The top-left row position of the token in the table.
    tl_col:     int     # The top-left column position of the token in the table.
    cell_span:  int     # The number of fine-grid cells the token spans.
    patch_size: int     # The size of the patch containing the token.
    score:      float   # The confidence score of the token.

# Named index constants for readable access
TL_ROW     = 0
TL_COL     = 1
CELL_SPAN  = 2
PATCH_SIZE = 3
SCORE      = 4

@dataclass
class ModelConfig:
    # image related configs
    image_size: int = 320  # recon + UQ map spatial size (H, W)
    max_patch_size: int = 32 # initial patch size for quadtree (must divide image_size)
    min_patch_size: int = 4  # minimum patch size for quadtree (must divide max_patch_size)
    num_scales: int = 4     # number of patch scales (for scale embedding)
    finest_grid: int = image_size // min_patch_size # number of cells in the finest grid
    
    # transformer related configs
    num_tokens: int = 1024 # number of tokens in one batch (one image) (sequence length)
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

    # CNN related configs
    base_ch: int = 32 # base channel count for CNN encoder/decoder
    in_chans: int = 2 # input channels for CNN encoder (recon + uncertainty map)
    # Total downsampling between input image and CNN feature map.
    # CNNEncoder applies AvgPool2d(2) twice => factor 4.
    downsample_factor: int = 4

#-------------------------------------------------------------------------------#
# Patch Scorer
#-------------------------------------------------------------------------------#

# abstract base class
class PatchScorer(ABC):
    @abstractmethod
    def score(self, region: np.ndarray) -> float:
        pass
    def __call__(self, region: np.ndarray) -> float:
        return self.score(region)

# baseline scorer that computes the mean of the region as the score
class UncertaintyPatchScorer(PatchScorer):
    def score(self, region: np.ndarray) -> float:
        return float(region.mean())

#-------------------------------------------------------------------------------#
# QuadTree Tokenizer
#-------------------------------------------------------------------------------#

class QuadNode:
    __slots__ = ("row", "col", "patch_size", "score", "is_leaf")
    def __init__(self, row, col, patch_size, score, min_patch_size):
        self.row, self.col  = row, col                          # The top-left position of the patch in the table.
        self.patch_size     = patch_size                        # The size of the patch.
        self.score          = score                             # The confidence score of the patch.
        self.is_leaf        = (patch_size == min_patch_size)    # Whether the node is a leaf node (i.e., cannot be further subdivided).
    
    # The __lt__ method is defined to allow QuadNode instances to be compared based on their scores, which is necessary for maintaining the heap property in the priority queue.
    def __lt__(self, other):
        return self.score > other.score


class QuadtreeTokenizer:
    def __init__(self, config):
        assert config.image_size % config.max_patch_size == 0             # to ensure that the initial grid of patches fits perfectly within the image dimensions.
        assert config.max_patch_size % config.min_patch_size == 0         # to ensure that the patch sizes are compatible.
        self.image_size     = config.image_size                    # The size of the input image (assumed to be square).
        self.max_patch_size = config.max_patch_size                # The maximum size of the patches to be considered during the quadtree decomposition.
        self.min_patch_size = config.min_patch_size                # The minimum size of the patches to be considered during the quadtree decomposition.
        self.num_tokens     = config.num_tokens                    # The number of tokens to be generated.
        self.finest_grid    = self.image_size // self.min_patch_size  # The number of cells in the finest grid

    def _score_patch(self, scorer, uq_map, row, col, patch_size):
        return scorer(uq_map[row:row+patch_size, col:col+patch_size])

    def run_quadtree(self, uq_map: np.ndarray, scorer: PatchScorer) -> torch.Tensor:
        # maintaining leaves so that we dont add them to heap (can't be subdivided) and we can add them to the final token list at the end
        heap, leaves = [], []

        # Initialize the heap with the scores of the initial patches (the largest patches defined by max_patch_size).
        for r in range(0, self.image_size, self.max_patch_size):
            for c in range(0, self.image_size, self.max_patch_size):
                s = self._score_patch(scorer, uq_map, r, c, self.max_patch_size)
                heappush(heap, QuadNode(r, c, self.max_patch_size, s,
                                        self.min_patch_size))
        
        # Iteratively subdivide the patches with the highest scores until we have enough tokens or there are no more patches to subdivide.
        while (len(heap) + len(leaves)) < self.num_tokens:
            if not heap:
                break
            node = heappop(heap)
            child_size = node.patch_size // 2
            for dr in [0, child_size]:
                for dc in [0, child_size]:
                    cr, cc = node.row + dr, node.col + dc
                    s = self._score_patch(scorer, uq_map, cr, cc, child_size)
                    child = QuadNode(cr, cc, child_size, s, self.min_patch_size)
                    if child.is_leaf:
                        leaves.append(child)
                    else:
                        heappush(heap, child)
        
        all_nodes = list(heap) + leaves

        # Build [T_actual, 5] tensor
        rows = [
            [
                n.row  // self.min_patch_size,   # tl_row
                n.col  // self.min_patch_size,   # tl_col
                n.patch_size // self.min_patch_size,  # cell_span
                n.patch_size,                    # patch_size
                n.score,                         # score
            ]
            for n in all_nodes
        ]
        meta_tensor = torch.tensor(rows, dtype=torch.float32)  # [T_actual, 5]
        return _pad_truncate_meta(meta_tensor, self.num_tokens)


def _pad_truncate_meta(meta_tensor: torch.Tensor, num_tokens: int) -> torch.Tensor:
    T_actual = meta_tensor.shape[0]
    if T_actual < num_tokens:
        pad = torch.zeros(
            num_tokens - T_actual, 5, dtype=meta_tensor.dtype, device=meta_tensor.device
        )
        return torch.cat([meta_tensor, pad], dim=0)
    return meta_tensor[:num_tokens]


class BatchedQuadtreeTokenizer:
    """
    Same saliency quadtree as `QuadtreeTokenizer`, with patch scores = mean(uq)
    on each patch (equivalent to `UncertaintyPatchScorer`).

    All patch means for every scale are computed in one batched `avg_pool2d` per
    level on `uq_map.device` (CUDA/MPS/CPU), then each image runs the usual heap
    splits using numpy views of those pools (one device→host copy of small tensors).
    """

    def __init__(self, config):
        assert config.image_size % config.max_patch_size == 0
        assert config.max_patch_size % config.min_patch_size == 0
        self.image_size = config.image_size
        self.max_patch_size = config.max_patch_size
        self.min_patch_size = config.min_patch_size
        self.num_tokens = config.num_tokens

    def run_quadtree(self, uq_map: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        uq_map : [B, 1, H, W] or [B, H, W], H = W = image_size

        Returns
        -------
        metas : [B, T, 5] float32 on the same device as uq_map
        """
        if uq_map.dim() == 3:
            uq_map = uq_map.unsqueeze(1)
        B, _, H, W = uq_map.shape
        assert H == W == self.image_size, (H, W, self.image_size)
        device = uq_map.device

        pools: dict[int, torch.Tensor] = {}
        ps = self.max_patch_size
        while ps >= self.min_patch_size:
            pools[ps] = F.avg_pool2d(uq_map, kernel_size=ps, stride=ps)
            ps //= 2

        pools_np = {k: v.squeeze(1).detach().cpu().numpy() for k, v in pools.items()}

        rows_per_batch: list[torch.Tensor] = []
        for b in range(B):

            def score_at(r: int, c: int, psize: int) -> float:
                return float(pools_np[psize][b, r // psize, c // psize])

            heap: list[QuadNode] = []
            leaves: list[QuadNode] = []
            for r in range(0, self.image_size, self.max_patch_size):
                for c in range(0, self.image_size, self.max_patch_size):
                    s = score_at(r, c, self.max_patch_size)
                    heappush(
                        heap,
                        QuadNode(r, c, self.max_patch_size, s, self.min_patch_size),
                    )
            while (len(heap) + len(leaves)) < self.num_tokens:
                if not heap:
                    break
                node = heappop(heap)
                child_size = node.patch_size // 2
                for dr in (0, child_size):
                    for dc in (0, child_size):
                        cr, cc = node.row + dr, node.col + dc
                        s = score_at(cr, cc, child_size)
                        child = QuadNode(cr, cc, child_size, s, self.min_patch_size)
                        if child.is_leaf:
                            leaves.append(child)
                        else:
                            heappush(heap, child)

            all_nodes = list(heap) + leaves
            rows = [
                [
                    n.row // self.min_patch_size,
                    n.col // self.min_patch_size,
                    n.patch_size // self.min_patch_size,
                    n.patch_size,
                    n.score,
                ]
                for n in all_nodes
            ]
            meta_tensor = torch.tensor(rows, dtype=torch.float32)
            rows_per_batch.append(_pad_truncate_meta(meta_tensor, self.num_tokens))

        return torch.stack(rows_per_batch, dim=0).to(device=device)

#-------------------------------------------------------------------------------#
# CNN Encoder (We need this to add skip connections to the CNN decoder)
#-------------------------------------------------------------------------------#

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(approximate='tanh'),
        )
    def forward(self, x):
        return self.block(x)

class CNNEncoder(nn.Module):
    """
    Shared CNN encoder (shared between the encoder and the token embedder).
    - Produces skip features at H/2, H/4, H/8 for the decoder.
    - Its final feature map is also cropped per-patch for the token embedder.
    in_chans = 2  (MRI reconstruction + uncertainty map)
    """
    def __init__(self, config):
        super().__init__()
        self.enc1 = ConvBlock(config.in_chans,    config.base_ch)        # → [B, 32, H, W]
        self.enc2 = ConvBlock(config.base_ch,     config.base_ch * 2)    # → [B, 64,  H/2, W/2]
        self.enc3 = ConvBlock(config.base_ch * 2, config.base_ch * 4)    # → [B, 128, H/4, W/4]
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        s1 = self.enc1(x)                                  # [B, 32,  H,   W  ]
        s2 = self.enc2(self.pool(s1))                      # [B, 64,  H/2, W/2]
        s3 = self.enc3(self.pool(s2))                      # [B, 128, H/4, W/4]
        return s3, [s1, s2, s3]                            # (final_feat, skip_list)

#-------------------------------------------------------------------------------#
# PATCH EMBEDDING
#-------------------------------------------------------------------------------#

# runs before PatchEmbedder, produces fixed-size tensor
def extract_patches(
    x_in:           torch.Tensor,           # [B, C_in, H, W]
    metas:          torch.Tensor,           # [B, T, 5]
    scale_to_idx:   dict[int, int],
    config:         ModelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    patches  : [B, T, C_in, min_patch_size, min_patch_size]
    cx       : [B, T]   float — patch center col in finest-grid coords
    cy       : [B, T]   float — patch center row in finest-grid coords
    scale_idx: [B, T]   long  — scale level index per token
    """
    B, C, H, W = x_in.shape
    device = x_in.device

    patches   = torch.zeros(B, config.num_tokens, C, config.min_patch_size, config.min_patch_size, device=device)
    cx        = metas[:, :, TL_COL] + metas[:, :, CELL_SPAN] / 2.0   # [B, T]
    cy        = metas[:, :, TL_ROW] + metas[:, :, CELL_SPAN] / 2.0   # [B, T]
    scale_idx = torch.zeros(B, config.num_tokens, dtype=torch.long, device=device)

    stride = config.min_patch_size // config.downsample_factor

    for b in range(B):
        for t in range(config.num_tokens):
            tlr = int(metas[b, t, TL_ROW].item())
            tlc = int(metas[b, t, TL_COL].item())
            p_px = int(metas[b, t, PATCH_SIZE].item())  # patch size in input-image pixels

            if p_px == 0:
                continue  # padded token

            # Convert pixel-space quadtree coordinates -> feature-map coordinates.
            r_feat = tlr * stride
            c_feat = tlc * stride
            p_feat = p_px // config.downsample_factor

            crop = x_in[b:b+1, :, r_feat:r_feat + p_feat, c_feat:c_feat + p_feat]

            # Normalize all tokens to a fixed patch size for the embedder.
            patches[b, t] = F.interpolate(
                crop,
                size=(config.min_patch_size, config.min_patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            # scale_to_idx expects pixel-space patch sizes.
            scale_idx[b, t] = scale_to_idx[p_px]

    return patches, cx, cy, scale_idx

# Patch Embedder for the transformer encoder tokens (with 2D sin/cos PE and scale embedding)
class PatchEmbedder(nn.Module):
    """
    Output: [B, T, n_embed]
    """
    def __init__(self, config):
        super().__init__()
        self.n_embed    = config.n_embd
        self.min_patch_size = config.min_patch_size

        self.cnn = nn.Sequential(
            # Tokens are embedded from CNN feature patches.
            # CNNEncoder outputs [B, base_ch*4, H/4, W/4] as its final feature map.
            nn.Conv2d(config.base_ch * 4, config.base_ch, kernel_size=3, padding=1),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(config.base_ch, config.base_ch * 2, kernel_size=3, padding=1),
            nn.GELU(approximate='tanh'),
            nn.AdaptiveAvgPool2d((4, 4)),   # always → [B*T, 64, 4, 4]
            nn.Flatten(),                   # → [B*T, 1024]
        )
        self.proj        = nn.Linear(64 * 16, config.n_embd)
        self.scale_embed = nn.Embedding(config.num_scales, config.n_embd)

    def _sincos_2d_batch(self, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        """
        Vectorized 2D sinusoidal PE for all tokens at once.
        cx, cy: [N]  → output: [N, n_embed]
        """
        d     = self.n_embed // 4
        omega = 1.0 / (10000 ** (
            torch.arange(d, dtype=torch.float32, device=cx.device) / d
        ))                                                      # [d]
        # outer products: [N, 1] × [1, d] → [N, d]
        return torch.cat([
            torch.sin(cx.unsqueeze(1) * omega.unsqueeze(0)),    # [N, d]
            torch.cos(cx.unsqueeze(1) * omega.unsqueeze(0)),    # [N, d]
            torch.sin(cy.unsqueeze(1) * omega.unsqueeze(0)),    # [N, d]
            torch.cos(cy.unsqueeze(1) * omega.unsqueeze(0)),    # [N, d]
        ], dim=1)                                               # [N, n_embed]

    def forward(
        self,
        patches:   torch.Tensor,   # [B, T, C_in, min_patch_size, min_patch_size]
        cx:        torch.Tensor,   # [B, T]
        cy:        torch.Tensor,   # [B, T]
        scale_idx: torch.Tensor,   # [B, T]  long
    ) -> torch.Tensor:

        B, T, C, P, _ = patches.shape

        # ── Single CNN forward pass over all B×T patches ──────────────────────
        x = patches.view(B * T, C, P, P)        # [B*T, C, P, P]
        x = self.cnn(x)                          # [B*T, 1024]
        x = self.proj(x)                         # [B*T, n_embed]

        # ── Batched 2D sinusoidal PE ───────────────────────────────────────────
        x = x + self._sincos_2d_batch(
            cx.view(B * T),
            cy.view(B * T),
        )                                        # [B*T, n_embed]

        # ── Learned scale embedding ────────────────────────────────────────────
        x = x + self.scale_embed(
            scale_idx.view(B * T)
        )                                        # [B*T, n_embed]

        return x.view(B, T, self.n_embed)        # [B, T, n_embed]

# function to restructure the token embeddings into a spatial feature map for the CNN decoder
def spatial_scatter(tokens: torch.Tensor, metas: torch.Tensor, config) -> torch.Tensor:
    """
    tokens:    [B, T, n_embed]
    Returns:   [B, n_embed, finest_grid, finest_grid]
    Each coarser token's embedding is tiled across all its finest-grid cells.
    """
    B      = tokens.shape[0]
    device = tokens.device
    fmap   = torch.zeros(B, config.n_embd, config.finest_grid, config.finest_grid, device=device)
    for b in range(B):
        for t in range(tokens.shape[1]):
            p = int(metas[b, t, PATCH_SIZE].item())
            if p == 0:
                continue   # padded token
            r    = int(metas[b, t, TL_ROW].item())
            c    = int(metas[b, t, TL_COL].item())
            span = int(metas[b, t, CELL_SPAN].item())
            fmap[b, :, r:r+span, c:c+span] = \
                tokens[b, t].view(config.n_embd, 1, 1).expand(-1, span, span)
    return fmap   # [B, n_embed, G, G]

#-------------------------------------------------------------------------------#
# CNN Decoder
#-------------------------------------------------------------------------------#

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CNNDecoder(nn.Module):
    """
    Output: residual map [B, 1, H, W]
    """
    def __init__(self, config):
        super().__init__()
        # fmap is at feature-map resolution (H/4, W/4). We upsample to (H, W).
        self.bottleneck = nn.Conv2d(config.n_embd, config.base_ch * 4, kernel_size=1)

        # We have two upsampling steps: (H/4) -> (H/2) -> (H)
        self.up1 = UpBlock(config.base_ch * 4, config.base_ch * 2, config.base_ch * 2)  # G -> 2G + s2
        self.up2 = UpBlock(config.base_ch * 2, config.base_ch, config.base_ch)              # 2G -> 4G + s1

        # Final 1×1 conv → residual
        self.head = nn.Conv2d(config.base_ch, 1, kernel_size=1)

    def forward(self, fmap: torch.Tensor,
                skips: list[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3 = skips   # [B,32,H,W], [B,64,H/2,W/2], [B,128,H/4,W/4]
        x = self.bottleneck(fmap)   # [B, 128, G, G]
        x = self.up1(x, s2)         # [B, 64,  G*2, G*2]  = [B, 64, H/2, H/2]
        x = self.up2(x, s1)         # [B, 32,  G*4, G*4]  = [B, 32, H, H]
        return self.head(x)         # [B, 1,   H,   W]

#-------------------------------------------------------------------------------#
# Transformer Encoder
#-------------------------------------------------------------------------------#

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        """
        x: (B, T, C)
        attn_mask: (B, T) binary mask (1 = valid token, 0 = pad). Used to mask keys in attention.
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        additive_mask = None
        if attn_mask is not None:
            valid = attn_mask.bool()
            neg_inf = torch.finfo(q.dtype).min
            # (B, 1, 1, T) additive mask; broadcast across queries and heads.
            additive_mask = torch.where(
                valid[:, None, None, :],
                torch.zeros((), device=x.device, dtype=q.dtype),
                torch.full((), neg_inf, device=x.device, dtype=q.dtype),
            )

        # Encoder self-attention: allow bidirectional attention, and mask padded keys.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

#-------------------------------------------------------------------------------#
# Final Uncertainty-Guided Corrector Model
#-------------------------------------------------------------------------------#

class UncertaintyGuidedCorrector(nn.Module):
    """
    Full pipeline: takes (e2evarnet_output, uncertainty_map) →
    predicts residual → adds back → DC layer → corrected MRI.

    quadtree_backend
    ----------------
    ``"batched"`` — `BatchedQuadtreeTokenizer`: patch means via `avg_pool2d` on the
    full batch on device (fast; equivalent to `UncertaintyPatchScorer`).

    ``"cpu"`` — `QuadtreeTokenizer` + arbitrary `PatchScorer` (numpy regions; use for
    custom scorers that are not plain patch means).
    """
    def __init__(self, config, quadtree_backend: str = "batched"):
        super().__init__()
        self.cfg = config
        if quadtree_backend not in ("batched", "cpu"):
            raise ValueError('quadtree_backend must be "batched" or "cpu"')
        self.quadtree_backend = quadtree_backend

        self.image_size     = config.image_size
        self.min_patch_size = config.min_patch_size
        self.num_tokens     = config.num_tokens
        self.finest_grid    = config.image_size // config.min_patch_size
        num_scales = int(math.log2(config.max_patch_size // config.min_patch_size)) + 1
        self.scale_to_idx = {
            config.min_patch_size * (2 ** i): (num_scales - 1 - i)
            for i in range(num_scales)
        }

        self.quadtree         = QuadtreeTokenizer(config)
        self.quadtree_batched = BatchedQuadtreeTokenizer(config)
        self.cnn_encoder = CNNEncoder(config)
        self.embedder    = PatchEmbedder(config)
        
        # Transformer encoder over token embeddings.
        # (No token-id embedding table is needed because tokens are already embedded.)
        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.cnn_decoder = CNNDecoder(config)
        self.dc_layer    = DataConsistency()

    def vit_encoder(self, tokens: torch.Tensor, metas: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder over quadformer tokens.
        metas: [B, T, 5] from QuadtreeTokenizer, where PATCH_SIZE==0 marks padded tokens.
        """
        attn_mask = (metas[..., PATCH_SIZE] > 0).to(tokens.device)
        x = tokens
        for blk in self.transformer.h:
            x = blk(x, attn_mask=attn_mask)
        return self.transformer.ln_f(x)

    def _build_tokens(self, cnn_feat: torch.Tensor, metas: torch.Tensor) -> torch.Tensor:
        """
        Build transformer tokens from CNN feature patches using the quadtree metas.

        Parameters
        ----------
        cnn_feat: [B, C, Hf, Wf]  CNN encoder final feature map.
        metas:    [B, T, 5]      Quadtree metas, padded with patch_size==0.
        """
        patches, cx, cy, scale_idx = extract_patches(
            cnn_feat, metas, self.scale_to_idx, self.cfg
        )
        return self.embedder(patches, cx, cy, scale_idx)

    def forward(
        self,
        recon:     torch.Tensor,   # [B, 1, H, W]  E2E-VarNet output
        uq_map:    torch.Tensor,   # [B, 1, H, W]  conformal uncertainty map
        k_sampled: torch.Tensor,   # [B, 1, H, W]  measured k-space (complex)
        mask:      torch.Tensor,   # [B, 1, H, W]  undersampling mask
        scorer:    PatchScorer,    # pluggable — UncertaintyPatchScorer in practice
        use_dc: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_final:   [B, 1, H, W]  corrected reconstruction after DC layer
            residual:  [B, 1, H, W]  predicted residual (for loss computation)
        """
        B = recon.shape[0]

        # ── Step 1: CNN encoder (shared) ──────────────────────────────────────
        x_in              = torch.cat([recon, uq_map], dim=1)  # [B, 2, H, W]
        cnn_feat, skips   = self.cnn_encoder(x_in)
        # cnn_feat: [B, 128, H/4, W/4]   skips: [[B,32,H,W], [B,64,H/2,W/2], [B,128,H/4,W/4]]

        # ── Step 2: Quadtree tokenization (uses uq_map) ───────────────────────
        if self.quadtree_backend == "batched":
            metas = self.quadtree_batched.run_quadtree(uq_map)  # [B, T, 5], on uq_map.device
        else:
            uq_np_batch = uq_map.squeeze(1).detach().cpu().numpy()  # [B, H, W]
            metas = torch.stack(
                [self.quadtree.run_quadtree(uq_np_batch[b], scorer) for b in range(B)],
                dim=0,
            ).to(device=cnn_feat.device)  # [B, T, 5]

        # ── Step 3–4: Patch extraction + embedding → transformer tokens ───────
        tokens = self._build_tokens(cnn_feat, metas)  # [B, T, n_embd]

        # ── Step 5: Transformer encoder with padding mask ─────────────────────
        tokens = self.vit_encoder(tokens, metas)  # [B, T, n_embd]

        # ── Step 6: Spatial scatter → 2D feature map ─────────────────────────
        fmap = spatial_scatter(tokens, metas, self.cfg)  # [B, n_embd, G, G]

        # ── Step 7: CNN decoder + skip connections → residual ───────────────
        residual = self.cnn_decoder(fmap, skips)                # [B, 1, H, W]

        # ── Step 8: Add residual + optional data consistency ───────────────────
        x_corrected = recon + residual
        x_final = self.dc_layer(x_corrected, k_sampled, mask) if use_dc else x_corrected

        return x_final, residual


@torch.no_grad()
def debug_shapes():
    """Sanity-check tensor shapes at default image_size (320×320); small transformer for speed."""
    cfg = ModelConfig()
    cfg.num_tokens = 128
    cfg.n_layer = 2
    cfg.n_head = 8
    cfg.n_embd = 256

    B, H, W = 2, cfg.image_size, cfg.image_size
    recon = torch.randn(B, 1, H, W)
    uq = torch.rand(B, 1, H, W)
    k = torch.zeros(B, 1, H, W, dtype=torch.complex64)
    m = torch.ones(B, 1, H, W)

    model = UncertaintyGuidedCorrector(cfg)
    x_out, res = model(recon, uq, k, m, UncertaintyPatchScorer(), use_dc=False)

    assert res.shape == (B, 1, H, W), res.shape
    assert x_out.shape == (B, 1, H, W), x_out.shape
    print("debug_shapes OK:", res.shape, x_out.shape)


@torch.no_grad()
def debug_quadtree_parity():
    """Batched vs CPU tokenizer metas should match for mean uncertainty (tiny float noise ok)."""
    cfg = ModelConfig()
    cfg.num_tokens = 256
    B = 3
    torch.manual_seed(0)
    uq = torch.rand(B, 1, cfg.image_size, cfg.image_size)

    cpu_tok = QuadtreeTokenizer(cfg)
    bat = BatchedQuadtreeTokenizer(cfg)
    scorer = UncertaintyPatchScorer()
    uq_np = uq.squeeze(1).cpu().numpy()
    m_cpu = torch.stack([cpu_tok.run_quadtree(uq_np[b], scorer) for b in range(B)], dim=0)
    m_bat = bat.run_quadtree(uq)
    ok = torch.allclose(m_cpu, m_bat.cpu(), rtol=1e-4, atol=1e-5)
    assert ok, (m_cpu - m_bat.cpu()).abs().max()
    print("debug_quadtree_parity OK (max |Δ| <= 1e-4 rtol bucket)")


if __name__ == "__main__":
    debug_quadtree_parity()
    debug_shapes()