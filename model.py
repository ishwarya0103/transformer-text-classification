"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is not None:
                att = att + attn_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class SmallGPTConfig:
    block_size: int = 256
    vocab_size: int = 4096
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128 # 64 is also good
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    num_classes: int = 7 # BI-RADS 0-6

class SmallGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # MLM head: tied to token embedding weights
        self.mlm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.mlm_head.weight = self.transformer.wte.weight
        # Learned query for attention pooling (classification only; trained with cls_head LR)
        self.cls_attn_query = nn.Parameter(torch.zeros(config.n_embd))
        nn.init.normal_(self.cls_attn_query, std=0.02)
        # concat(mean-pool, max-pool, attn-pool) -> num_classes
        self.cls_head = nn.Linear(3 * config.n_embd, config.num_classes, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_attn_mask(self, attention_mask):
        """Convert (B, T) binary mask to (B, 1, 1, T) additive mask for attention."""
        if attention_mask is None:
            return None
        # 0 positions -> -inf, 1 positions -> 0
        return attention_mask[:, None, None, :].float().log()

    def forward_encoder(self, input_ids, attention_mask=None):
        """Shared encoder: returns per-token hidden states (B, T, n_embd)."""
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        attn_mask = self._build_attn_mask(attention_mask)

        x = self.transformer.drop(
            self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x, attn_mask=attn_mask)
        return self.transformer.ln_f(x)

    def forward_mlm(self, input_ids, attention_mask=None):
        """MLM forward: returns per-token logits (B, T, vocab_size)."""
        x = self.forward_encoder(input_ids, attention_mask)
        return self.mlm_head(x)

    def _masked_attention_pool(self, x, attention_mask):
        """Single learned query, softmax over valid tokens -> (B, n_embd)."""
        # x: (B, T, C), attention_mask: (B, T) with 1 = real token
        scale = self.config.n_embd ** -0.5
        scores = (x * self.cls_attn_query).sum(dim=-1) * scale
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (w * x).sum(dim=1)

    def forward(self, input_ids, attention_mask=None):
        """Classification: concat(mean, max, attention-pool), then classify."""
        x = self.forward_encoder(input_ids, attention_mask)

        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float()
            mean_pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            m = attention_mask.unsqueeze(-1).bool()
            neg = torch.finfo(x.dtype).min
            x_max = torch.where(m, x, neg)
            max_pooled = x_max.max(dim=1).values
            attn_pooled = self._masked_attention_pool(x, attention_mask)
        else:
            mean_pooled = x.mean(dim=1)
            max_pooled = x.max(dim=1).values
            b, t, c = x.shape
            attn_pooled = self._masked_attention_pool(
                x, torch.ones(b, t, device=x.device, dtype=torch.long)
            )

        pooled = torch.cat([mean_pooled, max_pooled, attn_pooled], dim=-1)
        return self.cls_head(pooled)

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def configure_optimizers_discriminative(
        self, weight_decay, lr_encoder, lr_head, betas, device_type
    ):
        """AdamW with lower LR on encoder vs cls_* (head + attention pool query)."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        def is_head(n):
            return n.startswith("cls_")

        enc_decay = [
            p for n, p in param_dict.items() if p.dim() >= 2 and not is_head(n)
        ]
        enc_nodecay = [
            p for n, p in param_dict.items() if p.dim() < 2 and not is_head(n)
        ]
        head_decay = [p for n, p in param_dict.items() if p.dim() >= 2 and is_head(n)]
        head_nodecay = [p for n, p in param_dict.items() if p.dim() < 2 and is_head(n)]

        optim_groups = []
        if enc_decay:
            optim_groups.append(
                {
                    "params": enc_decay,
                    "weight_decay": weight_decay,
                    "lr": lr_encoder,
                }
            )
        if enc_nodecay:
            optim_groups.append(
                {"params": enc_nodecay, "weight_decay": 0.0, "lr": lr_encoder}
            )
        if head_decay:
            optim_groups.append(
                {"params": head_decay, "weight_decay": weight_decay, "lr": lr_head}
            )
        if head_nodecay:
            optim_groups.append(
                {"params": head_nodecay, "weight_decay": 0.0, "lr": lr_head}
            )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)
        print(
            f"discriminative AdamW: {len(optim_groups)} groups | "
            f"lr_encoder={lr_encoder:g} lr_head={lr_head:g} | fused={use_fused}"
        )
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu