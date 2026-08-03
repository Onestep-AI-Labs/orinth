"""Generated PyTorch module source for the named blocks.

The torch counterpart of `keras_helpers`, and deliberately a translation of the
same structures rather than an independent design — `test_architecture_families`
asserts the two emitters agree on every block's parameter count exactly, which
they cannot do if one of them quietly models something else.

Two conventions differ from the Keras side and are visible here:

- **Images are NCHW.** Convolutions take `(batch, channels, height, width)`,
  so the vision blocks index channels at dim 1 rather than -1. Sequences are
  `(batch, length, width)` in both.
- **Widths are constructor arguments.** torch layers cannot infer their input
  width, so every helper takes the incoming width explicitly; the emitter reads
  it from the analytic shape pass.
"""

# --- shared transformer primitives ------------------------------------------

ROPE_APPLY_HELPER = '''def rope_angles(length, head_dim, base, device, dtype):
    """cos/sin tables for rotary position embedding, shaped (1, 1, length, head_dim)."""

    half = head_dim // 2
    inverse = base ** (-torch.arange(0, half, device=device, dtype=torch.float32) * 2.0 / head_dim)
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    angles = positions * inverse[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos().to(dtype)[None, None], angles.sin().to(dtype)[None, None]


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    """Rotate a (batch, heads, length, head_dim) tensor by position.

    Applied to queries and keys inside attention — rotary position is a property
    of the dot product, not of the residual stream.
    """

    return x * cos + rotate_half(x) * sin'''


ATTENTION_MASK_HELPER = '''def attention_mask(length, causal, sliding_window, device):
    """Boolean "may attend" mask of shape (length, length), or None."""

    if not causal and not sliding_window:
        return None
    positions = torch.arange(length, device=device)
    distance = positions[:, None] - positions[None, :]
    mask = distance >= 0 if causal else torch.ones_like(distance, dtype=torch.bool)
    if sliding_window:
        window = distance < sliding_window if causal else distance.abs() < sliding_window
        mask = mask & window
    return mask'''


FAMILY_ATTENTION_HELPER = '''class FamilyAttention(nn.Module):
    """Multi-head or grouped-query attention with rotary position applied inside.

    `num_kv_heads` below `num_heads` is GQA; `qk_norm` is Qwen3's RMSNorm over
    each head vector; `query_scale` is Gemma's `query_pre_attn_scalar`; a
    `rope_theta` of 0 means the model takes absolute positions from elsewhere,
    which is how GPT-2 and BERT are expressed.
    """

    def __init__(self, width, num_heads, num_kv_heads, head_dim, rope_theta=0.0,
                 causal=True, qk_norm=False, query_scale=0, sliding_window=0,
                 dropout=0.0, use_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.causal = causal
        self.query_scale = query_scale
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=use_bias)
        self.k_proj = nn.Linear(width, self.num_kv_heads * head_dim, bias=use_bias)
        self.v_proj = nn.Linear(width, self.num_kv_heads * head_dim, bias=use_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=use_bias)
        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else None
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else None
        self.dropout = dropout

    def forward(self, x):
        batch, length, _ = x.shape

        def split(t, heads):
            return t.view(batch, length, heads, self.head_dim).transpose(1, 2)

        query = split(self.q_proj(x), self.num_heads)
        key = split(self.k_proj(x), self.num_kv_heads)
        value = split(self.v_proj(x), self.num_kv_heads)
        if self.q_norm is not None:
            query, key = self.q_norm(query), self.k_norm(key)
        if self.rope_theta:
            cos, sin = rope_angles(length, self.head_dim, self.rope_theta, x.device, x.dtype)
            query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)

        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)

        mask = attention_mask(length, self.causal, self.sliding_window, x.device)
        context = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=float(self.query_scale or self.head_dim) ** -0.5,
        )
        context = context.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(context)'''


LATENT_ATTENTION_HELPER = '''class LatentAttention(nn.Module):
    """Multi-head latent attention (MLA), as DeepSeek V2/V3 and Kimi K2 use it.

    Keys and values are reconstructed from one `kv_lora_rank`-wide latent plus a
    single rotary key shared across heads, so that latent is all the cache has to
    hold. Each head's query and key split into a content half that comes from the
    latent and a position half that rotary touches.
    """

    def __init__(self, width, num_heads, kv_lora_rank, qk_nope_head_dim,
                 qk_rope_head_dim, v_head_dim, q_lora_rank=0, rope_theta=10000.0,
                 causal=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.rope_theta = rope_theta
        self.causal = causal
        self.dropout = dropout
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        if q_lora_rank:
            self.q_a_proj = nn.Linear(width, q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(q_lora_rank)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(width, num_heads * qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(width, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(num_heads * v_head_dim, width, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape

        if self.q_lora_rank:
            query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            query = self.q_proj(x)
        query = query.view(batch, length, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = query.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        latent, k_rot = compressed.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_rot = k_rot.unsqueeze(1)

        key_value = self.kv_b_proj(self.kv_a_layernorm(latent))
        key_value = key_value.view(
            batch, length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_pass, value = key_value.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if self.rope_theta and self.qk_rope_head_dim:
            cos, sin = rope_angles(
                length, self.qk_rope_head_dim, self.rope_theta, x.device, x.dtype
            )
            q_rot, k_rot = apply_rope(q_rot, cos, sin), apply_rope(k_rot, cos, sin)
        k_rot = k_rot.expand(-1, self.num_heads, -1, -1)

        query = torch.cat([q_pass, q_rot], dim=-1)
        key = torch.cat([k_pass, k_rot], dim=-1)
        mask = attention_mask(length, self.causal, 0, x.device)
        context = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=float(self.qk_head_dim) ** -0.5,
        )
        context = context.transpose(1, 2).reshape(batch, length, self.num_heads * self.v_head_dim)
        return self.o_proj(context)'''


GATED_FFN_HELPER = '''class GatedFeedForward(nn.Module):
    """`down(act(gate(x)) * up(x))` — SwiGLU with SiLU, GeGLU with GELU.

    Three bias-free matrices, matching every published implementation of either.
    """

    def __init__(self, width, hidden_dim, activation="silu", dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden_dim, bias=False)
        self.up_proj = nn.Linear(width, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, width, bias=False)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh") if self.activation == "gelu" else F.silu(gate)
        return self.down_proj(self.drop(gate * self.up_proj(x)))'''


SPARSE_MOE_HELPER = '''class SparseMoE(nn.Module):
    """Routed feed-forward in both styles the open models use.

    `softmax` is Mixtral's: softmax over all experts, top-k, renormalize.
    `sigmoid_bias` is DeepSeek V3's and Kimi K2's: independent sigmoid scores, a
    learned per-expert bias used for *selection only*, renormalize, then scale by
    `routed_scaling`. That bias is what replaces an auxiliary load-balancing loss.

    A readable dense-gather implementation: every expert runs on every token and
    unselected outputs are masked out. Correct and slow rather than fast.
    """

    def __init__(self, width, num_experts, experts_per_token, hidden_dim,
                 shared_experts=0, router="softmax", routed_scaling=1.0,
                 activation="silu", dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = min(experts_per_token, num_experts)
        self.router = router
        self.routed_scaling = routed_scaling
        self.gate = nn.Linear(width, num_experts, bias=False)
        if router == "sigmoid_bias":
            # A buffer, not a parameter: DeepSeek updates it from observed expert
            # load rather than from the gradient. Nothing here updates it, so it
            # stays a documented hook rather than a pretend implementation.
            self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))
        self.experts = nn.ModuleList(
            GatedFeedForward(width, hidden_dim, activation, dropout)
            for _ in range(num_experts)
        )
        self.shared = nn.ModuleList(
            GatedFeedForward(width, hidden_dim, activation, dropout)
            for _ in range(shared_experts)
        )

    def forward(self, x):
        logits = self.gate(x)
        if self.router == "sigmoid_bias":
            scores = logits.sigmoid()
            _, indices = torch.topk(
                scores + self.e_score_correction_bias, self.experts_per_token, dim=-1
            )
            chosen = scores.gather(-1, indices)
            chosen = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20)
            chosen = chosen * self.routed_scaling
        else:
            probabilities = logits.softmax(dim=-1)
            chosen, indices = torch.topk(probabilities, self.experts_per_token, dim=-1)
            chosen = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20)

        weights = torch.zeros_like(logits).scatter(-1, indices, chosen)
        stacked = torch.stack([expert(x) for expert in self.experts], dim=-2)
        output = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        for expert in self.shared:
            output = output + expert(x)
        return output'''


_TORCH_ATTENTION_MLA = '''            self.attention_layers.append(
                LatentAttention(
                    width, num_heads, kv_lora_rank, qk_nope_head_dim,
                    qk_rope_head_dim, v_head_dim, q_lora_rank=q_lora_rank,
                    rope_theta=rope_theta, causal=causal, dropout=dropout,
                )
            )'''

_TORCH_ATTENTION_FAMILY = '''            self.attention_layers.append(
                FamilyAttention(
                    width, num_heads,
                    num_heads if attention == "mha" else (num_kv_heads or num_heads),
                    head_dim, rope_theta=rope_theta, causal=causal, qk_norm=qk_norm,
                    query_scale=query_scale, sliding_window=window, dropout=dropout,
                    use_bias=use_bias,
                )
            )'''

_TORCH_FFN_MOE = '''            # DeepSeek keeps its first `dense_layers` feed-forwards dense.
            if index >= dense_layers:
                self.feed_forwards.append(
                    SparseMoE(
                        width, num_experts, experts_per_token, ffn_dim,
                        shared_experts=shared_experts, router=router,
                        routed_scaling=routed_scaling, dropout=dropout,
                    )
                )
            else:
                self.feed_forwards.append(
                    GatedFeedForward(width, dense_ffn_dim, "silu", dropout)
                )'''

_TORCH_FFN_GATED = '''            self.feed_forwards.append(
                GatedFeedForward(
                    width, ffn_dim, "gelu" if ffn == "geglu" else "silu", dropout
                )
            )'''

_TORCH_FFN_CLASSIC = '''            self.feed_forwards.append(
                nn.Sequential(
                    nn.Linear(width, ffn_dim, bias=use_bias),
                    nn.GELU(approximate="tanh"),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, width, bias=use_bias),
                )
            )'''


def llm_block_source(structures: set[str]) -> str:
    """Assemble `LlmBlockStack` with only the branches this graph reaches.

    Mirrors `keras_helpers.llm_block_source` from the same structure set, so a
    Gemma export in either framework contains no mixture-of-experts branch and
    no call to a class the module never defines.
    """

    if "latent_attention" in structures and "family_attention" in structures:
        attention = "\n".join(
            [
                '            if attention == "mla":',
                _indent(_TORCH_ATTENTION_MLA, 4),
                "            else:",
                _indent(_TORCH_ATTENTION_FAMILY, 4),
            ]
        )
    elif "latent_attention" in structures:
        attention = _TORCH_ATTENTION_MLA
    else:
        attention = _TORCH_ATTENTION_FAMILY

    if "sparse_moe" in structures:
        feed_forward = _TORCH_FFN_MOE
    elif "gated_ffn" in structures:
        feed_forward = _TORCH_FFN_GATED
    else:
        feed_forward = _TORCH_FFN_CLASSIC

    if "rms_norm" in structures and "layer_norm" in structures:
        make_norm = (
            "\n        def make_norm():\n"
            '            return nn.RMSNorm(width) if norm == "rms" '
            "else nn.LayerNorm(width, eps=1e-6)\n"
        )
    elif "rms_norm" in structures:
        make_norm = "\n        def make_norm():\n            return nn.RMSNorm(width)\n"
    else:
        make_norm = (
            "\n        def make_norm():\n            return nn.LayerNorm(width, eps=1e-6)\n"
        )

    return (
        LLM_BLOCK_TEMPLATE.replace("__MAKE_NORM__", make_norm.rstrip("\n"))
        .replace("__ATTENTION__", attention)
        .replace("__FEED_FORWARD__", feed_forward)
    )


def _indent(block: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in block.splitlines())


LLM_BLOCK_TEMPLATE = '''class LlmBlockStack(nn.Module):
    """A stack of decoder layers in one of the published arrangements.

    `norm_placement` is the axis the families actually differ on:

        pre       norm(x) -> sublayer -> add          Llama, Qwen, Mistral, GPT-2
        sandwich  norm(x) -> sublayer -> norm -> add  Gemma 2/3, four norms a layer
        post      x -> sublayer -> add -> norm        the 2017 paper, BERT
    """

    def __init__(self, width, *, layers, attention="gqa", num_heads=32, num_kv_heads=0,
                 head_dim=128, qk_norm=False, query_scale=0, rope_theta=500000.0,
                 sliding_window=0, global_every=0, causal=True, use_bias=False,
                 q_lora_rank=0, kv_lora_rank=512, qk_rope_head_dim=64,
                 qk_nope_head_dim=128, v_head_dim=128, ffn="swiglu", ffn_dim=14336,
                 num_experts=8, experts_per_token=2, shared_experts=0, dense_layers=0,
                 dense_ffn_dim=18432, router="softmax", routed_scaling=1.0,
                 norm="rms", norm_placement="pre", dropout=0.0):
        super().__init__()
        self.norm_placement = norm_placement
__MAKE_NORM__

        self.attention_layers = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        self.norms = nn.ModuleList()
        for index in range(layers):
            is_global = bool(global_every) and (index + 1) % global_every == 0
            window = 0 if (is_global or not sliding_window) else sliding_window
__ATTENTION__
__FEED_FORWARD__
            # Two norms a layer, or four under sandwich placement.
            count = 4 if norm_placement == "sandwich" else 2
            self.norms.extend(make_norm() for _ in range(count))

    def forward(self, x):
        per_layer = 4 if self.norm_placement == "sandwich" else 2
        for index, (attend, feed_forward) in enumerate(
            zip(self.attention_layers, self.feed_forwards)
        ):
            norms = self.norms[index * per_layer : (index + 1) * per_layer]
            x = self._sublayer(x, attend, norms[0], norms[1] if per_layer == 4 else None)
            x = self._sublayer(
                x,
                feed_forward,
                norms[2] if per_layer == 4 else norms[1],
                norms[3] if per_layer == 4 else None,
            )
        return x

    def _sublayer(self, x, body, norm, post_norm):
        residual = x
        if self.norm_placement in ("pre", "sandwich"):
            x = norm(x)
        x = body(x)
        if post_norm is not None:
            x = post_norm(x)
        x = residual + x
        return norm(x) if self.norm_placement == "post" else x'''


# --- vision helpers ---------------------------------------------------------

SQUEEZE_EXCITE_HELPER = '''class SqueezeExcite(nn.Module):
    """Recalibrate channels by their global importance (Hu et al. 2017)."""

    def __init__(self, channels, ratio=4, gate="sigmoid"):
        super().__init__()
        squeezed = max(1, channels // ratio)
        self.reduce = nn.Linear(channels, squeezed)
        self.expand = nn.Linear(squeezed, channels)
        self.gate = gate

    def forward(self, x):
        weights = self.expand(F.relu(self.reduce(x.mean(dim=(2, 3)))))
        weights = F.hardsigmoid(weights) if self.gate == "hardsigmoid" else weights.sigmoid()
        return x * weights[:, :, None, None]'''


PATCH_EMBEDDING_HELPER = '''class PatchEmbedding(nn.Module):
    """Cut an image into non-overlapping patches and project each to a vector.

    A convolution whose kernel equals its stride, which is exactly the
    reshape-and-matmul the ViT paper describes. Takes NCHW, returns a
    (batch, tokens, embed_dim) sequence.
    """

    def __init__(self, channels, patch_size, embed_dim, class_token=False):
        super().__init__()
        self.project = nn.Conv2d(channels, embed_dim, patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) if class_token else None

    def forward(self, x):
        tokens = self.project(x).flatten(2).transpose(1, 2)
        if self.cls is not None:
            tokens = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        return tokens'''


LAYER_SCALE_HELPER = '''class LayerScale(nn.Module):
    """A learned per-channel multiplier on a residual branch, initialised near zero."""

    def __init__(self, channels, init_value=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((channels,), init_value))

    def forward(self, x):
        return x * self.gamma'''


RESNET_HELPER = '''class ResNetStage(nn.Module):
    """`blocks` residual blocks; only the first may change width or resolution.

    Convolutions carry no bias — a BatchNorm follows each and its beta does the
    same job, which is why torchvision omits them too.
    """

    def __init__(self, channels, filters, blocks, stride=1, variant="basic", expansion=4):
        super().__init__()
        out_channels = filters * expansion if variant == "bottleneck" else filters
        self.variant = variant
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        for index in range(blocks):
            source = channels if index == 0 else out_channels
            block_stride = stride if index == 0 else 1
            if variant == "bottleneck":
                body = nn.Sequential(
                    nn.Conv2d(source, filters, 1, bias=False),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filters, filters, 3, stride=block_stride, padding=1, bias=False),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filters, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                body = nn.Sequential(
                    nn.Conv2d(source, filters, 3, stride=block_stride, padding=1, bias=False),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filters, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            self.blocks.append(body)
            if source != out_channels or block_stride != 1:
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv2d(source, out_channels, 1, stride=block_stride, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )
                )
            else:
                self.shortcuts.append(nn.Identity())

    def forward(self, x):
        for body, shortcut in zip(self.blocks, self.shortcuts):
            x = F.relu(shortcut(x) + body(x))
        return x'''


INVERTED_RESIDUAL_HELPER = '''class InvertedResidual(nn.Module):
    """MobileNetV2's inverted residual, with MobileNetV3's optional SE.

    Expands into a wide depthwise convolution and projects back down. The
    projection has no activation: a ReLU on the narrow tensor would destroy
    information the residual cannot recover, which is the linear-bottleneck
    argument the paper is named for.
    """

    def __init__(self, channels, filters, expand_ratio=6, kernel_size=3, stride=1,
                 use_se=False, se_ratio=4, activation="relu6", blocks=1):
        super().__init__()
        act = {
            "relu6": nn.ReLU6,
            "relu": nn.ReLU,
            "hardswish": nn.Hardswish,
            "swish": nn.SiLU,
        }.get(activation, nn.ReLU6)
        self.blocks = nn.ModuleList()
        self.connect = []
        for index in range(blocks):
            source = channels if index == 0 else filters
            block_stride = stride if index == 0 else 1
            hidden = source * expand_ratio
            layers = []
            if expand_ratio != 1:
                layers += [
                    nn.Conv2d(source, hidden, 1, bias=False),
                    nn.BatchNorm2d(hidden),
                    act(),
                ]
            layers += [
                nn.Conv2d(
                    hidden, hidden, kernel_size, stride=block_stride,
                    padding=kernel_size // 2, groups=hidden, bias=False,
                ),
                nn.BatchNorm2d(hidden),
                act(),
            ]
            if use_se:
                layers.append(SqueezeExcite(hidden, se_ratio, gate="hardsigmoid"))
            layers += [nn.Conv2d(hidden, filters, 1, bias=False), nn.BatchNorm2d(filters)]
            self.blocks.append(nn.Sequential(*layers))
            # No projection shortcut: connect only where the shapes already agree.
            self.connect.append(block_stride == 1 and source == filters)

    def forward(self, x):
        for body, connect in zip(self.blocks, self.connect):
            x = x + body(x) if connect else body(x)
        return x'''


DENSE_BLOCK_HELPER = '''class DenseBlock(nn.Module):
    """DenseNet-BC: every layer reads the concatenation of all earlier outputs."""

    def __init__(self, channels, growth_rate, layers, bottleneck_ratio=4):
        super().__init__()
        inner = growth_rate * bottleneck_ratio
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm2d(channels + index * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels + index * growth_rate, inner, 1, bias=False),
                nn.BatchNorm2d(inner),
                nn.ReLU(inplace=True),
                nn.Conv2d(inner, growth_rate, 3, padding=1, bias=False),
            )
            for index in range(layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x'''


INCEPTION_HELPER = '''class InceptionModule(nn.Module):
    """GoogLeNet's module: four receptive fields in parallel, concatenated.

    The 1x1 reductions before the 3x3 and 5x5 paths cut the channel count first,
    which is what makes the wide kernels affordable.
    """

    def __init__(self, channels, filters_1x1, reduce_3x3, filters_3x3,
                 reduce_5x5, filters_5x5, filters_pool):
        super().__init__()
        self.path_1 = nn.Conv2d(channels, filters_1x1, 1)
        self.path_3 = nn.Sequential(
            nn.Conv2d(channels, reduce_3x3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, filters_3x3, 3, padding=1),
        )
        self.path_5 = nn.Sequential(
            nn.Conv2d(channels, reduce_5x5, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, filters_5x5, 5, padding=2),
        )
        self.path_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels, filters_pool, 1),
        )

    def forward(self, x):
        return torch.cat(
            [
                F.relu(self.path_1(x)),
                F.relu(self.path_3(x)),
                F.relu(self.path_5(x)),
                F.relu(self.path_pool(x)),
            ],
            dim=1,
        )'''


CONVNEXT_HELPER = '''class ConvNeXtStage(nn.Module):
    """ConvNeXt: a transformer block's layout built from convolutions.

    Large depthwise kernel in place of attention's receptive field, one
    LayerNorm rather than BatchNorm everywhere, one activation per block rather
    than three, and an inverted 4x bottleneck.
    """

    def __init__(self, channels, filters, blocks, kernel_size=7, expand_ratio=4,
                 layer_scale=1e-6):
        super().__init__()
        self.project = (
            nn.Conv2d(channels, filters, 1) if channels != filters else nn.Identity()
        )
        hidden = filters * expand_ratio
        self.depthwise = nn.ModuleList(
            nn.Conv2d(filters, filters, kernel_size, padding=kernel_size // 2, groups=filters)
            for _ in range(blocks)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(filters, eps=1e-6) for _ in range(blocks))
        self.pointwise1 = nn.ModuleList(nn.Linear(filters, hidden) for _ in range(blocks))
        self.pointwise2 = nn.ModuleList(nn.Linear(hidden, filters) for _ in range(blocks))
        self.scales = nn.ModuleList(
            (LayerScale(filters, layer_scale) if layer_scale else nn.Identity())
            for _ in range(blocks)
        )

    def forward(self, x):
        x = self.project(x)
        for depthwise, norm, up, down, scale in zip(
            self.depthwise, self.norms, self.pointwise1, self.pointwise2, self.scales
        ):
            residual = x
            h = depthwise(x)
            # The pointwise stage is channels-last, as in the reference
            # implementation — a Linear over channels, not a 1x1 convolution.
            h = h.permute(0, 2, 3, 1)
            h = down(F.gelu(up(norm(h))))
            h = scale(h)
            x = residual + h.permute(0, 3, 1, 2)
        return x'''


VIT_HELPER = '''class ViTEncoder(nn.Module):
    """Pre-LayerNorm attention and a GELU MLP over a patch sequence.

    Structurally a GPT block with the causal mask off — an image has no reading
    order, so a patch may attend to every other patch.
    """

    def __init__(self, width, layers, num_heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.attention = nn.ModuleList(
            nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        )
        self.attention_norms = nn.ModuleList(
            nn.LayerNorm(width, eps=1e-6) for _ in range(layers)
        )
        self.mlp_norms = nn.ModuleList(nn.LayerNorm(width, eps=1e-6) for _ in range(layers))
        self.mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, width),
            )
            for _ in range(layers)
        )

    def forward(self, x):
        for attend, attention_norm, mlp_norm, mlp in zip(
            self.attention, self.attention_norms, self.mlp_norms, self.mlps
        ):
            h = attention_norm(x)
            x = x + attend(h, h, h, need_weights=False)[0]
            x = x + mlp(mlp_norm(x))
        return x'''
