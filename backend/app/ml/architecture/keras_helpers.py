"""Generated Keras layer source for the named blocks.

These are strings, not code that runs here: `emit_keras` writes the ones a
graph actually uses above `build_model`, so a plain CNN's generated file
carries no transformer machinery and a Llama graph's carries no convolutions.

They are kept out of `emit_keras.py` because they are reference implementations
that want to be read as Python — the file they are pasted into is the artifact a
user downloads and runs. Each one names the paper or repository it follows, and
`backend/tests/test_architecture_families.py` asserts the parameter counts they
produce match those sources.
"""

# --- shared transformer primitives ------------------------------------------

ROPE_APPLY_HELPER = '''def rope_angles(length, head_dim, base, dtype=tf.float32):
    """cos/sin tables for rotary position embedding, shaped (1, 1, length, head_dim)."""

    half = head_dim // 2
    inverse = tf.pow(
        tf.cast(base, tf.float32), -tf.range(0, half, dtype=tf.float32) * 2.0 / float(head_dim)
    )
    positions = tf.cast(tf.range(length), tf.float32)[:, None]
    angles = positions * inverse[None, :]
    angles = tf.concat([angles, angles], axis=-1)
    return (
        tf.cast(tf.cos(angles), dtype)[None, None, :, :],
        tf.cast(tf.sin(angles), dtype)[None, None, :, :],
    )


def rotate_half(x):
    half = tf.shape(x)[-1] // 2
    return tf.concat([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, cos, sin):
    """Rotate a (batch, heads, length, head_dim) tensor by position.

    Applied to queries and keys *inside* attention, not to the token embeddings
    — rotary position is a property of the dot product, and rotating the
    residual stream instead is a different (and worse) model.
    """

    return x * cos + rotate_half(x) * sin'''


CAUSAL_MASK_HELPER = '''def attention_mask(length, causal, sliding_window, dtype=tf.bool):
    """Boolean "may attend" mask of shape (length, length)."""

    positions = tf.range(length)
    mask = None
    if causal:
        mask = positions[:, None] >= positions[None, :]
    if sliding_window:
        # A token sees only the last `sliding_window` positions. Mistral's
        # design, and what Gemma runs on five layers out of every six.
        within = positions[:, None] - positions[None, :] < sliding_window
        mask = within if mask is None else tf.logical_and(mask, within)
    return mask'''


FAMILY_ATTENTION_HELPER = '''class FamilyAttention(tf.keras.layers.Layer):
    """Multi-head or grouped-query attention, with rotary position applied inside.

    Covers what the Llama, Qwen, Mistral and Gemma families all do to the same
    four projections:

    - `num_kv_heads` below `num_heads` shares each key/value head across a group
      of query heads, which is what shrinks the KV cache (GQA).
    - `qk_norm` runs an RMSNorm over each query and key *head vector* before the
      dot product. Qwen3's signature; Gemma 3 uses it too.
    - `query_scale` divides scores by its square root instead of the head dim's.
      Gemma states 256 explicitly as `query_pre_attn_scalar`.
    - `rope_theta` of 0 means no rotary at all, which is how GPT-2 and BERT are
      expressed here — they take absolute positions from a separate layer.
    """

    def __init__(self, num_heads, num_kv_heads, head_dim, rope_theta=0.0, causal=True,
                 qk_norm=False, query_scale=0, sliding_window=0, dropout=0.0,
                 use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.causal = causal
        self.qk_norm = qk_norm
        self.query_scale = query_scale
        self.sliding_window = sliding_window
        self.dropout_rate = dropout
        self.use_bias = use_bias

    def build(self, input_shape):
        width = input_shape[-1]
        bias = self.use_bias
        self.q_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=bias, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.num_kv_heads * self.head_dim, use_bias=bias, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.num_kv_heads * self.head_dim, use_bias=bias, name="v_proj")
        self.o_proj = tf.keras.layers.Dense(width, use_bias=bias, name="o_proj")
        self.drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.q_norm = RMSNorm(name="q_norm") if self.qk_norm else None
        self.k_norm = RMSNorm(name="k_norm") if self.qk_norm else None
        super().build(input_shape)

    def call(self, inputs, training=None):
        batch = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]

        def split(x, heads):
            return tf.transpose(tf.reshape(x, (batch, length, heads, self.head_dim)), (0, 2, 1, 3))

        query = split(self.q_proj(inputs), self.num_heads)
        key = split(self.k_proj(inputs), self.num_kv_heads)
        value = split(self.v_proj(inputs), self.num_kv_heads)

        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.rope_theta:
            cos, sin = rope_angles(length, self.head_dim, self.rope_theta, inputs.dtype)
            query = apply_rope(query, cos, sin)
            key = apply_rope(key, cos, sin)

        # One key/value head serves this many query heads.
        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            key = tf.repeat(key, groups, axis=1)
            value = tf.repeat(value, groups, axis=1)

        scale = float(self.query_scale or self.head_dim) ** -0.5
        scores = tf.matmul(query, key, transpose_b=True) * tf.cast(scale, query.dtype)
        mask = attention_mask(length, self.causal, self.sliding_window)
        if mask is not None:
            scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        weights = self.drop(tf.nn.softmax(scores, axis=-1), training=training)
        context = tf.transpose(tf.matmul(weights, value), (0, 2, 1, 3))
        return self.o_proj(tf.reshape(context, (batch, length, self.num_heads * self.head_dim)))

    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "causal": self.causal,
            "qk_norm": self.qk_norm,
            "query_scale": self.query_scale,
            "sliding_window": self.sliding_window,
            "dropout": self.dropout_rate,
            "use_bias": self.use_bias,
        }'''


LATENT_ATTENTION_HELPER = '''class LatentAttention(tf.keras.layers.Layer):
    """Multi-head latent attention (MLA), as DeepSeek V2/V3 and Kimi K2 use it.

    Standard attention caches one key and one value vector per head per token.
    MLA caches a single `kv_lora_rank`-wide latent instead, plus one rotary key
    shared by every head, and reconstructs the per-head keys and values from it.
    At DeepSeek V3's 128 heads that is the difference between a servable model
    and an unservable one.

    Each head's query and key splits into two halves that do different jobs: the
    `nope` half carries content and comes out of the latent, the `rope` half
    carries position and is the only part rotary touches.
    """

    def __init__(self, num_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
                 v_head_dim, q_lora_rank=0, rope_theta=10000.0, causal=True,
                 dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.rope_theta = rope_theta
        self.causal = causal
        self.dropout_rate = dropout

    @property
    def qk_head_dim(self):
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def build(self, input_shape):
        width = input_shape[-1]
        if self.q_lora_rank:
            self.q_a_proj = tf.keras.layers.Dense(self.q_lora_rank, use_bias=False, name="q_a_proj")
            self.q_a_layernorm = RMSNorm(name="q_a_layernorm")
            self.q_b_proj = tf.keras.layers.Dense(
                self.num_heads * self.qk_head_dim, use_bias=False, name="q_b_proj"
            )
        else:
            self.q_proj = tf.keras.layers.Dense(
                self.num_heads * self.qk_head_dim, use_bias=False, name="q_proj"
            )
        # The latent plus the one rotary key every head shares.
        self.kv_a_proj_with_mqa = tf.keras.layers.Dense(
            self.kv_lora_rank + self.qk_rope_head_dim, use_bias=False, name="kv_a_proj_with_mqa"
        )
        self.kv_a_layernorm = RMSNorm(name="kv_a_layernorm")
        self.kv_b_proj = tf.keras.layers.Dense(
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False,
            name="kv_b_proj",
        )
        self.o_proj = tf.keras.layers.Dense(width, use_bias=False, name="o_proj")
        self.drop = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        batch = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]

        if self.q_lora_rank:
            query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(inputs)))
        else:
            query = self.q_proj(inputs)
        query = tf.transpose(
            tf.reshape(query, (batch, length, self.num_heads, self.qk_head_dim)), (0, 2, 1, 3)
        )
        q_pass = query[..., : self.qk_nope_head_dim]
        q_rot = query[..., self.qk_nope_head_dim :]

        compressed = self.kv_a_proj_with_mqa(inputs)
        latent = compressed[..., : self.kv_lora_rank]
        k_rot = compressed[..., self.kv_lora_rank :][:, None, :, :]

        key_value = self.kv_b_proj(self.kv_a_layernorm(latent))
        key_value = tf.transpose(
            tf.reshape(
                key_value,
                (batch, length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
            ),
            (0, 2, 1, 3),
        )
        k_pass = key_value[..., : self.qk_nope_head_dim]
        value = key_value[..., self.qk_nope_head_dim :]

        if self.rope_theta and self.qk_rope_head_dim:
            cos, sin = rope_angles(length, self.qk_rope_head_dim, self.rope_theta, inputs.dtype)
            q_rot = apply_rope(q_rot, cos, sin)
            k_rot = apply_rope(k_rot, cos, sin)
        k_rot = tf.repeat(k_rot, self.num_heads, axis=1)

        query = tf.concat([q_pass, q_rot], axis=-1)
        key = tf.concat([k_pass, k_rot], axis=-1)

        scale = float(self.qk_head_dim) ** -0.5
        scores = tf.matmul(query, key, transpose_b=True) * tf.cast(scale, query.dtype)
        mask = attention_mask(length, self.causal, 0)
        if mask is not None:
            scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        weights = self.drop(tf.nn.softmax(scores, axis=-1), training=training)
        context = tf.transpose(tf.matmul(weights, value), (0, 2, 1, 3))
        return self.o_proj(tf.reshape(context, (batch, length, self.num_heads * self.v_head_dim)))

    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "q_lora_rank": self.q_lora_rank,
            "rope_theta": self.rope_theta,
            "causal": self.causal,
            "dropout": self.dropout_rate,
        }'''


GATED_FFN_HELPER = '''class GatedFeedForward(tf.keras.layers.Layer):
    """`down(act(gate(x)) * up(x))` — SwiGLU when act is SiLU, GeGLU when GELU.

    Three bias-free matrices, which is what every published implementation of
    either uses. Llama, Qwen, Mistral and DeepSeek gate with SiLU; Gemma gates
    with the tanh approximation of GELU.
    """

    def __init__(self, hidden_dim, activation="silu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        width = input_shape[-1]
        self.gate_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=False, name="gate_proj")
        self.up_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=False, name="up_proj")
        self.down_proj = tf.keras.layers.Dense(width, use_bias=False, name="down_proj")
        self.drop = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        gate = self.gate_proj(inputs)
        gate = tf.nn.gelu(gate, approximate=True) if self.activation == "gelu" else tf.nn.silu(gate)
        return self.down_proj(self.drop(gate * self.up_proj(inputs), training=training))

    def get_config(self):
        return {
            **super().get_config(),
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "dropout": self.dropout_rate,
        }'''


SPARSE_MOE_HELPER = '''class SparseMoE(tf.keras.layers.Layer):
    """Routed feed-forward, in both routing styles the open models use.

    - `softmax` (Mixtral): softmax over every expert, take the top-k, then
      renormalize those k so they sum to 1.
    - `sigmoid_bias` (DeepSeek V3, Kimi K2): score each expert independently
      with a sigmoid, add a learned per-expert bias *for selection only*, take
      the top-k by the biased score but gate with the unbiased one, renormalize,
      and scale by `routed_scaling`. The bias is updated to balance load instead
      of an auxiliary loss, which is why DeepSeek calls it aux-loss-free.

    `shared_experts` run on every token, unrouted, alongside the top-k.

    This is a readable dense-gather implementation, not a production kernel:
    every expert runs on every token and unselected results are masked out. It
    is correct and slow rather than fast, which is the right trade for a file
    someone reads.
    """

    def __init__(self, num_experts, experts_per_token, hidden_dim, shared_experts=0,
                 router="softmax", routed_scaling=1.0, activation="silu", dropout=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.experts_per_token = min(experts_per_token, num_experts)
        self.hidden_dim = hidden_dim
        self.shared_experts = shared_experts
        self.router = router
        self.routed_scaling = routed_scaling
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.gate = tf.keras.layers.Dense(self.num_experts, use_bias=False, name="gate")
        if self.router == "sigmoid_bias":
            # Non-trainable: DeepSeek nudges it during training to balance
            # expert load. Nothing here updates it, so it stays a documented
            # hook rather than a pretend implementation.
            self.selection_bias = self.add_weight(
                name="e_score_correction_bias",
                shape=(self.num_experts,),
                initializer="zeros",
                trainable=False,
            )
        self.experts = [
            GatedFeedForward(
                self.hidden_dim, self.activation, self.dropout_rate, name=f"expert_{index}"
            )
            for index in range(self.num_experts)
        ]
        self.shared = [
            GatedFeedForward(
                self.hidden_dim, self.activation, self.dropout_rate, name=f"shared_expert_{index}"
            )
            for index in range(self.shared_experts)
        ]
        super().build(input_shape)

    def call(self, inputs, training=None):
        logits = self.gate(inputs)
        if self.router == "sigmoid_bias":
            scores = tf.nn.sigmoid(logits)
            _, indices = tf.math.top_k(scores + self.selection_bias, k=self.experts_per_token)
            chosen = tf.gather(scores, indices, batch_dims=2)
            chosen = chosen / (tf.reduce_sum(chosen, axis=-1, keepdims=True) + 1e-20)
            chosen = chosen * self.routed_scaling
        else:
            probabilities = tf.nn.softmax(logits, axis=-1)
            chosen, indices = tf.math.top_k(probabilities, k=self.experts_per_token)
            chosen = chosen / (tf.reduce_sum(chosen, axis=-1, keepdims=True) + 1e-20)

        # Scatter the k gate values back over all experts; unselected weight is 0.
        weights = tf.reduce_sum(
            tf.one_hot(indices, self.num_experts) * chosen[..., None], axis=-2
        )
        stacked = tf.stack(
            [expert(inputs, training=training) for expert in self.experts], axis=-2
        )
        output = tf.reduce_sum(stacked * weights[..., None], axis=-2)
        for expert in self.shared:
            output += expert(inputs, training=training)
        return output

    def get_config(self):
        return {
            **super().get_config(),
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "hidden_dim": self.hidden_dim,
            "shared_experts": self.shared_experts,
            "router": self.router,
            "routed_scaling": self.routed_scaling,
            "activation": self.activation,
            "dropout": self.dropout_rate,
        }'''


_LLM_BLOCK_ATTENTION_MLA = '''        attend = LatentAttention(
            num_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank=q_lora_rank,
            rope_theta=rope_theta,
            causal=causal,
            dropout=dropout,
            name=f"{prefix}_attn",
        )'''

_LLM_BLOCK_ATTENTION_FAMILY = '''        attend = FamilyAttention(
            num_heads,
            num_heads if attention == "mha" else (num_kv_heads or num_heads),
            head_dim,
            rope_theta=rope_theta,
            causal=causal,
            qk_norm=qk_norm,
            query_scale=query_scale,
            sliding_window=window,
            dropout=dropout,
            use_bias=use_bias,
            name=f"{prefix}_attn",
        )'''

_LLM_BLOCK_FFN_MOE = '''        # DeepSeek keeps its first `dense_layers` feed-forwards dense: the early
        # layers learn features every token needs, so routing them wastes the
        # experts' capacity.
        if index >= dense_layers:
            feed_forward = SparseMoE(
                num_experts,
                experts_per_token,
                ffn_dim,
                shared_experts=shared_experts,
                router=router,
                routed_scaling=routed_scaling,
                dropout=dropout,
                name=f"{prefix}_moe",
            )
        else:
            feed_forward = GatedFeedForward(
                dense_ffn_dim, "silu", dropout, name=f"{prefix}_ffn"
            )'''

_LLM_BLOCK_FFN_GATED = '''        feed_forward = GatedFeedForward(
            ffn_dim,
            "gelu" if ffn == "geglu" else "silu",
            dropout,
            name=f"{prefix}_ffn",
        )'''

_LLM_BLOCK_FFN_CLASSIC = '''        feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ffn_dim, use_bias=use_bias),
                tf.keras.layers.Lambda(lambda t: tf.nn.gelu(t, approximate=True)),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(int(x.shape[-1]), use_bias=use_bias),
            ],
            name=f"{prefix}_ffn",
        )'''


def llm_block_source(structures: set[str]) -> str:
    """Assemble `llm_block` with only the branches this graph reaches.

    A Gemma export should not contain a mixture-of-experts branch, and not only
    for tidiness: the helper classes are emitted on the same basis, so a dead
    branch here would call a class the module never defines. What the file
    contains is exactly what the architecture is.
    """

    attention = []
    if "latent_attention" in structures and "family_attention" in structures:
        attention = [
            '        if attention == "mla":',
            _indent(_LLM_BLOCK_ATTENTION_MLA, 4),
            "        else:",
            _indent(_LLM_BLOCK_ATTENTION_FAMILY, 4),
        ]
    elif "latent_attention" in structures:
        attention = [_LLM_BLOCK_ATTENTION_MLA]
    else:
        attention = [_LLM_BLOCK_ATTENTION_FAMILY]

    feed_forward = []
    if "sparse_moe" in structures:
        feed_forward.append(_LLM_BLOCK_FFN_MOE)
    elif "gated_ffn" in structures:
        feed_forward.append(_LLM_BLOCK_FFN_GATED)
    else:
        feed_forward.append(_LLM_BLOCK_FFN_CLASSIC)

    if "rms_norm" in structures and "layer_norm" in structures:
        make_norm = [
            "    def make_norm(norm_name):",
            '        if norm == "rms":',
            "            return RMSNorm(name=norm_name)",
            "        return tf.keras.layers.LayerNormalization(epsilon=1e-6, name=norm_name)",
        ]
    elif "rms_norm" in structures:
        make_norm = [
            "    def make_norm(norm_name):",
            "        return RMSNorm(name=norm_name)",
        ]
    else:
        make_norm = [
            "    def make_norm(norm_name):",
            "        return tf.keras.layers.LayerNormalization(epsilon=1e-6, name=norm_name)",
        ]

    return "\n".join(
        [
            _LLM_BLOCK_SIGNATURE,
            *make_norm,
            "",
            _LLM_BLOCK_SUBLAYER,
            "",
            "    for index in range(layers):",
            '        prefix = f"{name}_{index + 1}"',
            "        # Hybrid attention: most layers see a local band, every",
            "        # `global_every`-th sees the whole sequence. Gemma 3 runs five",
            "        # local to one global.",
            "        is_global = bool(global_every) and (index + 1) % global_every == 0",
            "        window = 0 if (is_global or not sliding_window) else sliding_window",
            "",
            *attention,
            '        x = sublayer(x, attend, f"{prefix}_attn")',
            "",
            *feed_forward,
            '        x = sublayer(x, feed_forward, f"{prefix}_ffn")',
            "    return x",
        ]
    )


def _indent(block: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in block.splitlines())


_LLM_BLOCK_SUBLAYER = '''    def sublayer(value, body, prefix):
        """Wrap one sub-layer in the residual and norms this family uses."""

        residual = value
        if norm_placement in ("pre", "sandwich"):
            value = make_norm(f"{prefix}_norm")(value)
        value = body(value)
        if norm_placement == "sandwich":
            value = make_norm(f"{prefix}_post_norm")(value)
        value = tf.keras.layers.Add(name=f"{prefix}_residual")([residual, value])
        if norm_placement == "post":
            value = make_norm(f"{prefix}_norm")(value)
        return value'''


_LLM_BLOCK_SIGNATURE = '''def llm_block(
    x,
    *,
    layers,
    attention="gqa",
    num_heads=32,
    num_kv_heads=0,
    head_dim=128,
    qk_norm=False,
    query_scale=0,
    rope_theta=500000.0,
    sliding_window=0,
    global_every=0,
    causal=True,
    use_bias=False,
    q_lora_rank=0,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    ffn="swiglu",
    ffn_dim=14336,
    num_experts=8,
    experts_per_token=2,
    shared_experts=0,
    dense_layers=0,
    dense_ffn_dim=18432,
    router="softmax",
    routed_scaling=1.0,
    norm="rms",
    norm_placement="pre",
    dropout=0.0,
    name="block",
):
    """A stack of decoder layers in one of the published arrangements.

    `norm_placement` is the axis the families actually differ on and the one
    that is easiest to get wrong:

        pre       norm(x) -> sublayer -> add          Llama, Qwen, Mistral, GPT-2
        sandwich  norm(x) -> sublayer -> norm -> add  Gemma 2/3, four norms a layer
        post      x -> sublayer -> add -> norm        the 2017 paper, BERT

    Pre-norm is what makes deep stacks trainable without a warmup schedule;
    post-norm needs one. Sandwich keeps the residual stream's scale bounded at
    both ends, which is what lets Gemma run a very wide vocabulary.
    """
'''


# --- vision helpers ---------------------------------------------------------

SQUEEZE_EXCITE_HELPER = '''class SqueezeExcite(tf.keras.layers.Layer):
    """Recalibrate channels by their global importance (Hu et al. 2017).

    Average each feature map to one number, learn a gate from that vector, and
    scale the channels by it. A few thousand parameters buys most of what a
    wider network would.
    """

    def __init__(self, ratio=4, gate="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gate = gate

    def build(self, input_shape):
        channels = int(input_shape[-1])
        squeezed = max(1, channels // self.ratio)
        self.pool = tf.keras.layers.GlobalAveragePooling2D(name="squeeze")
        self.reduce = tf.keras.layers.Dense(squeezed, activation="relu", name="reduce")
        self.expand = tf.keras.layers.Dense(channels, name="expand")
        super().build(input_shape)

    def call(self, inputs):
        weights = self.expand(self.reduce(self.pool(inputs)))
        if self.gate == "hardsigmoid":
            # MobileNetV3's gate: cheaper than sigmoid and quantizes cleanly.
            weights = tf.clip_by_value(weights / 6.0 + 0.5, 0.0, 1.0)
        else:
            weights = tf.nn.sigmoid(weights)
        return inputs * weights[:, None, None, :]

    def get_config(self):
        return {**super().get_config(), "ratio": self.ratio, "gate": self.gate}'''


PATCH_EMBEDDING_HELPER = '''class PatchEmbedding(tf.keras.layers.Layer):
    """Cut an image into non-overlapping patches and project each to a vector.

    Implemented as a convolution whose kernel equals its stride, which is
    exactly the reshape-and-matmul the ViT paper describes and what every
    implementation actually runs.
    """

    def __init__(self, patch_size, embed_dim, class_token=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.class_token = class_token

    def build(self, input_shape):
        self.project = tf.keras.layers.Conv2D(
            self.embed_dim, self.patch_size, strides=self.patch_size, name="projection"
        )
        # Built explicitly: this layer declares `compute_output_shape`, so Keras
        # takes the shape from that and never traces `call` symbolically, which
        # would otherwise leave the convolution unbuilt and weightless.
        self.project.build(input_shape)
        if self.class_token:
            self.cls = self.add_weight(
                name="class_token",
                shape=(1, 1, self.embed_dim),
                initializer="random_normal",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        patches = self.project(inputs)
        batch = tf.shape(patches)[0]
        tokens = tf.reshape(patches, (batch, -1, self.embed_dim))
        if self.class_token:
            tokens = tf.concat([tf.tile(self.cls, (batch, 1, 1)), tokens], axis=1)
        return tokens

    def compute_output_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        if height is None or width is None:
            return (input_shape[0], None, self.embed_dim)
        count = (height // self.patch_size) * (width // self.patch_size)
        return (input_shape[0], count + (1 if self.class_token else 0), self.embed_dim)

    def get_config(self):
        return {
            **super().get_config(),
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "class_token": self.class_token,
        }'''


LAYER_SCALE_HELPER = '''class LayerScale(tf.keras.layers.Layer):
    """A learned per-channel multiplier on a residual branch.

    Initialised near zero so a fresh block is almost the identity, which is what
    lets ConvNeXt and the deeper ViTs train without a warmup schedule.
    """

    def __init__(self, init_value=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.gamma

    def get_config(self):
        return {**super().get_config(), "init_value": self.init_value}'''


RESNET_HELPER = '''def resnet_stage(x, *, filters, blocks, stride=1, variant="basic", expansion=4, name="stage"):
    """One ResNet stage: `blocks` residual blocks, the first of which may downsample.

    The shortcut is a plain identity wherever the shape allows and a 1x1
    projection where it does not — that projection is the only place a ResNet
    stage changes width or resolution.

    Convolutions carry no bias: every one is followed by a BatchNormalization
    whose beta does the same job.
    """

    out_channels = filters * expansion if variant == "bottleneck" else filters

    for index in range(blocks):
        prefix = f"{name}_{index + 1}"
        block_stride = stride if index == 0 else 1
        residual = x

        if variant == "bottleneck":
            h = tf.keras.layers.Conv2D(filters, 1, use_bias=False, name=f"{prefix}_conv1")(x)
            h = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn1")(h)
            h = tf.keras.layers.Activation("relu", name=f"{prefix}_relu1")(h)
            h = tf.keras.layers.Conv2D(
                filters, 3, strides=block_stride, padding="same", use_bias=False, name=f"{prefix}_conv2"
            )(h)
            h = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn2")(h)
            h = tf.keras.layers.Activation("relu", name=f"{prefix}_relu2")(h)
            h = tf.keras.layers.Conv2D(out_channels, 1, use_bias=False, name=f"{prefix}_conv3")(h)
            h = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn3")(h)
        else:
            h = tf.keras.layers.Conv2D(
                filters, 3, strides=block_stride, padding="same", use_bias=False, name=f"{prefix}_conv1"
            )(x)
            h = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn1")(h)
            h = tf.keras.layers.Activation("relu", name=f"{prefix}_relu1")(h)
            h = tf.keras.layers.Conv2D(
                out_channels, 3, padding="same", use_bias=False, name=f"{prefix}_conv2"
            )(h)
            h = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn2")(h)

        if int(residual.shape[-1]) != out_channels or block_stride != 1:
            residual = tf.keras.layers.Conv2D(
                out_channels, 1, strides=block_stride, use_bias=False, name=f"{prefix}_shortcut"
            )(residual)
            residual = tf.keras.layers.BatchNormalization(name=f"{prefix}_shortcut_bn")(residual)

        x = tf.keras.layers.Add(name=f"{prefix}_add")([residual, h])
        x = tf.keras.layers.Activation("relu", name=f"{prefix}_out")(x)
    return x'''


INVERTED_RESIDUAL_HELPER = '''def inverted_residual(x, *, filters, expand_ratio=6, kernel_size=3, stride=1,
                      use_se=False, se_ratio=4, activation="relu6", blocks=1,
                      name="mbconv"):
    """MobileNetV2's inverted residual, with MobileNetV3's optional SE.

    "Inverted" because it expands into a wide depthwise convolution and projects
    back down, rather than the bottleneck-then-widen a ResNet block does — the
    expensive 3x3 then costs one filter per channel instead of C times as many.

    The projection has no activation. That is deliberate and load-bearing: a
    ReLU on a narrow tensor destroys information the residual then cannot
    recover, which is the linear-bottleneck argument the paper is named for.
    """

    def act(value, layer_name):
        if activation == "hardswish":
            return tf.keras.layers.Activation(
                lambda t: t * tf.clip_by_value(t + 3.0, 0.0, 6.0) / 6.0, name=layer_name
            )(value)
        if activation == "relu6":
            return tf.keras.layers.Activation(
                lambda t: tf.clip_by_value(t, 0.0, 6.0), name=layer_name
            )(value)
        return tf.keras.layers.Activation(activation, name=layer_name)(value)

    for index in range(blocks):
        prefix = f"{name}_{index + 1}"
        block_stride = stride if index == 0 else 1
        source = int(x.shape[-1])
        hidden = source * expand_ratio
        residual = x

        h = x
        if expand_ratio != 1:
            h = tf.keras.layers.Conv2D(hidden, 1, use_bias=False, name=f"{prefix}_expand")(h)
            h = tf.keras.layers.BatchNormalization(name=f"{prefix}_expand_bn")(h)
            h = act(h, f"{prefix}_expand_act")

        h = tf.keras.layers.DepthwiseConv2D(
            kernel_size, strides=block_stride, padding="same", use_bias=False, name=f"{prefix}_depthwise"
        )(h)
        h = tf.keras.layers.BatchNormalization(name=f"{prefix}_depthwise_bn")(h)
        h = act(h, f"{prefix}_depthwise_act")

        if use_se:
            h = SqueezeExcite(se_ratio, gate="hardsigmoid", name=f"{prefix}_se")(h)

        h = tf.keras.layers.Conv2D(filters, 1, use_bias=False, name=f"{prefix}_project")(h)
        h = tf.keras.layers.BatchNormalization(name=f"{prefix}_project_bn")(h)

        # Only connect where the shapes already agree — no projection shortcut.
        if block_stride == 1 and source == filters:
            x = tf.keras.layers.Add(name=f"{prefix}_add")([residual, h])
        else:
            x = h
    return x'''


DENSE_BLOCK_HELPER = '''def dense_block(x, *, growth_rate, layers, bottleneck_ratio=4, name="dense_block"):
    """DenseNet-BC: every layer reads every earlier layer's output.

    Each layer contributes `growth_rate` channels and the block's width grows
    linearly, which is why the growth rate is small (32 in DenseNet-121) and the
    1x1 bottleneck is there at all.
    """

    for index in range(layers):
        prefix = f"{name}_{index + 1}"
        h = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn1")(x)
        h = tf.keras.layers.Activation("relu", name=f"{prefix}_relu1")(h)
        h = tf.keras.layers.Conv2D(
            growth_rate * bottleneck_ratio, 1, use_bias=False, name=f"{prefix}_conv1"
        )(h)
        h = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn2")(h)
        h = tf.keras.layers.Activation("relu", name=f"{prefix}_relu2")(h)
        h = tf.keras.layers.Conv2D(
            growth_rate, 3, padding="same", use_bias=False, name=f"{prefix}_conv2"
        )(h)
        x = tf.keras.layers.Concatenate(name=f"{prefix}_concat")([x, h])
    return x'''


INCEPTION_HELPER = '''def inception_module(x, *, filters_1x1, reduce_3x3, filters_3x3, reduce_5x5,
                     filters_5x5, filters_pool, name="inception"):
    """GoogLeNet's module: four receptive fields in parallel, concatenated.

    The 1x1 reductions before the 3x3 and 5x5 paths are the whole trick — they
    cut the channel count first, so the wide kernels cost a fraction of what
    they otherwise would.
    """

    path_1 = tf.keras.layers.Conv2D(filters_1x1, 1, padding="same", activation="relu", name=f"{name}_1x1")(x)

    path_3 = tf.keras.layers.Conv2D(reduce_3x3, 1, padding="same", activation="relu", name=f"{name}_3x3_reduce")(x)
    path_3 = tf.keras.layers.Conv2D(filters_3x3, 3, padding="same", activation="relu", name=f"{name}_3x3")(path_3)

    path_5 = tf.keras.layers.Conv2D(reduce_5x5, 1, padding="same", activation="relu", name=f"{name}_5x5_reduce")(x)
    path_5 = tf.keras.layers.Conv2D(filters_5x5, 5, padding="same", activation="relu", name=f"{name}_5x5")(path_5)

    path_pool = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same", name=f"{name}_pool")(x)
    path_pool = tf.keras.layers.Conv2D(filters_pool, 1, padding="same", activation="relu", name=f"{name}_pool_proj")(path_pool)

    return tf.keras.layers.Concatenate(name=f"{name}_concat")([path_1, path_3, path_5, path_pool])'''


CONVNEXT_HELPER = '''def convnext_stage(x, *, filters, blocks, kernel_size=7, expand_ratio=4,
                   layer_scale=1e-6, name="convnext"):
    """ConvNeXt: a transformer block's layout, built from convolutions.

    Large depthwise kernel standing in for attention's receptive field, a single
    LayerNorm rather than BatchNorm after every convolution, one activation per
    block rather than three, and an inverted 4x bottleneck. Each of those is a
    ResNet design decision the paper reversed to match a Swin Transformer.
    """

    if int(x.shape[-1]) != filters:
        x = tf.keras.layers.Conv2D(filters, 1, name=f"{name}_project")(x)

    for index in range(blocks):
        prefix = f"{name}_{index + 1}"
        residual = x
        h = tf.keras.layers.DepthwiseConv2D(
            kernel_size, padding="same", name=f"{prefix}_depthwise"
        )(x)
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_norm")(h)
        h = tf.keras.layers.Dense(filters * expand_ratio, name=f"{prefix}_pointwise1")(h)
        h = tf.keras.layers.Activation("gelu", name=f"{prefix}_gelu")(h)
        h = tf.keras.layers.Dense(filters, name=f"{prefix}_pointwise2")(h)
        if layer_scale:
            h = LayerScale(layer_scale, name=f"{prefix}_scale")(h)
        x = tf.keras.layers.Add(name=f"{prefix}_add")([residual, h])
    return x'''


VIT_HELPER = '''def vit_encoder(x, *, layers, num_heads, head_dim, mlp_dim, dropout=0.0, name="vit"):
    """The ViT encoder: pre-LayerNorm attention and a GELU MLP, stacked.

    Structurally a GPT block with the causal mask off — a patch may attend to
    every other patch, because an image has no reading order.
    """

    for index in range(layers):
        prefix = f"{name}_{index + 1}"
        residual = x
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_attn_norm")(x)
        h = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_dim, dropout=dropout, name=f"{prefix}_attn"
        )(h, h)
        x = tf.keras.layers.Add(name=f"{prefix}_attn_add")([residual, h])

        residual = x
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_mlp_norm")(x)
        h = tf.keras.layers.Dense(mlp_dim, activation="gelu", name=f"{prefix}_mlp_up")(h)
        h = tf.keras.layers.Dropout(dropout, name=f"{prefix}_mlp_drop")(h)
        h = tf.keras.layers.Dense(int(x.shape[-1]), name=f"{prefix}_mlp_down")(h)
        x = tf.keras.layers.Add(name=f"{prefix}_mlp_add")([residual, h])
    return x'''
