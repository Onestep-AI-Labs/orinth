import type { CompletionContext, CompletionResult } from "@codemirror/autocomplete";

/**
 * Completion for Python that has no kernel behind it.
 *
 * The notebook completes from the live kernel, which is strictly better: it
 * introspects the real namespace. The architecture studio has no kernel — the
 * layer body is text that will be executed later, in a training process that
 * does not exist yet — so the alternative to a static table is nothing at all,
 * and "nothing at all" is what the studio shipped with.
 *
 * The table is deliberately the *writing* surface of a custom layer, not the
 * whole of TensorFlow: the namespaces you reach for inside `call()` and
 * `__init__`, in both frameworks the studio emits. A list that tried to be
 * complete would be stale within a release and would bury `Conv2D` under
 * fifty things nobody types.
 */

const KEYWORDS = [
  "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
  "elif", "else", "except", "False", "finally", "for", "from", "global", "if", "import",
  "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return",
  "True", "try", "while", "with", "yield"
];

const BUILTINS = [
  "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len",
  "list", "max", "min", "print", "range", "round", "set", "sorted", "str", "sum", "super",
  "tuple", "type", "zip"
];

/** Names worth offering with nothing typed: the imports the emitted code has. */
const ROOTS = ["tf", "keras", "layers", "torch", "nn", "F", "np", "self", "inputs", "x"];

//: Dotted namespace -> its members. Looked up by exact prefix, so `tf.keras.l`
//: resolves against `tf.keras` and filters to `layers`.
const NAMESPACES: Record<string, string[]> = {
  tf: [
    "keras", "nn", "math", "linalg", "random", "shape", "reshape", "cast", "concat",
    "stack", "split", "transpose", "expand_dims", "squeeze", "matmul", "einsum", "range",
    "zeros", "ones", "fill", "where", "clip_by_value", "reduce_mean", "reduce_sum",
    "reduce_max", "reduce_min", "sqrt", "exp", "sigmoid", "tanh", "Variable",
    "TensorShape", "float32", "int32", "bool", "newaxis", "function"
  ],
  "tf.keras": [
    "layers", "Model", "Sequential", "Input", "activations", "initializers",
    "regularizers", "constraints", "callbacks", "losses", "metrics", "optimizers",
    "backend", "saving", "utils"
  ],
  "tf.keras.layers": [
    "Layer", "Input", "Dense", "Conv1D", "Conv2D", "Conv3D", "Conv2DTranspose",
    "DepthwiseConv2D", "SeparableConv2D", "MaxPooling1D", "MaxPooling2D",
    "AveragePooling2D", "GlobalAveragePooling1D", "GlobalAveragePooling2D",
    "GlobalMaxPooling2D", "BatchNormalization", "LayerNormalization",
    "GroupNormalization", "UnitNormalization", "Dropout", "SpatialDropout2D",
    "GaussianNoise", "Activation", "ReLU", "LeakyReLU", "PReLU", "ELU", "Softmax",
    "Flatten", "Reshape", "Permute", "RepeatVector", "Cropping2D", "ZeroPadding2D",
    "UpSampling2D", "Resizing", "Rescaling", "Concatenate", "Add", "Subtract",
    "Multiply", "Average", "Maximum", "Minimum", "Dot", "Embedding", "LSTM", "GRU",
    "SimpleRNN", "Bidirectional", "TimeDistributed", "MultiHeadAttention", "Attention",
    "AdditiveAttention", "Lambda", "Masking", "Identity", "RandomFlip", "RandomRotation",
    "RandomZoom", "RandomContrast", "RandomTranslation"
  ],
  "tf.keras.activations": [
    "relu", "gelu", "silu", "swish", "elu", "selu", "mish", "softmax", "sigmoid", "tanh",
    "softplus", "softsign", "hard_sigmoid", "exponential", "linear"
  ],
  "tf.keras.initializers": [
    "GlorotUniform", "GlorotNormal", "HeNormal", "HeUniform", "LecunNormal", "Zeros",
    "Ones", "Constant", "RandomNormal", "RandomUniform", "TruncatedNormal", "Orthogonal",
    "Identity"
  ],
  "tf.keras.regularizers": ["l1", "l2", "l1_l2", "L1", "L2", "L1L2", "OrthogonalRegularizer"],
  "tf.keras.backend": ["int_shape", "shape", "epsilon", "floatx", "cast", "clear_session"],
  "tf.nn": [
    "relu", "relu6", "gelu", "silu", "swish", "elu", "selu", "leaky_relu", "softmax",
    "sigmoid", "tanh", "softplus", "dropout", "l2_normalize", "moments", "top_k",
    "conv2d", "max_pool2d", "avg_pool2d", "depth_to_space", "space_to_depth",
    "batch_normalization", "embedding_lookup"
  ],
  "tf.math": [
    "reduce_mean", "reduce_sum", "reduce_std", "reduce_max", "reduce_min", "sqrt",
    "rsqrt", "square", "abs", "exp", "log", "pow", "maximum", "minimum", "sigmoid",
    "tanh", "erf", "floor", "ceil", "round", "divide_no_nan", "l2_normalize", "cumsum"
  ],
  "tf.random": ["normal", "uniform", "shuffle", "set_seed", "stateless_normal"],
  "tf.linalg": ["matmul", "matvec", "norm", "diag", "band_part", "trace"],
  // `self` inside a Layer subclass. These are the members the emitted base
  // class actually gives you, which is the point of listing them.
  self: [
    "add_weight", "build", "built", "call", "compute_output_shape", "get_config",
    "from_config", "trainable", "trainable_weights", "non_trainable_weights", "weights",
    "dtype", "name", "supports_masking", "add_loss", "losses"
  ],
  torch: [
    "nn", "cat", "stack", "split", "chunk", "matmul", "einsum", "mean", "sum", "sqrt",
    "exp", "log", "sigmoid", "tanh", "softmax", "clamp", "zeros", "ones", "arange",
    "randn", "permute", "reshape", "transpose", "unsqueeze", "squeeze", "Tensor",
    "float32", "no_grad"
  ],
  nn: [
    "Module", "Sequential", "ModuleList", "ModuleDict", "Parameter", "Linear", "Conv1d",
    "Conv2d", "ConvTranspose2d", "BatchNorm1d", "BatchNorm2d", "LayerNorm", "GroupNorm",
    "Dropout", "Dropout2d", "ReLU", "LeakyReLU", "GELU", "SiLU", "ELU", "Sigmoid",
    "Tanh", "Softmax", "Flatten", "Identity", "Embedding", "LSTM", "GRU",
    "MultiheadAttention", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "functional"
  ],
  "nn.functional": [
    "relu", "gelu", "silu", "elu", "softmax", "log_softmax", "sigmoid", "tanh", "dropout",
    "normalize", "pad", "interpolate", "layer_norm", "scaled_dot_product_attention"
  ],
  F: [
    "relu", "gelu", "silu", "elu", "softmax", "log_softmax", "sigmoid", "tanh", "dropout",
    "normalize", "pad", "interpolate", "layer_norm", "scaled_dot_product_attention"
  ],
  np: [
    "array", "asarray", "zeros", "ones", "arange", "linspace", "concatenate", "stack",
    "expand_dims", "squeeze", "reshape", "transpose", "mean", "sum", "std", "sqrt",
    "exp", "log", "clip", "where", "argmax", "argmin", "float32", "newaxis"
  ]
};

//: `tf.keras.layers` is the one people type dozens of times, so the emitted
//: alias is offered as a namespace of its own too.
NAMESPACES.keras = NAMESPACES["tf.keras"];
NAMESPACES.layers = NAMESPACES["tf.keras.layers"];

/** A capitalized name is a class; everything else is called. */
function kindOf(label: string): "class" | "function" | "property" {
  if (/^[A-Z]/.test(label)) return "class";
  if (/^[a-z_]+$/.test(label) && label === label.toLowerCase()) return "function";
  return "property";
}

/** Identifiers the author already wrote — the only local knowledge available. */
function documentWords(text: string, exclude: string): string[] {
  const found = new Set<string>();
  for (const match of text.matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)) {
    if (match[0] !== exclude && match[0].length > 2) found.add(match[0]);
  }
  return [...found];
}

/**
 * Completion over the table above plus the document's own identifiers.
 *
 * Fires on an explicit request and after a `.`, the same rule the kernel source
 * uses: a popup on every character in a twenty-line layer body is noise.
 */
export function pythonApiCompletions() {
  return (context: CompletionContext): CompletionResult | null => {
    const dotted = context.matchBefore(/[A-Za-z_][A-Za-z0-9_.]*/);
    if (!dotted && !context.explicit) return null;

    const text = dotted?.text ?? "";
    const cut = text.lastIndexOf(".");

    if (cut >= 0) {
      const namespace = text.slice(0, cut);
      const members = NAMESPACES[namespace];
      if (!members) return null;
      return {
        from: (dotted?.from ?? context.pos) + cut + 1,
        options: members.map((label) => ({ label, type: kindOf(label), detail: namespace })),
        validFor: /^[A-Za-z0-9_]*$/
      };
    }

    if (!context.explicit && text.length < 2) return null;
    const words = documentWords(context.state.doc.toString(), text);
    return {
      from: dotted?.from ?? context.pos,
      options: [
        ...ROOTS.map((label) => ({ label, type: "variable" as const, boost: 2 })),
        ...KEYWORDS.map((label) => ({ label, type: "keyword" as const })),
        ...BUILTINS.map((label) => ({ label, type: "function" as const })),
        ...words.map((label) => ({ label, type: "variable" as const, detail: "in file" }))
      ],
      validFor: /^[A-Za-z0-9_]*$/
    };
  };
}

export const PYTHON_NAMESPACES = NAMESPACES;
