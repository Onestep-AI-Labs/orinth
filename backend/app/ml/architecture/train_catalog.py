"""Training-catalog entry for visually authored architectures (phase 17).

One option covers every graph. Which graph to train rides in
`hyperparameters.architecture_id`, the same open-ended dict phase 13 uses for
advanced values, so no schema or endpoint changes.

The advanced set is the Keras classification set minus the knobs the graph now
owns — `dropout` and `unfreeze_layers` are nodes and node params, not
hyperparameters, and offering them here would let a form value silently
contradict the canvas. Augmentation is absent for the same reason, and now for
a stronger one: the augmentation nodes have landed (`random_flip`,
`random_rotation`, `random_zoom`, `random_contrast`), so it is expressed on the
canvas where the rest of the model lives.
"""

from app.ml.common.advanced import OPTIMIZATION, REGULARIZATION, RUNTIME, number, select, toggle
from app.ml.common.catalog import TrainingModelDefinition
from app.schemas import AdvancedParameterSpec

ARCHITECTURE_FAMILY = "architecture_graph"
ARCHITECTURE_OPTION_ID = "architecture_graph"
# The same graph contract over a labelled text corpus. A separate family
# because the dataset pipelines share nothing: one batches images, the other
# tokenizes and pads strings. The catalog entry is otherwise identical, so a
# text graph is trained with the same knobs an image graph is.
TEXT_FAMILY = "architecture_text"
TEXT_OPTION_ID = "architecture_text"
# Which graph to train rides under this key inside `hyperparameters`.
ARCHITECTURE_ID_KEY = "architecture_id"

ARCHITECTURE_ADVANCED_PARAMETERS: list[AdvancedParameterSpec] = [
    select(
        "lr_schedule",
        "LR schedule",
        options=["constant", "cosine", "step", "plateau"],
        default="constant",
        group=OPTIMIZATION,
    ),
    number(
        "early_stop_patience",
        "Early stop patience",
        default=0,
        group=OPTIMIZATION,
        minimum=0,
        maximum=50,
        integer=True,
        help="Stop after N epochs without val-accuracy gain; 0 disables.",
    ),
    number(
        "label_smoothing",
        "Label smoothing",
        default=0.0,
        group=REGULARIZATION,
        minimum=0.0,
        maximum=0.3,
        step=0.01,
    ),
    toggle("class_weighting", "Balance class weights", default=False, group=REGULARIZATION),
    number(
        "seed",
        "Seed",
        default=42,
        group=RUNTIME,
        minimum=0,
        maximum=1_000_000,
        integer=True,
        help="Determinism is best-effort on GPU.",
    ),
]


def architecture_training_options() -> list[TrainingModelDefinition]:
    return [
        TrainingModelDefinition(
            id=ARCHITECTURE_OPTION_ID,
            name="Visual architecture",
            family=ARCHITECTURE_FAMILY,
            task_types=["classification"],
            source="local",
            runnable=True,
            needs_download=False,
            description=(
                "Train a model you composed in the architecture studio. "
                "Input resolution comes from the graph's Input node."
            ),
            defaults={"epochs": 15, "batch_size": 32, "optimizer": "adam", "learning_rate": 0.001},
            advanced_parameters=ARCHITECTURE_ADVANCED_PARAMETERS,
        ),
        TrainingModelDefinition(
            id=TEXT_OPTION_ID,
            name="Visual architecture (text classifier)",
            family=TEXT_FAMILY,
            task_types=["text_classification"],
            source="local",
            runnable=True,
            needs_download=False,
            description=(
                "Train a text model you composed in the architecture studio. "
                "Sequence length comes from the graph's Input node and the vocabulary "
                "from its Embedding node."
            ),
            defaults={"epochs": 15, "batch_size": 32, "optimizer": "adam", "learning_rate": 0.001},
            advanced_parameters=ARCHITECTURE_ADVANCED_PARAMETERS,
        ),
    ]
