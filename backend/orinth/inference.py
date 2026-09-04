"""Run one input through a registered model, the way `/inference` does.

There are two ways to predict from a notebook and they answer different
questions:

- `orinth.models.predictor(id).predict(path, params)` loads the model into
  *this kernel* and returns raw detections. Fast in a loop, records nothing.
  That is the one the batch-inference template uses for a whole split.
- `orinth.inference.predict(...)` posts to the API. Slower per call, and the
  result is **recorded**: it gets an id, an overlay image, and a row on the
  Inference page. That is the one to use when the answer is something you want
  to point at later.

Both run the same predictor on the same weights. Neither is a shortcut around
the other.
"""

import builtins
from pathlib import Path

from orinth import _http, _workspace
from orinth.errors import OrinthError
from orinth.types import Prediction

#: An image upload plus a cold model load. The API's own inference route has no
#: shorter bound, and a 30s default would time out on the first YOLO call.
TIMEOUT = 600.0


def _prediction(payload: dict) -> Prediction:
    return Prediction(
        id=payload.get("id", ""),
        model_id=payload.get("model_id", ""),
        label=payload.get("image_level_label", ""),
        scores=dict(payload.get("class_scores") or {}),
        detections=builtins.list(payload.get("detections") or []),
        text=payload.get("nlp_result"),
        overlay_url=payload.get("overlay_url"),
    )


def predict(
    model_id: str,
    image: "str | Path | None" = None,
    *,
    text: str | None = None,
    project_id: str | None = None,
    confidence_threshold: float = 0.65,
    iou_threshold: float = 0.7,
    question: str | None = None,
    max_length: int = 120,
) -> Prediction:
    """One image or one string through one model.

    Exactly one of `image` or `text`, because the two take different predictors
    and passing both would leave the API to guess which one you meant.
    """
    if (image is None) == (text is None):
        raise OrinthError(
            "predict() takes exactly one of image= or text=. "
            "An image model and a text model are different predictors."
        )

    data = {
        "model_id": model_id,
        "project_id": project_id or _workspace.default_project_id(),
        "confidence_threshold": str(confidence_threshold),
        "iou_threshold": str(iou_threshold),
        "max_length": str(max_length),
    }
    if question:
        data["question"] = question
    if text is not None:
        data["text_content"] = text
        return _prediction(_http.post("/api/inference", data=data, timeout=TIMEOUT) or {})

    path = Path(image)  # type: ignore[arg-type]
    if not path.is_file():
        raise OrinthError(f"No such image: {path}")
    with path.open("rb") as handle:
        files = {"file": (path.name, handle.read())}
    return _prediction(_http.post("/api/inference", data=data, files=files, timeout=TIMEOUT) or {})


def list(  # noqa: A001 - matches `orinth.datasets.list`
    *, project_id: str | None = None, model_id: str | None = None, limit: int | None = None
) -> builtins.list[Prediction]:
    """Recorded predictions, newest first — the Inference page's own list."""
    params: dict[str, object] = {
        "project_id": project_id or _workspace.default_project_id(),
        # The route caps at 1000; asking for the page the caller wants is
        # cheaper than filtering a default 25 down to nothing and calling it
        # empty.
        "limit": min(int(limit), 1000) if limit else 100,
    }
    found = [_prediction(entry) for entry in _http.get("/api/inference", params=params) or []]
    if model_id is not None:
        found = [entry for entry in found if entry.model_id == model_id]
    return found


def get(inference_id: str) -> Prediction:
    return _prediction(_http.get(f"/api/inference/{inference_id}") or {})
