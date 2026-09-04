"""Where a run could execute. Today: here.

Remote GPU rental — Vast.ai, Modal, RunPod — is planned and not built. The
providers are **declared** rather than omitted, for two reasons that pull the
same way:

- A roadmap nobody can see is a roadmap nobody can plan against. Someone
  deciding whether Orinth fits their workflow needs to know a remote target is
  coming and what it will need.
- `available: false` should be a fact the UI reads, not a state it invents. The
  alternative — the frontend hardcoding "coming soon" strings — puts the roadmap
  in two places and guarantees they disagree.

`requirements` is deliberately concrete. It doubles as the implementation
checklist, and it keeps this file honest: a provider whose requirements nobody
can write down is not a plan, it is a wish.
"""

from app.schemas import ComputeProvider

PROVIDERS: list[ComputeProvider] = [
    ComputeProvider(
        id="local",
        name="This machine",
        available=True,
        summary=(
            "Runs on the hardware the backend is running on. Devices and their memory are "
            "probed per framework."
        ),
        requirements=[],
    ),
    ComputeProvider(
        id="vast",
        name="Vast.ai",
        available=False,
        summary=(
            "Rented consumer GPUs, billed by the second. The cheapest way to reach a 4090 or "
            "an A100 for a single fine-tune."
        ),
        requirements=[
            "An API key in settings, stored like the OpenRouter key — never returned to a client.",
            "A dataset transfer path: instances have their own disk, so the run needs the split "
            "uploaded rather than a shared filesystem.",
            "Artifact retrieval, since an instance is destroyed when the run ends.",
            "A cost ceiling per job. Billing by the second with no cap is how a training run "
            "becomes an invoice nobody approved.",
        ],
        docs_url="https://vast.ai/docs/",
    ),
    ComputeProvider(
        id="modal",
        name="Modal",
        available=False,
        summary=(
            "Serverless GPU functions. Best fit for the batch shapes here — a training run is a "
            "function with a start and an end, not a server."
        ),
        requirements=[
            "A Modal token in settings.",
            "The runners packaged as a Modal image; they currently assume a local filesystem "
            "layout that `training_root` resolves.",
            "Log streaming back into the job's log tail, so a remote run reads the same as a "
            "local one on the job page.",
            "A volume for datasets and artifacts, or an upload/download step per run.",
        ],
        docs_url="https://modal.com/docs",
    ),
    ComputeProvider(
        id="runpod",
        name="RunPod",
        available=False,
        summary="Rented pods and serverless endpoints, positioned between Vast and Modal.",
        requirements=[
            "An API key in settings.",
            "The same dataset transfer and artifact retrieval Vast needs — the two share most "
            "of an implementation.",
        ],
        docs_url="https://docs.runpod.io/",
    ),
]


def providers() -> list[ComputeProvider]:
    return list(PROVIDERS)
