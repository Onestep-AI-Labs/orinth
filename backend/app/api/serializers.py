from app.core.defaults import DEFAULT_PROJECT_ID, DEFAULT_TASK_TYPE
from app.schemas import (
    ComparisonMetricSummary,
    EvaluationJobRead,
    TrainingJobCreate,
    TrainingJobRead,
)
from app.services.job_progress import progress_from_artifacts


def evaluation_job_read(job) -> EvaluationJobRead:
    return EvaluationJobRead(
        id=job.id,
        project_id=job.project_id or DEFAULT_PROJECT_ID,
        comparison_id=job.comparison_id or job.id,
        model_id=job.model_id,
        dataset_key=job.dataset_key,
        status=job.status,
        limit=job.limit,
        metrics=job.metrics or {},
        artifacts=job.artifacts or {},
        progress=progress_from_artifacts(job.artifacts),
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def comparison_metric_summary(job) -> ComparisonMetricSummary:
    metrics = job.metrics or {}
    return ComparisonMetricSummary(
        job_id=job.id,
        model_id=job.model_id,
        status=job.status,
        samples=metrics.get("samples"),
        accuracy=((metrics.get("image") or {}).get("overall") or {}).get("accuracy"),
        macro_f1=((metrics.get("image") or {}).get("overall") or {}).get("macro_f1"),
        text_accuracy=((metrics.get("text_classification") or {}).get("overall") or {}).get("accuracy"),
        text_macro_f1=((metrics.get("text_classification") or {}).get("overall") or {}).get("macro_f1"),
        rougeL=(metrics.get("summarization") or {}).get("rougeL"),
        exact_match=(metrics.get("question_answering") or {}).get("exact_match"),
        qa_f1=(metrics.get("question_answering") or {}).get("f1"),
        pixel_dice=(metrics.get("pixel") or {}).get("dice"),
        object_recall=(metrics.get("object") or {}).get("recall"),
    )


def training_job_read(job) -> TrainingJobRead:
    return TrainingJobRead(
        id=job.id,
        project_id=job.project_id or DEFAULT_PROJECT_ID,
        task_type=(job.parameters or {}).get("task_type", DEFAULT_TASK_TYPE),
        model_family=job.model_family,
        status=job.status,
        parameters=TrainingJobCreate.model_validate(job.parameters or {}),
        artifacts=job.artifacts or {},
        artifact_urls=(job.artifacts or {}).get("artifact_urls", {}),
        progress=progress_from_artifacts(job.artifacts),
        metrics=(job.artifacts or {}).get("metrics", {}),
        history=(job.artifacts or {}).get("history", []),
        curves=(job.artifacts or {}).get("curves", {}),
        promoted_model_id=job.promoted_model_id,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
