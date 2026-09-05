"use client";

import { AlertTriangle, Check, Play, RotateCcw, Sparkles, Table2, Trash2, X } from "lucide-react";
import { ReadinessBadge } from "@/features/datasets/readiness-badge";
import { PrepTransformPanel } from "@/features/datasets/prep/transform-panel";
import {
  prepIsRunning,
  useApplyPrepMutation,
  useDatasetPrepQuery,
  useDatasetPrepStatusQuery,
  useDiscardStagedMutation,
  useRunPrepMutation,
  useUndoPrepMutation
} from "@/features/datasets/prep/prep-hooks";
import { PrepProgress, prepJob } from "@/features/datasets/prep/prep-progress";
import {
  useForegroundJob,
  usePublishForegroundJob
} from "@/features/platform/foreground-job";
import {
  Badge,
  Button,
  EmptyState,
  InlineSpinner,
  MutationError,
  PanelTitle,
  useConfirmationDialog
} from "@/features/platform/ui";
import { formatDatasetTask } from "@/features/platform/utils";
import type { DatasetPrepPlan, DatasetSummary, PrepDecision } from "@/types/api";

/**
 * What Orinth did to this dataset, and whether it can be trained on.
 *
 * This is the landing tab and the answer to the question the studio never used
 * to answer: is this data usable yet? Before phase 21 the only way to find out
 * was to start a training run.
 *
 * Every agent decision is listed with its source, its reasoning, and the raw
 * signal behind it. That last part is the point — "312 files across 3 folders:
 * normal, kista, granuloma" can be checked against what was actually uploaded,
 * where "classification, 85% confident" can only be believed. Auto-applying is
 * defensible because of it, not in spite of it.
 */

const SOURCE_LABEL: Record<string, string> = {
  detected: "from the files",
  heuristic: "rule",
  llm: "model",
  user: "you"
};

function DecisionRow({ decision }: { decision: PrepDecision }) {
  const value = Array.isArray(decision.value)
    ? decision.value.join(", ")
    : typeof decision.value === "object" && decision.value !== null
      ? Object.entries(decision.value as Record<string, unknown>)
          .map(([key, entry]) => `${key}: ${String(entry)}`)
          .join(", ")
      : String(decision.value ?? "—");

  return (
    <li className="prep-decision">
      <div className="prep-decision-head">
        <span className="prep-decision-field">{decision.field.replace(/_/g, " ")}</span>
        {/* `info` for a model, `neutral` for a rule — the same provenance pair
            the phase-11 recipe records use, so the distinction reads the same
            wherever it appears. */}
        <Badge tone={decision.source === "llm" ? "info" : "neutral"}>
          {SOURCE_LABEL[decision.source] ?? decision.source}
        </Badge>
      </div>
      <p className="prep-decision-value">{value}</p>
      {decision.rationale && <p className="form-caption">{decision.rationale}</p>}
      {decision.evidence && <p className="prep-decision-evidence">{decision.evidence}</p>}
    </li>
  );
}

export function DatasetOverviewTab({
  dataset,
  onGoToData
}: {
  dataset: DatasetSummary;
  onGoToData: () => void;
}) {
  const prepQuery = useDatasetPrepQuery(dataset.id);
  // Polls only while a run is moving, and keeps polling across a page reload —
  // the run lives on the server, so leaving and coming back must not lose the
  // readout.
  const statusQuery = useDatasetPrepStatusQuery(dataset.id);
  const runMutation = useRunPrepMutation();
  const applyMutation = useApplyPrepMutation();
  const undoMutation = useUndoPrepMutation();
  const discardMutation = useDiscardStagedMutation();
  const { confirm, confirmationDialog } = useConfirmationDialog();

  const plan: DatasetPrepPlan | null = prepQuery.data ?? null;
  const readiness = dataset.readiness;
  const prep = dataset.prep;
  const staged = prep?.staged_files ?? 0;
  const hasRun = Boolean(plan);
  const running = prepIsRunning(statusQuery.data) || readiness.busy;
  const busy = runMutation.isPending || applyMutation.isPending || running;
  // A dataset that already trains does not need preparing again, so the agent
  // stops being the headline action. Offering "Prepare with Orinth" as the
  // primary button on a ready dataset is what made re-running it look like the
  // expected next step rather than a redo.
  const ready = readiness.trainable;

  // A run this tab started is blocked behind the shell's overlay, so it is
  // published there instead of drawn under it. A run this browser did not start
  // — a Hub import still downloading, or a reload mid-run — raises no mutation
  // and therefore no overlay, and that is the case the panel below is for.
  const published = useForegroundJob();
  usePublishForegroundJob(
    runMutation.isPending
      ? (prepJob(statusQuery.data) ?? { label: "Orinth is reading this dataset", percent: null })
      : null
  );

  const blocking = readiness.checks.filter((check) => check.severity === "blocking");
  const advisories = readiness.checks.filter(
    (check) => check.severity === "advisory" && !check.passed
  );

  return (
    <div className="space-y-5">
      {confirmationDialog}

      <section className="panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle icon={<Sparkles size={18} />} title="Overview" />
          <div className="flex items-center gap-2">
            {prepQuery.isFetching && <InlineSpinner label="Refreshing" />}
            <ReadinessBadge readiness={readiness} />
          </div>
        </div>

        <p className="form-caption prep-summary">{readiness.summary}</p>

        {running && !published && <PrepProgress status={statusQuery.data} />}

        <div className="prep-actions action-row">
          {ready ? (
            <Button variant="primary" onClick={onGoToData}>
              <Table2 size={16} /> View data
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={() => runMutation.mutate({ datasetId: dataset.id })}
              disabled={busy}
            >
              <Play size={16} /> {hasRun ? "Try preparing again" : "Prepare with Orinth"}
            </Button>
          )}
          {ready && (
            <Button
              variant="secondary"
              onClick={() => runMutation.mutate({ datasetId: dataset.id })}
              disabled={busy}
            >
              <Play size={16} /> Re-run Orinth
            </Button>
          )}
          {hasRun && (
            <Button
              variant="secondary"
              onClick={() =>
                confirm({
                  title: "Undo Orinth's changes?",
                  message:
                    "The dataset goes back to how it was before Orinth prepared it. The " +
                    "uploaded files are kept, so you can prepare it again.",
                  confirmLabel: "Undo",
                  tone: "warning",
                  onConfirm: () => undoMutation.mutate(dataset.id)
                })
              }
              disabled={undoMutation.isPending}
            >
              <RotateCcw size={16} /> Undo
            </Button>
          )}
        </div>

        <MutationError mutations={[runMutation, applyMutation, undoMutation, discardMutation]} />
      </section>

      <section className="panel">
        <PanelTitle icon={<Check size={18} />} title="Readiness" />
        <ul className="prep-check-list">
          {blocking.concat(advisories).map((check) => (
            <li key={check.id} className={`prep-check ${check.passed ? "" : "prep-check-failed"}`}>
              <span className="prep-check-icon" aria-hidden="true">
                {check.passed ? (
                  <Check size={15} />
                ) : check.severity === "advisory" ? (
                  <AlertTriangle size={15} />
                ) : (
                  <X size={15} />
                )}
              </span>
              <span className="prep-check-body">
                <strong>{check.label}</strong>
                {!check.passed && check.detail && (
                  <span className="form-caption">{check.detail}</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      </section>

      {plan?.transform && <PrepTransformPanel transform={plan.transform} />}

      <section className="panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle icon={<Sparkles size={18} />} title="What Orinth found" />
          {plan && (
            <Badge tone={plan.engine.mode === "llm" ? "info" : "neutral"}>
              {plan.engine.mode === "llm" ? `model · ${plan.engine.model ?? "openrouter"}` : "file inspection"}
            </Badge>
          )}
        </div>

        {!plan ? (
          <EmptyState
            icon={<Sparkles size={26} />}
            label={staged ? "Not analysed yet" : "Nothing uploaded yet"}
            description={
              staged
                ? `${staged} file${staged === 1 ? "" : "s"} are waiting. Run Orinth to work out what they are.`
                : "Drop files into this dataset and Orinth will work out what they are."
            }
          />
        ) : (
          <>
            {plan.engine.notice && <p className="form-caption prep-notice">{plan.engine.notice}</p>}
            {plan.task_type && (
              <p className="prep-verdict">
                Read as <strong>{formatDatasetTask(plan.task_type)}</strong>
                {plan.labels.length > 0 && <> with {plan.labels.length} classes</>}.
              </p>
            )}
            {plan.needs_input && <p className="prep-needs-input">{plan.needs_input}</p>}
            <ul className="prep-decision-list">
              {plan.decisions.map((decision, index) => (
                <DecisionRow decision={decision} key={`${decision.field}-${index}`} />
              ))}
            </ul>
            {plan.warnings.length > 0 && (
              <ul className="prep-warning-list">
                {plan.warnings.map((warning) => (
                  <li key={warning}>
                    <AlertTriangle size={14} aria-hidden="true" /> {warning}
                  </li>
                ))}
              </ul>
            )}
            {staged > 0 && (
              <div className="prep-staged">
                <p className="form-caption">
                  The original upload ({staged} file{staged === 1 ? "" : "s"}) is kept so Orinth can
                  re-read it without you uploading again.
                </p>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() =>
                    confirm({
                      title: "Discard the original upload?",
                      message:
                        "The prepared dataset is unaffected, but Orinth will not be able to " +
                        "re-analyse this data without a fresh upload.",
                      confirmLabel: "Discard",
                      tone: "danger",
                      onConfirm: () => discardMutation.mutate(dataset.id)
                    })
                  }
                  disabled={discardMutation.isPending}
                >
                  <Trash2 size={15} /> Discard original files
                </Button>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
