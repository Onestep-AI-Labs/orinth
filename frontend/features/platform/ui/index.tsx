"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, ImageIcon, RefreshCw, Trash2 } from "lucide-react";
import { formatSeconds } from "@/features/platform/utils";
import type { JobProgress } from "@/types/api";
import { Badge, Button, IconButton } from "./primitives";

export { Badge, Button, ButtonLink, IconButton, badgeVariants, buttonVariants } from "./primitives";
export type { BadgeProps, ButtonLinkProps, ButtonProps, IconButtonProps } from "./primitives";
export { cn } from "./cn";

export type ConfirmationTone = "danger" | "warning";
export type ConfirmationDialogOptions = {
  title: string;
  message: string;
  confirmLabel: string;
  tone?: ConfirmationTone;
  onConfirm: () => void | Promise<void>;
};

export function useConfirmationDialog() {
  const [dialog, setDialog] = useState<ConfirmationDialogOptions | null>(null);
  const confirm = useCallback((options: ConfirmationDialogOptions) => {
    setDialog(options);
  }, []);
  const close = useCallback(() => setDialog(null), []);
  const confirmationDialog = dialog ? (
    <ConfirmationDialog
      {...dialog}
      onCancel={close}
      onConfirm={async () => {
        await dialog.onConfirm();
        close();
      }}
    />
  ) : null;

  return { confirm, confirmationDialog };
}

export function ConfirmationDialog({
  title,
  message,
  confirmLabel,
  tone = "danger",
  onCancel,
  onConfirm
}: ConfirmationDialogOptions & { onCancel: () => void; onConfirm: () => void | Promise<void> }) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onCancel();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  return (
    <div className="confirmation-overlay" role="presentation" onMouseDown={onCancel}>
      <section
        className={`confirmation-dialog confirmation-dialog-${tone}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirmation-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="confirmation-dialog-header">
          <span className="confirmation-dialog-icon" aria-hidden="true">
            <AlertTriangle size={19} />
          </span>
          <div>
            <h2 id="confirmation-dialog-title">{title}</h2>
            <p>{message}</p>
          </div>
        </div>
        <div className="confirmation-dialog-actions">
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant={tone === "danger" ? "danger" : "primary"} onClick={onConfirm} autoFocus>
            {confirmLabel}
          </Button>
        </div>
      </section>
    </div>
  );
}

export function ProgressPanel({ progress, status, error }: { progress: JobProgress; status: string; error?: string | null }) {
  const progressLabel = progress.total ? `${progress.processed}/${progress.total}` : `${progress.percent.toFixed(0)}%`;
  const percent = Math.max(0, Math.min(100, progress.percent));
  return (
    <div className="progress-panel">
      <div className="flex items-center justify-between gap-3">
        <StatusBadge status={status} />
        <span className="text-sm text-ink-subtle">{progressLabel}</span>
      </div>
      <div className="progress-track"><span style={{ width: `${percent}%` }} /></div>
      <div className="progress-meta">
        <span>{progress.current_step}</span>
        <span>{formatSeconds(progress.elapsed_seconds)}</span>
      </div>
      {progress.current_item && <p className="text-sm text-ink-subtle">{progress.current_item}</p>}
      {error && <p className="error-text">{error}</p>}
      {progress.logs.length > 0 && (
        <div className="mini-log">
          {progress.logs.slice(-5).map((line, index) => <span key={`${line}-${index}`}>{line}</span>)}
        </div>
      )}
    </div>
  );
}

export function HistoryHeader({
  title,
  selectedCount,
  onRefresh,
  onDelete,
  onClear
}: {
  title: string;
  selectedCount: number;
  onRefresh: () => void;
  onDelete: () => void;
  onClear: () => void;
}) {
  return (
    <div className="history-header">
      <PanelTitle icon={<RefreshCw size={18} />} title={title} />
      <div className="history-actions">
        <IconButton aria-label="Refresh" onClick={onRefresh}><RefreshCw size={16} /></IconButton>
        <Button variant="secondary" onClick={onDelete} disabled={selectedCount === 0}><Trash2 size={16} /> Delete</Button>
        <Button variant="danger" onClick={onClear}><Trash2 size={16} /> Clear all</Button>
      </div>
    </div>
  );
}

export function Field({
  label,
  hint,
  children
}: {
  label: string;
  /** Secondary note shown on the right of the label row. Lives there rather
   *  than under the control so it adds no height — a caption below one field
   *  in a two-column row knocks its neighbour's control out of alignment. */
  hint?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="field">
      <label>
        <span>{label}</span>
        {hint ? <span className="field-label-hint">{hint}</span> : null}
      </label>
      {children}
    </div>
  );
}

/**
 * Numeric input that tolerates being emptied while you retype it.
 *
 * A plain `onChange={(e) => set(Number(e.target.value))}` turns "" into 0 the
 * moment you clear the field, so the caret jumps and you end up editing "0"
 * instead of an empty box. Here the draft stays a string while focused and is
 * only committed on blur, falling back to the last good value if what is left
 * cannot be parsed.
 */
export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  disabled
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);

  function commit(raw: string) {
    setDraft(null);
    const parsed = Number(raw);
    if (raw.trim() === "" || Number.isNaN(parsed)) return;
    let next = parsed;
    if (min !== undefined) next = Math.max(min, next);
    if (max !== undefined) next = Math.min(max, next);
    if (next !== value) onChange(next);
  }

  return (
    <input
      type="number"
      inputMode="decimal"
      value={draft ?? value}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(event) => {
        const raw = event.target.value;
        setDraft(raw);
        // Propagate only complete values, so a parent that derives state from
        // this field is not driven by half-typed input.
        const parsed = Number(raw);
        if (raw.trim() !== "" && !Number.isNaN(parsed)) onChange(parsed);
      }}
      onBlur={(event) => commit(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") commit((event.target as HTMLInputElement).value);
      }}
    />
  );
}

export function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="field">
      <label>{label} <span>{value.toFixed(2)}</span></label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </div>
  );
}

export function PageHeader({ title, subtitle, icon, eyebrow }: { title: string; subtitle: string; icon: React.ReactNode; eyebrow?: string }) {
  return (
    <header className="page-header">
      <div>
        <span>{icon}</span>
        <div>
          {eyebrow && <span className="page-eyebrow">{eyebrow}</span>}
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
    </header>
  );
}

export function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="mb-4 flex items-center gap-2">
      <span className="text-ink-subtle">{icon}</span>
      <h2 className="text-base font-semibold">{title}</h2>
    </div>
  );
}

export function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function PageSkeleton({ title = "Loading" }: { title?: string }) {
  return (
    <div className="space-y-5">
      <header className="page-header">
        <div>
          <span className="skeleton skeleton-icon" />
          <div className="skeleton-stack">
            <span className="skeleton skeleton-title" />
            <span className="skeleton skeleton-line short" />
          </div>
        </div>
      </header>
      <section className="panel">
        <span className="sr-only">{title}</span>
        <CardGridSkeleton count={4} />
      </section>
    </div>
  );
}

export function CardGridSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="skeleton-card-grid">
      {Array.from({ length: count }).map((_, index) => (
        <div className="skeleton-card" key={index}>
          <span className="skeleton skeleton-title" />
          <span className="skeleton skeleton-line" />
          <span className="skeleton skeleton-line short" />
          <div className="skeleton-chip-row">
            <span className="skeleton skeleton-chip" />
            <span className="skeleton skeleton-chip" />
            <span className="skeleton skeleton-chip" />
          </div>
        </div>
      ))}
    </div>
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="table-wrap skeleton-table">
      {Array.from({ length: rows }).map((_, index) => (
        <div className="skeleton-table-row" key={index}>
          <span className="skeleton skeleton-line" />
          <span className="skeleton skeleton-line" />
          <span className="skeleton skeleton-line short" />
        </div>
      ))}
    </div>
  );
}

export function InlineSpinner({ label }: { label: string }) {
  return (
    <span className="inline-spinner">
      <span className="spinner" />
      <span>{label}</span>
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const ok = status === "completed" || status === "editable" || status === "available";
  const failed = status === "failed" || status === "canceled" || status === "missing";
  const active = status === "running" || status === "queued" || status === "preparing";
  return <Badge tone={ok ? "ok" : failed ? "fail" : active ? "info" : "neutral"}>{status}</Badge>;
}

export function EmptyState({
  label,
  icon,
  centered,
  description,
  action
}: {
  label: string;
  icon?: React.ReactNode;
  centered?: boolean;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className={`empty-state${centered ? " empty-state-centered" : ""}`}>
      <span className="empty-state-icon">{icon ?? <ImageIcon size={centered ? 32 : 28} />}</span>
      {centered ? <strong>{label}</strong> : <span>{label}</span>}
      {description && <span>{description}</span>}
      {action && <div className="empty-state-action">{action}</div>}
    </div>
  );
}

export function MutationError({ mutations }: { mutations: Array<{ error: Error | null }> }) {
  const error = mutations.find((mutation) => mutation.error)?.error;
  return error ? <p className="error-text">{error.message}</p> : null;
}

export function toggleId(id: string, checked: boolean, selectedIds: string[], setSelectedIds: (ids: string[]) => void) {
  setSelectedIds(checked ? [...selectedIds, id] : selectedIds.filter((item) => item !== id));
}
