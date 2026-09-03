"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ImageIcon,
  RefreshCw,
  Trash2
} from "lucide-react";
import { formatSeconds } from "@/features/platform/utils";
import type { JobProgress } from "@/types/api";
import { Badge, Button, IconButton } from "./primitives";

export { Badge, Button, ButtonLink, IconButton, Select, badgeVariants, buttonVariants } from "./primitives";
export type { BadgeProps, ButtonLinkProps, ButtonProps, IconButtonProps } from "./primitives";
export { cn } from "./cn";
export { TaskSelect } from "./task-select";
export { NumberCombo } from "./number-combo";
export { Explained, InfoTip } from "./info-tip";
export { Modal } from "./modal";

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

export function ProgressPanel({
  progress,
  status,
  error,
  hideLogs = false
}: {
  progress: JobProgress;
  status: string;
  error?: string | null;
  /** Suppress the inline mini-log when the surface renders its own, richer log
   *  view — otherwise the same lines appear twice on the page. */
  hideLogs?: boolean;
}) {
  const percent = Math.max(0, Math.min(100, progress.percent));
  // A unified single-bar percent is the honest primary number; the step/epoch
  // count is a secondary detail so the header never contradicts the bar.
  const count = progress.total ? `${progress.processed}/${progress.total}` : null;
  const lines = hideLogs ? [] : dedupeConsecutive(progress.logs).slice(-5);
  return (
    <div className="progress-panel">
      <div className="flex items-center justify-between gap-3">
        <StatusBadge status={status} />
        <span className="progress-readout">
          <strong>{percent.toFixed(0)}%</strong>
          {count && <span className="text-ink-subtle">{count}</span>}
        </span>
      </div>
      <div className="progress-track"><span style={{ width: `${percent}%` }} /></div>
      <div className="progress-meta">
        <span>{progress.current_step}</span>
        <span>
          {formatSeconds(progress.elapsed_seconds)}
          {progress.eta_seconds ? ` · ETA ${formatSeconds(progress.eta_seconds)}` : ""}
        </span>
      </div>
      {progress.current_item && <p className="text-sm text-ink-subtle">{progress.current_item}</p>}
      {error && <p className="error-text">{error}</p>}
      {lines.length > 0 && (
        <div className="mini-log">
          {lines.map((line, index) => <span key={`${line}-${index}`}>{line}</span>)}
        </div>
      )}
    </div>
  );
}

/** Collapse runs of identical adjacent lines (e.g. repeated "downloading base
 *  model: 43%" redraws) so the log reads as progress, not noise. */
export function dedupeConsecutive(lines: string[]): string[] {
  const out: string[] = [];
  for (const line of lines) {
    if (out[out.length - 1] !== line) out.push(line);
  }
  return out;
}

export function HistoryHeader({
  title,
  selectedCount,
  onRefresh,
  onDelete,
  onClear,
  dataTour
}: {
  title: string;
  selectedCount: number;
  onRefresh: () => void;
  onDelete: () => void;
  onClear: () => void;
  /** Guided-tour anchor, forwarded to the compact title. */
  dataTour?: string;
}) {
  return (
    <div className="history-header">
      <PanelTitle icon={<RefreshCw size={18} />} title={title} dataTour={dataTour} />
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

/**
 * Dropdown that selects several options at once.
 *
 * A native `<select multiple>` is the literal control for this, but it renders
 * as an always-open scroll box and requires ctrl/cmd-click to add a second
 * item, which almost nobody discovers. This keeps the closed, summarised
 * affordance of a dropdown and puts checkboxes in the panel.
 */
export function MultiSelect({
  options,
  values,
  onChange,
  placeholder = "Choose…",
  disabled
}: {
  options: Array<{ value: string; label: string; hint?: string }>;
  values: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const selected = options.filter((option) => values.includes(option.value));
  const summary =
    selected.length === 0
      ? placeholder
      : selected.length <= 2
        ? selected.map((option) => option.label).join(", ")
        : `${selected.length} selected`;

  function toggle(value: string) {
    onChange(values.includes(value) ? values.filter((item) => item !== value) : [...values, value]);
  }

  return (
    <div className="multiselect" ref={rootRef}>
      <button
        type="button"
        className="multiselect-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled || options.length === 0}
        onClick={() => setOpen((value) => !value)}
      >
        <span className={selected.length === 0 ? "multiselect-placeholder" : undefined}>{summary}</span>
        <ChevronDown size={16} />
      </button>
      {open && (
        <div className="multiselect-panel" role="listbox" aria-multiselectable="true">
          <div className="choice-list compact">
            {options.map((option) => (
              <label key={option.value}>
                <input
                  type="checkbox"
                  checked={values.includes(option.value)}
                  onChange={() => toggle(option.value)}
                />
                <span>
                  {option.label}
                  {option.hint ? <small className="multiselect-hint">{option.hint}</small> : null}
                </span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
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

export function PageHeader({
  title,
  subtitle,
  icon,
  eyebrow,
  actions
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  eyebrow?: string;
  /** Right-aligned status or controls, per DESIGN.md §6 page-header anatomy. */
  actions?: React.ReactNode;
}) {
  return (
    <header className={`page-header ${actions ? "page-header-with-actions" : ""}`}>
      <div>
        <span>{icon}</span>
        <div>
          {eyebrow && <span className="page-eyebrow">{eyebrow}</span>}
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
      {actions ? <div className="page-header-actions">{actions}</div> : null}
    </header>
  );
}

export function PanelTitle({
  icon,
  title,
  dataTour
}: {
  icon: React.ReactNode;
  title: string;
  /** Guided-tour anchor (see features/platform/tour). Kept on the compact title
   *  so a tour step spotlights the heading, not the whole panel. */
  dataTour?: string;
}) {
  return (
    <div className="mb-4 flex items-center gap-2" data-tour={dataTour}>
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

/**
 * Previous / next over a paged list.
 *
 * Numbered page links are deliberately absent: they are only useful when a
 * specific page means something, and on a grid of cards ordered by category it
 * does not. Renders nothing at all for a single page — a disabled pager on a
 * six-item list is chrome asserting there is more.
 */
export function Pager({
  page,
  pageCount,
  onChange,
  label,
  unit = "items"
}: {
  page: number;
  pageCount: number;
  onChange: (page: number) => void;
  /** Names the region for a screen reader: "Template pages". */
  label: string;
  unit?: string;
}) {
  if (pageCount <= 1) return null;
  return (
    <nav className="pager" aria-label={label}>
      <IconButton
        aria-label={`Previous page of ${unit}`}
        onClick={() => onChange(Math.max(0, page - 1))}
        disabled={page === 0}
      >
        <ChevronLeft size={17} />
      </IconButton>
      <span className="pager-status" aria-live="polite">
        {page + 1} / {pageCount}
      </span>
      <IconButton
        aria-label={`Next page of ${unit}`}
        onClick={() => onChange(Math.min(pageCount - 1, page + 1))}
        disabled={page >= pageCount - 1}
      >
        <ChevronRight size={17} />
      </IconButton>
    </nav>
  );
}

/**
 * A stack of rows, for a narrow panel or rail.
 *
 * `TableSkeleton` is three columns wide and reads as broken at rail width; this
 * is the same contract (initial query renders a skeleton, never an empty state)
 * in a shape that fits beside content rather than under a header.
 */
export function ListSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="skeleton-list">
      {Array.from({ length: rows }).map((_, index) => (
        <div className="skeleton-list-row" key={index}>
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
