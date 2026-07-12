"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, ImageIcon, RefreshCw, Trash2 } from "lucide-react";
import { formatSeconds } from "@/features/platform/utils";
import type { JobProgress } from "@/types/api";

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
          <button className="secondary-button" type="button" onClick={onCancel}>
            Cancel
          </button>
          <button className={tone === "danger" ? "danger-button" : "primary-button"} type="button" onClick={onConfirm} autoFocus>
            {confirmLabel}
          </button>
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
        <span className="text-sm text-slate-500">{progressLabel}</span>
      </div>
      <div className="progress-track"><span style={{ width: `${percent}%` }} /></div>
      <div className="progress-meta">
        <span>{progress.current_step}</span>
        <span>{formatSeconds(progress.elapsed_seconds)}</span>
      </div>
      {progress.current_item && <p className="text-sm text-slate-500">{progress.current_item}</p>}
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
        <button className="icon-button" onClick={onRefresh} title="Refresh"><RefreshCw size={16} /></button>
        <button className="secondary-button" onClick={onDelete} disabled={selectedCount === 0}><Trash2 size={16} /> Delete</button>
        <button className="danger-button" onClick={onClear}><Trash2 size={16} /> Clear all</button>
      </div>
    </div>
  );
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
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

export function PageHeader({ title, subtitle, icon }: { title: string; subtitle: string; icon: React.ReactNode }) {
  return (
    <header className="page-header">
      <div>
        <span>{icon}</span>
        <div>
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
      <span className="text-slate-500">{icon}</span>
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
  return <span className={`badge ${ok ? "badge-ok" : failed ? "badge-fail" : ""}`}>{status}</span>;
}

export function EmptyState({ label, icon, centered, description }: { label: string; icon?: React.ReactNode; centered?: boolean; description?: string }) {
  return (
    <div className={`empty-state${centered ? " empty-state-centered" : ""}`}>
      <span className="empty-state-icon">{icon ?? <ImageIcon size={centered ? 32 : 28} />}</span>
      {centered ? <strong>{label}</strong> : <span>{label}</span>}
      {description && <span>{description}</span>}
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
