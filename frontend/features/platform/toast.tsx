"use client";

import { useEffect, useSyncExternalStore } from "react";
import { AlertTriangle, CheckCircle2, X } from "lucide-react";

export type ToastTone = "success" | "error";

export type ToastItem = {
  id: number;
  tone: ToastTone;
  message: string;
  action?: { label: string; href: string };
};

type ToastInput = { action?: ToastItem["action"] };

const MAX_TOASTS = 3;
const AUTO_DISMISS_MS = 6000;

let nextId = 1;
let toasts: ToastItem[] = [];
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

function push(tone: ToastTone, message: string, input?: ToastInput) {
  toasts = [...toasts, { id: nextId++, tone, message, action: input?.action }].slice(-MAX_TOASTS);
  emit();
}

export function dismissToast(id: number) {
  toasts = toasts.filter((item) => item.id !== id);
  emit();
}

export const toast = {
  success(message: string, input?: ToastInput) {
    push("success", message, input);
  },
  error(message: string, input?: ToastInput) {
    push("error", message, input);
  }
};

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot() {
  return toasts;
}

function getServerSnapshot(): ToastItem[] {
  return [];
}

function ToastCard({ item }: { item: ToastItem }) {
  useEffect(() => {
    let remaining = AUTO_DISMISS_MS;
    let startedAt = Date.now();
    let timer = window.setTimeout(() => dismissToast(item.id), remaining);

    const element = document.getElementById(`toast-${item.id}`);
    const pause = () => {
      window.clearTimeout(timer);
      remaining -= Date.now() - startedAt;
    };
    const resume = () => {
      startedAt = Date.now();
      timer = window.setTimeout(() => dismissToast(item.id), Math.max(remaining, 800));
    };
    element?.addEventListener("mouseenter", pause);
    element?.addEventListener("mouseleave", resume);
    return () => {
      window.clearTimeout(timer);
      element?.removeEventListener("mouseenter", pause);
      element?.removeEventListener("mouseleave", resume);
    };
  }, [item.id]);

  return (
    <div
      id={`toast-${item.id}`}
      className={`toast toast-${item.tone}`}
      role={item.tone === "error" ? "alert" : undefined}
    >
      <span className="toast-icon" aria-hidden="true">
        {item.tone === "error" ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
      </span>
      <span className="toast-message">{item.message}</span>
      {item.action && (
        <a className="toast-action" href={item.action.href}>
          {item.action.label}
        </a>
      )}
      <button className="toast-dismiss" type="button" aria-label="Dismiss notification" onClick={() => dismissToast(item.id)}>
        <X size={14} />
      </button>
    </div>
  );
}

export function Toaster() {
  const items = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  return (
    <div className="toast-viewport" role="status" aria-live="polite">
      {items.map((item) => (
        <ToastCard key={item.id} item={item} />
      ))}
    </div>
  );
}
