"use client";

import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { IconButton } from "@/features/platform/ui";

/**
 * The settings dialog for whatever is selected on the canvas.
 *
 * Settings used to occupy a permanent right rail. On a node editor the canvas
 * is the scarce resource, and a rail that is empty until something is selected
 * spends a fifth of the window saying "select a node". Double-clicking the
 * thing you want to edit is the gesture every graph tool already trains, so the
 * rail became this — opened on demand, closed on Escape, gone the rest of the
 * time.
 *
 * The body is the existing inspector, unchanged: `.arch-modal .arch-inspector`
 * strips the panel chrome in CSS so the same component serves both.
 */
export function SettingsModal({
  title,
  subtitle,
  onClose,
  children
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const dialogRef = useRef<HTMLElement>(null);

  useEffect(() => {
    // Focus moves into the dialog so Escape and Tab land here rather than on
    // whatever was focused on the canvas behind it.
    dialogRef.current?.focus();
  }, []);

  return (
    <div
      className="arch-modal-overlay"
      role="presentation"
      onMouseDown={onClose}
      onContextMenu={(event) => event.preventDefault()}
    >
      <section
        ref={dialogRef}
        className="arch-modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="arch-modal-head">
          <div className="arch-modal-heading">
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <IconButton aria-label="Close settings" title="Close settings (Esc)" onClick={onClose}>
            <X size={18} />
          </IconButton>
        </header>
        <div className="arch-modal-body">{children}</div>
      </section>
    </div>
  );
}
