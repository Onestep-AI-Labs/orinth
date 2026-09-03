"use client";

import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { IconButton } from "./primitives";

/**
 * The dialog shell: overlay, panel, header, close.
 *
 * Lifted out of the architecture studio, which had the only one. There is
 * exactly one other overlay in the app — `ConfirmationDialog`, which is a fixed
 * question with two buttons and deliberately not configurable — so this is the
 * general case, and a third implementation would have been the point where the
 * app had three different dialogs that looked like two.
 *
 * Closes on Escape and on a click outside. `onMouseDown` rather than `onClick`
 * for the backdrop, so a drag that starts inside the panel and ends outside it
 * does not dismiss the thing being dragged.
 */
export function Modal({
  title,
  subtitle,
  onClose,
  children,
  footer,
  className = "",
  labelledBy
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: React.ReactNode;
  /** Actions, outside the scrolling body so they stay reachable. */
  footer?: React.ReactNode;
  className?: string;
  labelledBy?: string;
}) {
  const dialogRef = useRef<HTMLElement>(null);

  useEffect(() => {
    // Focus moves into the dialog so Escape and Tab land here rather than on
    // whatever was focused behind it.
    dialogRef.current?.focus();
  }, []);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      className="modal-overlay"
      role="presentation"
      onMouseDown={onClose}
      onContextMenu={(event) => event.preventDefault()}
    >
      <section
        ref={dialogRef}
        className={`modal-panel ${className}`}
        role="dialog"
        aria-modal="true"
        aria-label={labelledBy ? undefined : title}
        aria-labelledby={labelledBy}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="modal-head">
          <div className="modal-heading">
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <IconButton aria-label="Close" title="Close (Esc)" onClick={onClose}>
            <X size={18} />
          </IconButton>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-actions">{footer}</div>}
      </section>
    </div>
  );
}
