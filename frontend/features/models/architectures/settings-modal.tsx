"use client";

import { Modal } from "@/features/platform/ui";

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
 * The shell is the shared `Modal`; what is left here is the studio's two
 * specifics: the body *is* the existing inspector (`.modal-panel
 * .arch-inspector` strips the rail chrome in CSS so one component serves both),
 * and the subtitle is an identifier, so it renders in the mono face.
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
  return (
    <Modal
      title={title}
      subtitle={subtitle}
      onClose={onClose}
      className="modal-panel-mono-subtitle"
    >
      {children}
    </Modal>
  );
}
