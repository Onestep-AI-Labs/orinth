"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";

export type ContextMenuItem =
  | { kind: "separator" }
  | {
      kind?: "item";
      label: string;
      shortcut?: string;
      icon?: React.ReactNode;
      danger?: boolean;
      disabled?: boolean;
      onSelect: () => void;
    };

export type ContextMenuState = {
  /** Viewport coordinates of the click that opened it. */
  x: number;
  y: number;
  items: ContextMenuItem[];
};

/**
 * The canvas right-click menu.
 *
 * Fixed-positioned against the viewport rather than the canvas, because the
 * event that opens it reports viewport coordinates and the canvas is a
 * transformed, scrollable surface. It flips back inside the window when opened
 * near an edge, so a right-click in the bottom-right corner is still usable.
 *
 * Every entry here is also a keyboard shortcut; the menu is the discoverable
 * path to the same commands, which is why each row prints its accelerator.
 */
export function CanvasContextMenu({
  state,
  onClose
}: {
  state: ContextMenuState | null;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useLayoutEffect(() => {
    if (!state || !ref.current) return;
    const { width, height } = ref.current.getBoundingClientRect();
    setPosition({
      x: Math.max(8, Math.min(state.x, window.innerWidth - width - 8)),
      y: Math.max(8, Math.min(state.y, window.innerHeight - height - 8))
    });
  }, [state]);

  useEffect(() => {
    if (!state) return;
    const dismiss = () => onClose();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    // `capture` so the menu closes before the click reaches the canvas and
    // starts a selection behind it.
    window.addEventListener("pointerdown", dismiss, true);
    window.addEventListener("wheel", dismiss, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", dismiss, true);
      window.removeEventListener("wheel", dismiss, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [state, onClose]);

  if (!state) return null;

  return (
    <div
      ref={ref}
      className="arch-context-menu"
      role="menu"
      style={{ left: position.x, top: position.y }}
      onContextMenu={(event) => event.preventDefault()}
    >
      {state.items.map((item, index) =>
        item.kind === "separator" ? (
          <hr className="arch-context-separator" key={`separator-${index}`} />
        ) : (
          <button
            type="button"
            role="menuitem"
            key={item.label}
            disabled={item.disabled}
            className={item.danger ? "arch-context-item arch-context-item-danger" : "arch-context-item"}
            onClick={() => {
              item.onSelect();
              onClose();
            }}
          >
            <span className="arch-context-icon" aria-hidden>
              {item.icon}
            </span>
            <span className="arch-context-label">{item.label}</span>
            {item.shortcut && <span className="arch-context-shortcut">{item.shortcut}</span>}
          </button>
        )
      )}
    </div>
  );
}
