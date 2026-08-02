"use client";

import { useCallback, useEffect, useState } from "react";

type Axis = "horizontal" | "vertical";

/**
 * A draggable pane size, persisted per key.
 *
 * The studio's panes compete for one screen, and how much each deserves
 * depends entirely on what the user is doing — reading generated code wants a
 * tall drawer, wiring a graph wants none. Rather than guess, every edge is
 * draggable and remembers where it was left.
 *
 * Pointer events (not mouse) so trackpad, touch, and pen all work, with
 * capture so a fast drag that leaves the handle keeps tracking.
 */
export function useResizable(
  key: string,
  initial: number,
  {
    min,
    max,
    axis = "horizontal",
    invert = false
  }: { min: number; max: number; axis?: Axis; invert?: boolean }
) {
  const storageKey = `arch-pane:${key}`;
  const [size, setSize] = useState(initial);
  const [dragging, setDragging] = useState(false);

  // Read the persisted size after mount: localStorage is unavailable during
  // the server render, so seeding state from it would hydrate mismatched.
  useEffect(() => {
    const stored = window.localStorage.getItem(storageKey);
    if (stored === null) return;
    const parsed = Number(stored);
    if (Number.isFinite(parsed)) setSize(clamp(parsed, min, max));
  }, [storageKey, min, max]);

  const persist = useCallback(
    (value: number) => window.localStorage.setItem(storageKey, String(Math.round(value))),
    [storageKey]
  );

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLElement>) => {
      event.preventDefault();
      const handle = event.currentTarget;
      handle.setPointerCapture(event.pointerId);
      setDragging(true);
      const start = axis === "horizontal" ? event.clientX : event.clientY;
      const origin = size;
      let latest = origin;

      const onMove = (move: PointerEvent) => {
        const current = axis === "horizontal" ? move.clientX : move.clientY;
        const delta = (current - start) * (invert ? -1 : 1);
        latest = clamp(origin + delta, min, max);
        setSize(latest);
      };
      const onUp = () => {
        setDragging(false);
        handle.releasePointerCapture(event.pointerId);
        handle.removeEventListener("pointermove", onMove);
        handle.removeEventListener("pointerup", onUp);
        persist(latest);
      };
      handle.addEventListener("pointermove", onMove);
      handle.addEventListener("pointerup", onUp);
    },
    [axis, invert, min, max, size, persist]
  );

  // Keyboard resizing, so a pane is never mouse-only.
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLElement>) => {
      const decrease = axis === "horizontal" ? "ArrowLeft" : "ArrowUp";
      const increase = axis === "horizontal" ? "ArrowRight" : "ArrowDown";
      if (event.key !== decrease && event.key !== increase) return;
      event.preventDefault();
      const step = event.shiftKey ? 48 : 16;
      const direction = (event.key === increase ? 1 : -1) * (invert ? -1 : 1);
      setSize((value) => {
        const next = clamp(value + direction * step, min, max);
        persist(next);
        return next;
      });
    },
    [axis, invert, min, max, persist]
  );

  const handleProps = {
    role: "separator" as const,
    tabIndex: 0,
    "aria-orientation": (axis === "horizontal" ? "vertical" : "horizontal") as
      | "vertical"
      | "horizontal",
    "aria-valuenow": Math.round(size),
    "aria-valuemin": min,
    "aria-valuemax": max,
    onPointerDown,
    onKeyDown
  };

  return { size, dragging, handleProps };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
