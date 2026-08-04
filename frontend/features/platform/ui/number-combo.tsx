"use client";

import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";

/**
 * A number field with a menu of common values — an editable dropdown.
 *
 * The two halves answer different needs and neither replaces the other: the
 * menu is how you pick 12 or 24 without thinking, and the field is how you get
 * 17 when that is what the layout wants. A plain `<select>` cannot express the
 * second, and a plain number input makes the first a typing exercise.
 *
 * Built rather than using `<input list>` + `<datalist>`: native datalist gives
 * no control over the popup's appearance, renders differently in every browser,
 * and shows no affordance that the list exists at all.
 *
 * The draft-while-focused behaviour matches {@link NumberInput} — clearing the
 * field must not rewrite it to 0 under the caret.
 */
export function NumberCombo({
  value,
  onChange,
  options,
  min,
  max,
  step,
  unit,
  disabled,
  ariaLabel
}: {
  value: number;
  onChange: (value: number) => void;
  /** Preset values offered in the menu. */
  options: number[];
  min?: number;
  max?: number;
  step?: number;
  /** Suffix shown on each menu row, e.g. "px". */
  unit?: string;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<string | null>(null);
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

  function clamp(next: number): number {
    let result = next;
    if (min !== undefined) result = Math.max(min, result);
    if (max !== undefined) result = Math.min(max, result);
    return result;
  }

  function commit(raw: string) {
    setDraft(null);
    const parsed = Number(raw);
    if (raw.trim() === "" || Number.isNaN(parsed)) return;
    const next = clamp(parsed);
    if (next !== value) onChange(next);
  }

  return (
    <div className="number-combo" ref={rootRef}>
      <input
        type="number"
        inputMode="numeric"
        className="number-combo-input"
        aria-label={ariaLabel}
        value={draft ?? value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => {
          const raw = event.target.value;
          setDraft(raw);
          const parsed = Number(raw);
          if (raw.trim() !== "" && !Number.isNaN(parsed)) onChange(clamp(parsed));
        }}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") commit(event.currentTarget.value);
        }}
      />
      <button
        type="button"
        className="number-combo-trigger"
        aria-label={ariaLabel ? `${ariaLabel} presets` : "Presets"}
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
      >
        <ChevronDown size={14} aria-hidden />
      </button>
      {open && (
        <ul className="number-combo-menu" role="listbox">
          {options.map((option) => (
            <li key={option}>
              <button
                type="button"
                role="option"
                aria-selected={option === value}
                className={
                  option === value
                    ? "number-combo-option number-combo-option-active"
                    : "number-combo-option"
                }
                onClick={() => {
                  setDraft(null);
                  if (option !== value) onChange(clamp(option));
                  setOpen(false);
                }}
              >
                <span>
                  {option}
                  {unit ? ` ${unit}` : ""}
                </span>
                {option === value && <Check size={14} aria-hidden />}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
