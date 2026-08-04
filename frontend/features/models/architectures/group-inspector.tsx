"use client";

import { Check, Frame, Trash2, Type } from "lucide-react";
import { Button, Field, NumberCombo, NumberInput } from "@/features/platform/ui";
import type { StoredGroup } from "./graph-state";
import {
  GROUP_TINTS,
  TITLE_DEFAULTS,
  TITLE_SIZE_PRESETS,
  TITLE_SIZE_RANGE,
  TITLE_TINT_INK,
  titleColorFor
} from "./group-frame";

/**
 * Format settings for the selected group frame, in the same rail the node
 * inspector uses.
 *
 * A frame has exactly the properties a node does not — a title, a colour, and a
 * size you set rather than one derived from a tensor — and cramming those onto
 * the frame itself meant a colour picker floating over the canvas and a title
 * you could only reach by clicking a 16px target.
 *
 * Position is deliberately absent. A frame is placed by dragging it and sized
 * by dragging its edges; typing coordinates for an annotation is a control
 * nobody reaches for, and it took the room the title controls needed.
 */
export function GroupInspector({
  group,
  onChange,
  onDelete
}: {
  group: StoredGroup;
  onChange: (patch: Partial<StoredGroup>) => void;
  onDelete: () => void;
}) {
  const tint = GROUP_TINTS[group.tint % GROUP_TINTS.length];
  const titleTint = group.titleTint ?? group.tint;
  const titleSize = group.titleSize ?? TITLE_DEFAULTS.size;

  return (
    <aside className="arch-inspector" aria-label="Group settings">
      <div className="arch-inspector-scroll">
        <div className="arch-inspector-head" style={{ background: tint.fill }}>
          <p className="arch-inspector-eyebrow" style={{ color: tint.edge }}>
            <span className="arch-palette-dot" style={{ background: tint.edge }} aria-hidden />
            <Frame size={12} aria-hidden />
            Group frame
          </p>
          <p className="arch-inspector-title">{group.title || "Untitled group"}</p>
          <p className="arch-inspector-desc">
            An annotation only — it holds no tensors and emits no code. Drag the frame to
            move it, its edges to resize, and its title to place the label.
          </p>
        </div>

        <div className="arch-inspector-body">
          <div className="arch-inspector-group">
            <p className="advanced-group-label">
              <Type size={12} aria-hidden /> Title
            </p>
            <Field label="Text">
              <input
                value={group.title}
                placeholder="Untitled group"
                onChange={(event) => onChange({ title: event.target.value })}
              />
            </Field>
            <Field label="Font size" hint="px">
              <NumberCombo
                value={titleSize}
                options={TITLE_SIZE_PRESETS}
                min={TITLE_SIZE_RANGE.min}
                max={TITLE_SIZE_RANGE.max}
                step={1}
                unit="px"
                ariaLabel="Title font size"
                onChange={(value) => onChange({ titleSize: Math.round(value) })}
              />
            </Field>
            <div className="arch-group-tints" role="radiogroup" aria-label="Title colour">
              <button
                type="button"
                role="radio"
                aria-checked={titleTint === TITLE_TINT_INK}
                aria-label="Default ink"
                title="Default ink"
                className={
                  titleTint === TITLE_TINT_INK
                    ? "arch-group-tint arch-group-tint-ink arch-group-tint-active"
                    : "arch-group-tint arch-group-tint-ink"
                }
                onClick={() => onChange({ titleTint: TITLE_TINT_INK })}
              >
                {titleTint === TITLE_TINT_INK && <Check size={13} aria-hidden />}
              </button>
              {GROUP_TINTS.map((option, index) => (
                <button
                  key={option.name}
                  type="button"
                  role="radio"
                  aria-checked={index === titleTint}
                  aria-label={option.name}
                  title={option.name}
                  className={
                    index === titleTint
                      ? "arch-group-tint arch-group-tint-active"
                      : "arch-group-tint"
                  }
                  style={{ background: option.edge }}
                  onClick={() => onChange({ titleTint: index })}
                >
                  {index === titleTint && <Check size={13} aria-hidden />}
                </button>
              ))}
            </div>
            {/* The size goes in as a custom property rather than `fontSize` so
                the stylesheet can cap it: the rail is a few hundred pixels wide
                and the ramp now reaches banner sizes, which would otherwise
                push the frame and title controls off the panel. */}
            <p
              className="arch-group-title-preview"
              style={
                {
                  "--title-preview-size": `${titleSize}px`,
                  color: titleColorFor(titleTint, group.tint)
                } as React.CSSProperties
              }
            >
              {group.title || "Untitled group"}
            </p>
          </div>

          <div className="arch-inspector-group">
            <p className="advanced-group-label">Frame colour</p>
            <div className="arch-group-tints" role="radiogroup" aria-label="Frame colour">
              {GROUP_TINTS.map((option, index) => (
                <button
                  key={option.name}
                  type="button"
                  role="radio"
                  aria-checked={index === group.tint}
                  aria-label={option.name}
                  title={option.name}
                  className={
                    index === group.tint
                      ? "arch-group-tint arch-group-tint-active"
                      : "arch-group-tint"
                  }
                  style={{ background: option.edge }}
                  onClick={() => onChange({ tint: index })}
                >
                  {index === group.tint && <Check size={13} aria-hidden />}
                </button>
              ))}
            </div>
          </div>

          <div className="arch-inspector-group">
            <p className="advanced-group-label">Size</p>
            <div className="form-grid">
              <Field label="Width">
                <NumberInput
                  value={Math.round(group.width)}
                  min={160}
                  max={8000}
                  step={10}
                  onChange={(value) => onChange({ width: Math.round(value) })}
                />
              </Field>
              <Field label="Height">
                <NumberInput
                  value={Math.round(group.height)}
                  min={120}
                  max={8000}
                  step={10}
                  onChange={(value) => onChange({ height: Math.round(value) })}
                />
              </Field>
            </div>
          </div>
        </div>
      </div>

      <div className="arch-inspector-foot">
        <Button variant="danger" size="sm" onClick={onDelete}>
          <Trash2 size={15} aria-hidden /> Remove frame
        </Button>
      </div>
    </aside>
  );
}
