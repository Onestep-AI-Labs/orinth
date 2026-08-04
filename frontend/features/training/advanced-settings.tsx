"use client";

import { useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import { Field, NumberInput, Select } from "@/features/platform/ui";
import type { AdvancedParameterSpec } from "@/types/api";

// The group micro-headers render in this order; anything unrecognised trails.
// Mirrors the constants in `backend/app/ml/common/advanced.py` — the first
// three groups are LLM-specific (phase 14).
const GROUP_ORDER = ["Method", "LoRA", "Quantization", "Sequence", "Optimization", "Augmentation", "Regularization", "Generation", "Runtime"];

export type AdvancedValues = Record<string, unknown>;

/** Catalog defaults for a spec list, ready to seed form state. */
export function advancedDefaults(params: AdvancedParameterSpec[]): AdvancedValues {
  const values: AdvancedValues = {};
  for (const spec of params) {
    values[spec.key] = spec.type === "multiselect" ? (spec.default ?? []) : spec.default;
  }
  return values;
}

function groupOrderIndex(group: string): number {
  const index = GROUP_ORDER.indexOf(group);
  return index === -1 ? GROUP_ORDER.length : index;
}

/**
 * Generic, catalog-driven renderer for a model option's advanced parameters.
 *
 * It has no per-family knowledge: fields come entirely from `type`/`min`/`max`/
 * `step`/`options` and are grouped by `group`. A new model family gets its
 * advanced UI purely by declaring specs on the backend. Collapsed by default —
 * a closed accordion signals "defaults are fine".
 */
export function AdvancedSettings({
  params,
  values,
  onChange
}: {
  params: AdvancedParameterSpec[];
  values: AdvancedValues;
  onChange: (key: string, value: unknown) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const groups = useMemo(() => {
    const byGroup = new Map<string, AdvancedParameterSpec[]>();
    for (const spec of params) {
      const list = byGroup.get(spec.group) ?? [];
      list.push(spec);
      byGroup.set(spec.group, list);
    }
    return [...byGroup.entries()].sort(
      ([a], [b]) => groupOrderIndex(a) - groupOrderIndex(b) || a.localeCompare(b)
    );
  }, [params]);

  const changedCount = useMemo(
    () => params.filter((spec) => !isDefault(spec, values[spec.key])).length,
    [params, values]
  );

  if (params.length === 0) return null;

  return (
    <div className="accordion-panel advanced-panel">
      <div className="accordion-header">
        <button
          className="accordion-title"
          type="button"
          onClick={() => setExpanded((value) => !value)}
          aria-expanded={expanded}
        >
          <ChevronDown className={`accordion-icon ${expanded ? "" : "accordion-icon-collapsed"}`} size={17} />
          <span>Advanced settings</span>
        </button>
        {changedCount > 0 && (
          <span className="advanced-changed-count">
            {changedCount} changed
          </span>
        )}
      </div>
      {expanded && (
        <div className="accordion-body">
          {groups.map(([group, specs]) => (
            <div className="advanced-group" key={group}>
              <p className="advanced-group-label">{group}</p>
              <div className="form-grid form-grid-two">
                {specs.map((spec) => (
                  <AdvancedField
                    key={spec.key}
                    spec={spec}
                    value={values[spec.key]}
                    onChange={(value) => onChange(spec.key, value)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function isDefault(spec: AdvancedParameterSpec, value: unknown): boolean {
  if (spec.type === "multiselect") {
    const current = Array.isArray(value) ? value : [];
    const base = Array.isArray(spec.default) ? spec.default : [];
    return current.length === base.length && current.every((item) => base.includes(item));
  }
  return value === spec.default;
}

/**
 * One catalog-driven field. Exported because the architecture studio's node
 * inspector (phase 17) renders `NodeSpec.params` — the same
 * `AdvancedParameterSpec` type — so a new node type gets its inspector UI
 * without a second field renderer.
 */
export function AdvancedField({
  spec,
  value,
  onChange
}: {
  spec: AdvancedParameterSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (spec.type === "bool") {
    return (
      <label className="toggle-row advanced-toggle">
        <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
        <span>{spec.label}</span>
        {spec.help ? <span className="field-hint advanced-help">{spec.help}</span> : null}
      </label>
    );
  }

  if (spec.type === "code") {
    // Source code needs room and a mono face; a single-line input makes even a
    // short layer body unreadable.
    const lines = String(value ?? "").split("\n").length;
    return (
      <Field label={spec.label} hint={spec.help ?? undefined}>
        <textarea
          className="code-field"
          value={String(value ?? "")}
          spellCheck={false}
          rows={Math.min(Math.max(lines + 1, 3), 24)}
          onChange={(event) => onChange(event.target.value)}
        />
      </Field>
    );
  }

  if (spec.type === "select") {
    return (
      <Field label={spec.label} hint={spec.help ?? undefined}>
        <Select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
          {spec.options.map((option) => (
            <option key={String(option)} value={String(option)}>
              {String(option)}
            </option>
          ))}
        </Select>
      </Field>
    );
  }

  if (spec.type === "multiselect") {
    const selected = Array.isArray(value) ? value.map(String) : [];
    return (
      <Field label={spec.label} hint={spec.help ?? undefined}>
        <div className="advanced-multiselect">
          {spec.options.map((option) => {
            const key = String(option);
            const checked = selected.includes(key);
            return (
              <label className="toggle-row" key={key}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={(event) =>
                    onChange(
                      event.target.checked
                        ? [...selected, key]
                        : selected.filter((item) => item !== key)
                    )
                  }
                />
                <span>{key}</span>
              </label>
            );
          })}
        </div>
      </Field>
    );
  }

  if (spec.type === "text") {
    return (
      <Field label={spec.label} hint={spec.help ?? undefined}>
        <input value={String(value ?? "")} onChange={(event) => onChange(event.target.value)} />
      </Field>
    );
  }

  // int / float
  return (
    <Field label={spec.label} hint={spec.help ?? undefined}>
      <NumberInput
        value={Number(value ?? 0)}
        onChange={(next) => onChange(spec.type === "int" ? Math.round(next) : next)}
        min={spec.min ?? undefined}
        max={spec.max ?? undefined}
        step={spec.step ?? (spec.type === "int" ? 1 : undefined)}
      />
    </Field>
  );
}
