"use client";

import { useMemo, useState } from "react";
import { Layers, Search } from "lucide-react";
import type { NodeSpec } from "@/types/api";
import { CategoryIcon, categoryColorVar, categoryTint } from "./node-visuals";

type Filter = "all" | "block" | "layer";

const FILTERS: { value: Filter; label: string; hint: string }[] = [
  { value: "all", label: "All", hint: "Every node type" },
  { value: "block", label: "Blocks", hint: "Named multi-layer structures — a ResNet stage, a Qwen3 decoder block" },
  { value: "layer", label: "Layers", hint: "One operation at a time — a convolution, a norm, an activation" }
];

/**
 * The layer palette: categories from the backend catalog, filtered by a search
 * across name, type, and description, and by whether an entry is one layer or a
 * whole block.
 *
 * The Blocks/Layers split is the palette's most useful control. "Which
 * operation comes next" and "which architecture am I building" are different
 * questions, and a flat list of sixty-five entries answers neither well — a
 * `qwen3_block` and a `dropout` do not belong in the same scan.
 *
 * Clicking adds a node at the canvas centre; dragging drops it where released.
 * Both paths exist because click is the discoverable one and drag is the fast
 * one, and a node editor that only supports drag is unusable by keyboard.
 */
export function NodePalette({
  specs,
  categories,
  onAdd,
  disabled
}: {
  specs: NodeSpec[];
  categories: string[];
  onAdd: (spec: NodeSpec) => void;
  disabled?: boolean;
}) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const grouped = useMemo(() => {
    const query = search.trim().toLowerCase();
    let matches = filter === "all" ? specs : specs.filter((spec) => spec.kind === filter);
    if (query) {
      matches = matches.filter((spec) =>
        `${spec.name} ${spec.type} ${spec.description} ${spec.source}`.toLowerCase().includes(query)
      );
    }
    const order = new Map(categories.map((category, index) => [category, index]));
    const byCategory = new Map<string, NodeSpec[]>();
    for (const spec of matches) {
      byCategory.set(spec.category, [...(byCategory.get(spec.category) ?? []), spec]);
    }
    return [...byCategory.entries()].sort(
      ([a], [b]) => (order.get(a) ?? 99) - (order.get(b) ?? 99) || a.localeCompare(b)
    );
  }, [specs, categories, search, filter]);

  const counts = useMemo(
    () => ({
      all: specs.length,
      block: specs.filter((spec) => spec.kind === "block").length,
      layer: specs.filter((spec) => spec.kind === "layer").length
    }),
    [specs]
  );

  return (
    <aside className="arch-palette" aria-label="Layer palette">
      <div className="arch-palette-search">
        <Search size={15} aria-hidden />
        <input
          type="search"
          value={search}
          placeholder="Find a layer or block"
          aria-label="Find a layer or block"
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>
      <div className="arch-palette-filter segmented-control" role="group" aria-label="Node kind">
        {FILTERS.map((option) => (
          <button
            key={option.value}
            type="button"
            title={option.hint}
            aria-pressed={filter === option.value}
            className={filter === option.value ? "segmented-active" : ""}
            onClick={() => setFilter(option.value)}
          >
            {option.label}
            <span className="segmented-count">{counts[option.value]}</span>
          </button>
        ))}
      </div>
      <div className="arch-palette-scroll">
        {grouped.length === 0 && (
          <p className="arch-palette-empty">
            {search ? `No node matches “${search}”.` : "Nothing in this category."}
          </p>
        )}
        {grouped.map(([category, entries]) => (
          <section className="arch-palette-group" key={category}>
            <p className="arch-palette-label" style={{ color: categoryColorVar(category) }}>
              <span
                className="arch-palette-dot"
                style={{ background: categoryColorVar(category) }}
                aria-hidden
              />
              {category}
            </p>
            {entries.map((spec) => (
              <button
                key={spec.type}
                type="button"
                className={`arch-palette-item${spec.kind === "block" ? " arch-palette-item-block" : ""}`}
                title={spec.source ? `${spec.description}\n\nSource: ${spec.source}` : spec.description}
                disabled={disabled}
                draggable={!disabled}
                style={
                  spec.kind === "block"
                    ? { background: categoryTint(spec.category, 8) }
                    : undefined
                }
                onDragStart={(event) => {
                  event.dataTransfer.setData("application/x-arch-node", spec.type);
                  event.dataTransfer.effectAllowed = "move";
                }}
                onClick={() => onAdd(spec)}
              >
                <span className="arch-palette-item-name">
                  <CategoryIcon category={spec.category} size={12} />
                  {spec.name}
                  {spec.kind === "block" && (
                    <span className="arch-palette-badge" title="Expands into several layers">
                      <Layers size={10} aria-hidden />
                    </span>
                  )}
                </span>
                <span className="arch-palette-item-desc">{spec.description}</span>
              </button>
            ))}
          </section>
        ))}
      </div>
    </aside>
  );
}
