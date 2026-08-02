"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import type { NodeSpec } from "@/types/api";
import { CategoryIcon, categoryColorVar } from "./node-visuals";

/**
 * The layer palette: categories from the backend catalog, filtered by a search
 * across name, type, and description.
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

  const grouped = useMemo(() => {
    const query = search.trim().toLowerCase();
    const matches = query
      ? specs.filter((spec) =>
          `${spec.name} ${spec.type} ${spec.description}`.toLowerCase().includes(query)
        )
      : specs;
    const order = new Map(categories.map((category, index) => [category, index]));
    const byCategory = new Map<string, NodeSpec[]>();
    for (const spec of matches) {
      byCategory.set(spec.category, [...(byCategory.get(spec.category) ?? []), spec]);
    }
    return [...byCategory.entries()].sort(
      ([a], [b]) => (order.get(a) ?? 99) - (order.get(b) ?? 99) || a.localeCompare(b)
    );
  }, [specs, categories, search]);

  return (
    <aside className="arch-palette" aria-label="Layer palette">
      <div className="arch-palette-search">
        <Search size={15} aria-hidden />
        <input
          type="search"
          value={search}
          placeholder="Find a layer"
          aria-label="Find a layer"
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>
      <div className="arch-palette-scroll">
        {grouped.length === 0 && <p className="arch-palette-empty">No layer matches “{search}”.</p>}
        {grouped.map(([category, entries]) => (
          <section className="arch-palette-group" key={category}>
            <p className="arch-palette-label">
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
                className="arch-palette-item"
                title={spec.description}
                disabled={disabled}
                draggable={!disabled}
                onDragStart={(event) => {
                  event.dataTransfer.setData("application/x-arch-node", spec.type);
                  event.dataTransfer.effectAllowed = "move";
                }}
                onClick={() => onAdd(spec)}
              >
                <span className="arch-palette-item-name">
                  <CategoryIcon category={spec.category} size={12} />
                  {spec.name}
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
