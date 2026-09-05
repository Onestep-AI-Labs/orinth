"use client";

import { type LucideIcon, Table2, Tags } from "lucide-react";

export type DatasetDataView = "table" | "gallery" | "annotate";

/**
 * The Data tab's view switcher.
 *
 * Phase 21 collapsed the old Annotate tab into the gallery, stacking the two
 * panels on one scroll. That put a second copy of the same item grid under the
 * first and pushed the editor — the whole point of annotating — below the fold
 * of a wall of thumbnails. Labelling is a third way of working on the same
 * dataset, not a footnote to the second, so it gets a peer segment: Table for
 * "what is in here", Gallery for the items themselves, Annotate for labelling
 * one at a time.
 *
 * Table stays the default: it is the only view that works for every modality.
 */

type View = { key: DatasetDataView; label: string; icon: LucideIcon };

function viewsFor(isLlm: boolean, itemIcon: LucideIcon): View[] {
  return [
    { key: "table", label: "Table", icon: Table2 },
    { key: "gallery", label: isLlm ? "Records" : "Gallery", icon: itemIcon },
    // Every modality gets Annotate, `llm_finetune` included: what changes is
    // what the editor on the right is. See annotate-tab.tsx.
    { key: "annotate", label: "Annotate", icon: Tags }
  ];
}

export function DatasetDataViewTabs({
  itemIcon,
  isLlm,
  activeView,
  setView
}: {
  itemIcon: LucideIcon;
  isLlm: boolean;
  activeView: DatasetDataView;
  setView: (view: DatasetDataView) => void;
}) {
  return (
    <div className="segmented-control" role="tablist" aria-label="Data view">
      {viewsFor(isLlm, itemIcon).map((view) => {
        const Icon = view.icon;
        return (
          <button
            key={view.key}
            type="button"
            role="tab"
            aria-selected={activeView === view.key}
            className={activeView === view.key ? "segmented-active" : ""}
            onClick={() => setView(view.key)}
          >
            <Icon size={15} /> {view.label}
          </button>
        );
      })}
    </div>
  );
}
