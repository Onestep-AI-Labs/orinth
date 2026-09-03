"use client";

import { FileJson, type LucideIcon, Save, Sparkles } from "lucide-react";
import type { DatasetReadiness } from "@/types/api";

export type DatasetDetailTab = "overview" | "data" | "prepare";

/**
 * Detail-view tab row.
 *
 * Phase 21 replaced four peer tabs (Images / Annotate / EDA / Config) with three
 * ordered ones. The old set presented a pipeline as unordered siblings, and put
 * the action that actually makes a dataset trainable inside a collapsed
 * accordion on the fourth of them — the least discoverable control in the app.
 *
 * Overview leads because it answers the question the studio previously could
 * not: is this data usable yet, and what did Orinth do to it. Each tab carries a
 * completion dot driven by readiness, so the row shows progress rather than
 * just location.
 */

type Tab = { key: DatasetDetailTab; label: string; icon: LucideIcon; tour: string };

export function DatasetDetailTabs({
  itemIcon,
  itemLabel,
  activeTab,
  setActiveTab,
  isLlm = false,
  readiness
}: {
  itemIcon: LucideIcon;
  itemLabel: string;
  activeTab: DatasetDetailTab;
  setActiveTab: (tab: DatasetDetailTab) => void;
  isLlm?: boolean;
  readiness?: DatasetReadiness | null;
}) {
  const tabs: Tab[] = [
    { key: "overview", label: "Overview", icon: Sparkles, tour: "dataset-studio-overview" },
    {
      key: "data",
      label: isLlm ? "Records" : itemLabel,
      icon: isLlm ? FileJson : itemIcon,
      tour: "dataset-studio-items"
    },
    { key: "prepare", label: "Prepare", icon: Save, tour: "dataset-studio-config" }
  ];

  // The dot marks the dataset as trainable, not the tab as visited: a tab is not
  // an achievement, and "you have looked at this" is not information.
  const ready = readiness?.trainable ?? false;

  return (
    <div className="detail-tab-row">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        return (
          <button
            key={tab.key}
            className={`detail-tab ${activeTab === tab.key ? "detail-tab-active" : ""}`}
            onClick={() => setActiveTab(tab.key)}
            type="button"
            data-tour={tab.tour}
          >
            <Icon size={16} /> {tab.label}
            {tab.key === "overview" && ready && (
              <span className="detail-tab-dot" aria-label="Ready to train" />
            )}
          </button>
        );
      })}
    </div>
  );
}
