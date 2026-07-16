"use client";

import { BarChart3, type LucideIcon, Save } from "lucide-react";

export type DatasetDetailTab = "images" | "annotate" | "eda" | "config";

/** Detail-view tab row (images/texts, annotate, eda, config). */
export function DatasetDetailTabs({
  itemIcon: ItemIcon,
  itemLabel,
  activeTab,
  setActiveTab
}: {
  itemIcon: LucideIcon;
  itemLabel: string;
  activeTab: DatasetDetailTab;
  setActiveTab: (tab: DatasetDetailTab) => void;
}) {
  return (
    <div className="detail-tab-row">
      <button
        className={`detail-tab ${activeTab === "images" ? "detail-tab-active" : ""}`}
        onClick={() => setActiveTab("images")}
        type="button"
      >
        <ItemIcon size={16} /> {itemLabel}
      </button>
      <button
        className={`detail-tab ${activeTab === "annotate" ? "detail-tab-active" : ""}`}
        onClick={() => setActiveTab("annotate")}
        type="button"
      >
        <BarChart3 size={16} /> Annotate
      </button>
      <button
        className={`detail-tab ${activeTab === "eda" ? "detail-tab-active" : ""}`}
        onClick={() => setActiveTab("eda")}
        type="button"
      >
        <BarChart3 size={16} /> EDA
      </button>
      <button
        className={`detail-tab ${activeTab === "config" ? "detail-tab-active" : ""}`}
        onClick={() => setActiveTab("config")}
        type="button"
      >
        <Save size={16} /> Config
      </button>
    </div>
  );
}
