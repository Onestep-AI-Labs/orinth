"use client";

import { BarChart3, FileJson, type LucideIcon, Save } from "lucide-react";

export type DatasetDetailTab = "images" | "annotate" | "eda" | "config" | "records";

/**
 * Detail-view tab row. Image/text datasets show items + Annotate; `llm_finetune`
 * datasets replace both with a single Records tab (phase 10). EDA and Config are
 * shared across every task type.
 */
export function DatasetDetailTabs({
  itemIcon: ItemIcon,
  itemLabel,
  activeTab,
  setActiveTab,
  isLlm = false
}: {
  itemIcon: LucideIcon;
  itemLabel: string;
  activeTab: DatasetDetailTab;
  setActiveTab: (tab: DatasetDetailTab) => void;
  isLlm?: boolean;
}) {
  return (
    <div className="detail-tab-row">
      {isLlm ? (
        <button
          className={`detail-tab ${activeTab === "records" ? "detail-tab-active" : ""}`}
          onClick={() => setActiveTab("records")}
          type="button"
        >
          <FileJson size={16} /> Records
        </button>
      ) : (
        <>
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
        </>
      )}
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
