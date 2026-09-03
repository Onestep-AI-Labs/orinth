import { fireEvent, render, screen } from "@testing-library/react";
import { ImageIcon } from "lucide-react";
import { describe, expect, it, vi } from "vitest";
import { DatasetDetailTabs } from "@/features/datasets/detail-tabs";
import type { DatasetReadiness } from "@/types/api";

function readiness(trainable: boolean): DatasetReadiness {
  return {
    state: trainable ? "ready" : "needs_prep",
    trainable,
    summary: trainable ? "Ready to train." : "The train split is empty.",
    next_action: trainable ? "none" : "run_prep",
    checks: [],
    busy: false
  } as DatasetReadiness;
}

function renderTabs(overrides: Partial<Parameters<typeof DatasetDetailTabs>[0]> = {}) {
  const setActiveTab = vi.fn();
  render(
    <DatasetDetailTabs
      itemIcon={ImageIcon}
      itemLabel="Images"
      activeTab="overview"
      setActiveTab={setActiveTab}
      {...overrides}
    />
  );
  return { setActiveTab };
}

describe("DatasetDetailTabs", () => {
  it("leads with Overview, then the data, then Prepare", () => {
    renderTabs();
    const labels = screen.getAllByRole("button").map((button) => button.textContent?.trim());
    expect(labels).toEqual(["Overview", "Images", "Prepare"]);
  });

  it("names the data tab after the item kind", () => {
    renderTabs({ itemLabel: "Texts" });
    expect(screen.getByText("Texts")).toBeInTheDocument();
  });

  it("calls it Records for llm_finetune datasets", () => {
    renderTabs({ isLlm: true });
    expect(screen.getByText("Records")).toBeInTheDocument();
    expect(screen.queryByText("Images")).not.toBeInTheDocument();
  });

  it("marks the active tab", () => {
    renderTabs({ activeTab: "prepare" });
    expect(screen.getByText("Prepare").closest("button")).toHaveClass("detail-tab-active");
  });

  it("selects a tab on click", () => {
    const { setActiveTab } = renderTabs();
    fireEvent.click(screen.getByText("Prepare"));
    expect(setActiveTab).toHaveBeenCalledWith("prepare");
  });

  it("shows a readiness dot only once the dataset can actually train", () => {
    const { unmount } = render(
      <DatasetDetailTabs
        itemIcon={ImageIcon}
        itemLabel="Images"
        activeTab="overview"
        setActiveTab={vi.fn()}
        readiness={readiness(true)}
      />
    );
    expect(screen.getByLabelText("Ready to train")).toBeInTheDocument();
    unmount();

    renderTabs({ readiness: readiness(false) });
    expect(screen.queryByLabelText("Ready to train")).not.toBeInTheDocument();
  });

  it("renders without readiness at all", () => {
    // Reference and sample datasets reach the studio before anything computes.
    renderTabs({ readiness: null });
    expect(screen.getByText("Overview")).toBeInTheDocument();
  });
});
