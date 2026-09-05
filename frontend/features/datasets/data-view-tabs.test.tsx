import { fireEvent, render, screen } from "@testing-library/react";
import { ImageIcon } from "lucide-react";
import { describe, expect, it, vi } from "vitest";
import { DatasetDataViewTabs } from "@/features/datasets/data-view-tabs";

function renderTabs(overrides: Partial<Parameters<typeof DatasetDataViewTabs>[0]> = {}) {
  const setView = vi.fn();
  render(
    <DatasetDataViewTabs
      itemIcon={ImageIcon}
      isLlm={false}
      activeView="table"
      setView={setView}
      {...overrides}
    />
  );
  return { setView };
}

describe("DatasetDataViewTabs", () => {
  it("offers Table, Gallery and Annotate as peers", () => {
    renderTabs();
    const labels = screen.getAllByRole("tab").map((tab) => tab.textContent?.trim());
    expect(labels).toEqual(["Table", "Gallery", "Annotate"]);
  });

  it("calls the gallery Records for llm_finetune datasets, keeping all three", () => {
    // Records are the one modality made entirely of text, so Annotate is the
    // view they need most; what changes is the editor, not the segment row.
    renderTabs({ isLlm: true });
    const labels = screen.getAllByRole("tab").map((tab) => tab.textContent?.trim());
    expect(labels).toEqual(["Table", "Records", "Annotate"]);
  });

  it("marks the active view", () => {
    renderTabs({ activeView: "annotate" });
    expect(screen.getByText("Annotate").closest("button")).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Table").closest("button")).toHaveAttribute("aria-selected", "false");
  });

  it("selects a view on click", () => {
    const { setView } = renderTabs();
    fireEvent.click(screen.getByText("Annotate"));
    expect(setView).toHaveBeenCalledWith("annotate");
  });
});
