import { fireEvent, render, screen } from "@testing-library/react";
import { ImageIcon } from "lucide-react";
import { describe, expect, it, vi } from "vitest";
import { DatasetDataViewTabs, resolveDataView } from "@/features/datasets/data-view-tabs";

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

  it("calls the gallery Records for llm_finetune datasets and drops Annotate", () => {
    renderTabs({ isLlm: true });
    const labels = screen.getAllByRole("tab").map((tab) => tab.textContent?.trim());
    expect(labels).toEqual(["Table", "Records"]);
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

describe("resolveDataView", () => {
  it("keeps a view the dataset has", () => {
    expect(resolveDataView("annotate", false)).toBe("annotate");
    expect(resolveDataView("gallery", true)).toBe("gallery");
  });

  it("falls back to Table when the dataset has no Annotate segment", () => {
    // Carrying Annotate into an llm_finetune dataset would render an editor for
    // records that have nothing to annotate.
    expect(resolveDataView("annotate", true)).toBe("table");
  });
});
