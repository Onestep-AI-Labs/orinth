import { ImageIcon } from "lucide-react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DatasetDetailTabs } from "@/features/datasets/detail-tabs";

describe("DatasetDetailTabs", () => {
  it("renders all four tabs with the active tab marked", () => {
    render(
      <DatasetDetailTabs itemIcon={ImageIcon} itemLabel="Images" activeTab="annotate" setActiveTab={vi.fn()} />
    );

    expect(screen.getByRole("button", { name: /Images/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Annotate/ })).toHaveClass("detail-tab-active");
    expect(screen.getByRole("button", { name: /EDA/ })).not.toHaveClass("detail-tab-active");
    expect(screen.getByRole("button", { name: /Config/ })).not.toHaveClass("detail-tab-active");
  });

  it("uses the itemLabel prop for the first tab (e.g. 'Texts' for NLP datasets)", () => {
    render(<DatasetDetailTabs itemIcon={ImageIcon} itemLabel="Texts" activeTab="images" setActiveTab={vi.fn()} />);

    expect(screen.getByRole("button", { name: /Texts/ })).toBeInTheDocument();
  });

  it("calls setActiveTab with the clicked tab's key", async () => {
    const user = userEvent.setup();
    const setActiveTab = vi.fn();
    render(
      <DatasetDetailTabs itemIcon={ImageIcon} itemLabel="Images" activeTab="images" setActiveTab={setActiveTab} />
    );

    await user.click(screen.getByRole("button", { name: /Config/ }));
    expect(setActiveTab).toHaveBeenCalledWith("config");

    await user.click(screen.getByRole("button", { name: /EDA/ }));
    expect(setActiveTab).toHaveBeenCalledWith("eda");

    await user.click(screen.getByRole("button", { name: /Annotate/ }));
    expect(setActiveTab).toHaveBeenCalledWith("annotate");
  });
});
