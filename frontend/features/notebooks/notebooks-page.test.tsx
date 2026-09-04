import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { categoriesOf, pageCountOf, pageOf } from "./notebooks-page";
import { Pager } from "@/features/platform/ui";
import type { NotebookTemplate } from "@/types/api";

function template(id: string, category: string): NotebookTemplate {
  return { id, name: id, description: id, category, task_types: [] };
}

describe("categoriesOf", () => {
  it("keeps the order the backend sent, not alphabetical", () => {
    // The service orders by CATEGORY_ORDER — the order the work happens in.
    // Sorting here would reopen the picker on "Data".
    const categories = categoriesOf([
      template("blank", "Start"),
      template("eda", "Data"),
      template("train", "Training"),
      template("evaluate", "Testing")
    ]);
    expect(categories).toEqual(["Start", "Data", "Training", "Testing"]);
  });

  it("lists each category once however many templates it holds", () => {
    expect(
      categoriesOf([template("a", "Data"), template("b", "Data"), template("c", "Text")])
    ).toEqual(["Data", "Text"]);
  });
});

describe("paging", () => {
  const items = Array.from({ length: 22 }, (_, index) => index);

  it("slices the page it is asked for", () => {
    expect(pageOf(items, 0, 6)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(pageOf(items, 3, 6)).toEqual([18, 19, 20, 21]);
  });

  it("counts a partial last page, and never reports zero pages", () => {
    expect(pageCountOf(22, 6)).toBe(4);
    expect(pageCountOf(6, 6)).toBe(1);
    // An empty list still has one (empty) page — a pager over "0 / 0" is worse
    // than no pager, and the empty state renders instead.
    expect(pageCountOf(0, 6)).toBe(1);
  });
});

describe("Pager", () => {
  it("renders nothing when everything fits on one page", () => {
    const { container } = render(
      <Pager page={0} pageCount={1} onChange={vi.fn()} label="Template pages" />
    );
    expect(container.firstChild).toBeNull();
  });

  it("disables the end it is already at", () => {
    render(<Pager page={0} pageCount={3} onChange={vi.fn()} label="Template pages" unit="templates" />);
    expect(screen.getByLabelText("Previous page of templates")).toBeDisabled();
    expect(screen.getByLabelText("Next page of templates")).not.toBeDisabled();
    expect(screen.getByText("1 / 3")).toBeTruthy();
  });

  it("steps one page at a time", async () => {
    const onChange = vi.fn();
    render(<Pager page={1} pageCount={3} onChange={onChange} label="Template pages" unit="templates" />);
    await userEvent.click(screen.getByLabelText("Next page of templates"));
    expect(onChange).toHaveBeenCalledWith(2);
    await userEvent.click(screen.getByLabelText("Previous page of templates"));
    expect(onChange).toHaveBeenCalledWith(0);
  });
});
