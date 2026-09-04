import { describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MarkdownCell, MarkdownEditor } from "./markdown-cell";

/**
 * The formatting toolbar exists because markdown is a language people
 * half-know. These pin the two things that makes true: it acts on the
 * *selection*, and the preview shows the result without leaving the cell.
 */

function Harness({ initial = "" }: { initial?: string }) {
  const [value, setValue] = useState(initial);
  return <MarkdownEditor value={value} onChange={setValue} onDone={() => undefined} />;
}

async function select(text: string, from: number, to: number) {
  const field = screen.getByLabelText("Text cell source") as HTMLTextAreaElement;
  field.focus();
  field.setSelectionRange(from, to);
  expect(field.value.slice(from, to)).toBe(text);
  return field;
}

describe("MarkdownEditor", () => {
  it("wraps the selection rather than appending at the end", async () => {
    render(<Harness initial="heading text" />);
    const field = await select("heading", 0, 7);

    await userEvent.click(screen.getByLabelText("Bold"));

    expect(field.value).toBe("**heading** text");
  });

  it("prefixes every line the selection touches", async () => {
    render(<Harness initial={"one\ntwo\nthree"} />);
    await select("one\ntwo", 0, 7);

    await userEvent.click(screen.getByLabelText("Bulleted list"));

    const field = screen.getByLabelText("Text cell source") as HTMLTextAreaElement;
    expect(field.value).toBe("- one\n- two\n- three");
  });

  it("toggles a prefix off when it is already there", async () => {
    render(<Harness initial="> quoted" />);
    await select("quoted", 2, 8);

    await userEvent.click(screen.getByLabelText("Quote"));

    expect((screen.getByLabelText("Text cell source") as HTMLTextAreaElement).value).toBe("quoted");
  });

  it("renders the preview beside the source as it is typed", async () => {
    render(<Harness />);
    // Empty says what the pane is for rather than sitting blank.
    expect(screen.getByText(/render appears here/i)).toBeTruthy();

    await userEvent.type(screen.getByLabelText("Text cell source"), "## Section");

    expect(screen.getByRole("heading", { name: "Section" })).toBeTruthy();
  });

  it("Shift-Enter renders the cell", async () => {
    const onDone = vi.fn();
    render(<MarkdownEditor value="text" onChange={vi.fn()} onDone={onDone} />);

    screen.getByLabelText("Text cell source").focus();
    await userEvent.keyboard("{Shift>}{Enter}{/Shift}");

    expect(onDone).toHaveBeenCalled();
  });
});

describe("MarkdownCell", () => {
  it("renders markdown rather than showing its source", () => {
    render(<MarkdownCell source="## Findings" editing={false} onEdit={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "Findings" })).toBeTruthy();
  });

  it("says what an empty cell is for", () => {
    render(<MarkdownCell source="" editing={false} onEdit={vi.fn()} />);
    expect(screen.getByText(/double-click to write/i)).toBeTruthy();
  });

  it("renders nothing while the cell is being edited", () => {
    const { container } = render(<MarkdownCell source="x" editing onEdit={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });
});
