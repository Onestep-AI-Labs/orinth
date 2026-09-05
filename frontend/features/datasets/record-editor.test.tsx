import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DatasetRecordEditor } from "@/features/datasets/record-editor";
import type { DatasetSummary } from "@/types/api";

function fakeDataset(overrides: Partial<DatasetSummary> = {}): DatasetSummary {
  return {
    id: "ds-1",
    name: "Records",
    task_type: "llm_finetune",
    format: "instruction_jsonl",
    editable: true,
    labels: [],
    counts: { train: 0, val: 0, test: 0, unassigned: 0 },
    ...overrides
  } as DatasetSummary;
}

function renderEditor(props: Partial<Parameters<typeof DatasetRecordEditor>[0]> = {}) {
  const onSave = vi.fn();
  render(
    <DatasetRecordEditor
      dataset={fakeDataset()}
      mode="edit"
      record={{ instruction: "Summarise", output: "A summary" }}
      onSave={onSave}
      {...props}
    />
  );
  return { onSave };
}

describe("DatasetRecordEditor", () => {
  it("hydrates the instruction fields from the record", () => {
    renderEditor();
    expect(screen.getByDisplayValue("Summarise")).toBeInTheDocument();
    expect(screen.getByDisplayValue("A summary")).toBeInTheDocument();
  });

  it("saves an edited field", async () => {
    const { onSave } = renderEditor();
    await userEvent.clear(screen.getByDisplayValue("A summary"));
    await userEvent.type(screen.getByPlaceholderText("Target response"), "Better");
    await userEvent.click(screen.getByText("Save record"));
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ instruction: "Summarise", output: "Better" }),
      "unassigned"
    );
  });

  it("carries forward fields the structured editor does not surface", async () => {
    // An imported row can hold columns this form has no input for. Dropping them
    // on save would silently rewrite the user's data.
    const { onSave } = renderEditor({
      record: { instruction: "Ask", output: "Answer", source: "hf://squad", row_id: 41 }
    });
    await userEvent.click(screen.getByText("Save record"));
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ source: "hf://squad", row_id: 41 }),
      "unassigned"
    );
  });

  it("names every field the record actually has", () => {
    renderEditor({ record: { instruction: "Ask", output: "Answer", source: "hf://squad" } });
    expect(screen.getByText("source")).toBeInTheDocument();
  });

  it("omits an empty optional input rather than writing a blank one", async () => {
    const { onSave } = renderEditor();
    await userEvent.click(screen.getByText("Save record"));
    expect(onSave.mock.calls[0][0]).not.toHaveProperty("input");
  });

  it("edits chat records as roled messages", () => {
    renderEditor({
      dataset: fakeDataset({ format: "chat_jsonl" }),
      record: { messages: [{ role: "user", content: "Hi" }, { role: "assistant", content: "Hello" }] }
    });
    expect(screen.getByDisplayValue("Hi")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Hello")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Target response")).not.toBeInTheDocument();
  });

  it("round-trips through the raw JSON editor", async () => {
    const { onSave } = renderEditor();
    await userEvent.click(screen.getByText("Raw JSON"));
    const raw = screen.getByPlaceholderText('{ "instruction": "...", "output": "..." }');
    expect(raw).toHaveValue(JSON.stringify({ instruction: "Summarise", output: "A summary" }, null, 2));
    await userEvent.clear(raw);
    await userEvent.type(raw, '{{"kept": 1}');
    await userEvent.click(screen.getByText("Save record"));
    expect(onSave).toHaveBeenCalledWith({ kept: 1 }, "unassigned");
  });

  it("offers a split only when creating, since an existing record already has one", () => {
    const { unmount } = render(
      <DatasetRecordEditor dataset={fakeDataset()} mode="create" record={null} onSave={vi.fn()} />
    );
    expect(screen.getByText("Split")).toBeInTheDocument();
    expect(screen.getByText("Add record")).toBeInTheDocument();
    unmount();

    renderEditor();
    expect(screen.queryByText("Split")).not.toBeInTheDocument();
  });

  it("shows Cancel only where there is something to cancel out of", () => {
    const { unmount } = render(
      <DatasetRecordEditor dataset={fakeDataset()} mode="edit" record={{}} onSave={vi.fn()} onCancel={vi.fn()} />
    );
    expect(screen.getByText("Cancel")).toBeInTheDocument();
    unmount();

    // Inline in Annotate the editor is the panel, not an overlay.
    renderEditor();
    expect(screen.queryByText("Cancel")).not.toBeInTheDocument();
  });
});
