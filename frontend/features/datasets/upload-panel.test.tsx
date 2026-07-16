import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DatasetUploadPanel } from "@/features/datasets/upload-panel";
import type { DatasetSummary } from "@/types/api";

vi.mock("@/features/datasets/upload-drop-card", () => ({
  UploadDropCard: (props: {
    files: File[];
    onUpload: () => void;
    pending: boolean;
    errors: Array<Record<string, string>>;
  }) => (
    <div>
      <span data-testid="file-count">{props.files.length}</span>
      <span data-testid="pending">{String(props.pending)}</span>
      <span data-testid="error-count">{props.errors.length}</span>
      <button onClick={props.onUpload}>Upload</button>
    </div>
  )
}));

function fakeDataset(overrides: Partial<DatasetSummary> = {}): DatasetSummary {
  return {
    id: "ds-1",
    name: "Dataset 1",
    task_type: "classification",
    editable: true,
    labels: ["cat", "dog"],
    counts: { train: 0, val: 0, test: 0, unassigned: 0 },
    ...overrides
  } as DatasetSummary;
}

describe("DatasetUploadPanel", () => {
  it("renders nothing when the dataset is not editable", () => {
    const { container } = render(
      <DatasetUploadPanel
        dataset={fakeDataset({ editable: false })}
        split="train"
        files={[]}
        setFiles={vi.fn()}
        uploadClassId={0}
        setUploadClassId={vi.fn()}
        errors={[]}
        uploadMutation={{ mutate: vi.fn(), isPending: false } as never}
      />
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("submits a FormData upload with the mapped split and class_id for classification datasets", async () => {
    const user = userEvent.setup();
    const mutate = vi.fn();
    const file = new File(["data"], "a.png", { type: "image/png" });

    render(
      <DatasetUploadPanel
        dataset={fakeDataset({ task_type: "classification" })}
        split="all"
        files={[file]}
        setFiles={vi.fn()}
        uploadClassId={2}
        setUploadClassId={vi.fn()}
        errors={[]}
        uploadMutation={{ mutate, isPending: false } as never}
      />
    );

    await user.click(screen.getByRole("button", { name: "Upload" }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const call = mutate.mock.calls[0][0] as { datasetId: string; form: FormData };
    expect(call.datasetId).toBe("ds-1");
    expect(call.form.get("split")).toBe("unassigned");
    expect(call.form.get("class_id")).toBe("2");
    expect(call.form.getAll("files")).toEqual([file]);
  });

  it("does not append class_id for non-classification task types", async () => {
    const user = userEvent.setup();
    const mutate = vi.fn();
    const file = new File(["data"], "a.txt", { type: "text/plain" });

    render(
      <DatasetUploadPanel
        dataset={fakeDataset({ task_type: "detection" as never })}
        split="train"
        files={[file]}
        setFiles={vi.fn()}
        uploadClassId={2}
        setUploadClassId={vi.fn()}
        errors={[]}
        uploadMutation={{ mutate, isPending: false } as never}
      />
    );

    await user.click(screen.getByRole("button", { name: "Upload" }));

    const call = mutate.mock.calls[0][0] as { form: FormData };
    expect(call.form.get("split")).toBe("train");
    expect(call.form.get("class_id")).toBeNull();
  });

  it("does not submit when there are no files selected", async () => {
    const user = userEvent.setup();
    const mutate = vi.fn();

    render(
      <DatasetUploadPanel
        dataset={fakeDataset()}
        split="train"
        files={[]}
        setFiles={vi.fn()}
        uploadClassId={0}
        setUploadClassId={vi.fn()}
        errors={[]}
        uploadMutation={{ mutate, isPending: false } as never}
      />
    );

    await user.click(screen.getByRole("button", { name: "Upload" }));

    expect(mutate).not.toHaveBeenCalled();
  });
});
