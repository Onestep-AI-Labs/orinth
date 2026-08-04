import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { UploadDropCard } from "@/features/datasets/upload-drop-card";
import type { DatasetSummary } from "@/types/api";

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

function renderCard(props: Partial<Parameters<typeof UploadDropCard>[0]> = {}) {
  const setFiles = vi.fn();
  render(
    <UploadDropCard
      files={[]}
      setFiles={setFiles}
      dataset={fakeDataset()}
      uploadClassId={0}
      setUploadClassId={vi.fn()}
      onUpload={vi.fn()}
      pending={false}
      errors={[]}
      {...props}
    />
  );
  return { setFiles };
}

/** The hidden `webkitdirectory` input the folder fallback drives. */
function folderInput(): HTMLInputElement {
  const input = document
    .querySelectorAll<HTMLInputElement>('input[type="file"]')[1];
  return input;
}

afterEach(() => {
  // `showDirectoryPicker` is assigned per-test; make sure it never leaks.
  delete (window as { showDirectoryPicker?: unknown }).showDirectoryPicker;
  vi.restoreAllMocks();
});

describe("UploadDropCard folder selection", () => {
  it("renders a webkitdirectory input so WebKit and Firefox can pick folders", () => {
    renderCard();
    // The desktop app runs on WKWebView, which has no File System Access API;
    // without this input, "Select folder" dead-ends in an error there.
    expect(folderInput()).toHaveAttribute("webkitdirectory");
    expect(folderInput()).toHaveAttribute("multiple");
  });

  it("falls back to the directory input when showDirectoryPicker is absent", async () => {
    renderCard();
    const click = vi.spyOn(folderInput(), "click").mockImplementation(() => {});

    await userEvent.click(screen.getByRole("button", { name: /select folder/i }));

    // The old behavior stopped here with "not supported in this browser",
    // which is what every WebKit and Firefox user hit — the desktop app
    // included.
    expect(click).toHaveBeenCalledOnce();
    expect(
      screen.queryByText(/folder selection is not supported/i)
    ).not.toBeInTheDocument();
  });

  it("prefers showDirectoryPicker when the browser provides it", async () => {
    const picker = vi.fn().mockRejectedValue(
      Object.assign(new DOMException("cancelled", "AbortError"))
    );
    (window as { showDirectoryPicker?: unknown }).showDirectoryPicker = picker;
    renderCard();
    const click = vi.spyOn(folderInput(), "click").mockImplementation(() => {});

    await userEvent.click(screen.getByRole("button", { name: /select folder/i }));

    expect(picker).toHaveBeenCalledOnce();
    expect(click).not.toHaveBeenCalled();
  });

  it("keeps files whose type is empty, as directory picks often are", async () => {
    const { setFiles } = renderCard();
    // Files read out of a folder frequently arrive with an empty MIME type;
    // filtering on type alone silently discarded the whole selection.
    const typeless = new File(["x"], "scan_01.png", { type: "" });
    const jpeg = new File(["x"], "scan_02.JPG", { type: "" });
    const other = new File(["x"], "notes.pdf", { type: "application/pdf" });

    await userEvent.upload(folderInput(), [typeless, jpeg, other]);

    expect(setFiles).toHaveBeenCalledOnce();
    expect(setFiles.mock.calls[0][0].map((file: File) => file.name)).toEqual([
      "scan_01.png",
      "scan_02.JPG"
    ]);
  });

  it("explains when a chosen folder holds no supported files", async () => {
    renderCard();
    await userEvent.upload(folderInput(), [
      new File(["x"], "readme.md", { type: "text/markdown" })
    ]);

    expect(await screen.findByText(/no supported image files were found/i)).toBeInTheDocument();
  });
});
