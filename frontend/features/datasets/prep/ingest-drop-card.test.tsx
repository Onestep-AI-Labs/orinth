import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { IngestDropCard } from "@/features/datasets/prep/ingest-drop-card";

function renderCard() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <IngestDropCard projectId="p-1" onCreated={vi.fn()} />
    </QueryClientProvider>
  );
}

function file(name: string, body = "hello") {
  return new File([body], name, { type: "text/plain" });
}

/**
 * The card used to report a pick as a bare count. Picking the wrong folder
 * produced exactly the same sentence as picking the right one, and the mistake
 * surfaced only after an upload and a prep run.
 */
describe("IngestDropCard", () => {
  it("names the files that were chosen, not just how many", () => {
    const { container } = renderCard();
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(input, { target: { files: [file("reviews.csv"), file("labels.json")] } });

    expect(screen.getByText("reviews.csv")).toBeInTheDocument();
    expect(screen.getByText("labels.json")).toBeInTheDocument();
  });

  it("names the folder and says its structure is kept", () => {
    const { container } = renderCard();
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const nested = file("cat-01.txt");
    Object.defineProperty(nested, "webkitRelativePath", { value: "pets/cat/cat-01.txt" });

    fireEvent.change(input, { target: { files: [nested] } });

    expect(screen.getByText("pets")).toBeInTheDocument();
    // The subfolder is the label for a classification upload, so the listed
    // path has to keep it rather than showing the bare file name.
    expect(screen.getByText("pets/cat/cat-01.txt")).toBeInTheDocument();
    expect(screen.getByText(/folder structure preserved/)).toBeInTheDocument();
  });

  it("collapses a large pick to a sample plus a count", () => {
    const { container } = renderCard();
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const files = Array.from({ length: 20 }, (_entry, index) => file(`row-${index}.txt`));

    fireEvent.change(input, { target: { files } });

    expect(screen.getByText("row-0.txt")).toBeInTheDocument();
    expect(screen.getByText("and 14 more")).toBeInTheDocument();
  });

  it("says nothing was usable rather than silently accepting an empty pick", () => {
    const { container } = renderCard();
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;

    fireEvent.change(input, { target: { files: [file("model.bin")] } });

    expect(screen.getByText(/None of those files are a format Orinth can read/)).toBeInTheDocument();
  });
});
