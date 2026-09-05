import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { HubBrowser } from "@/features/datasets/hub/hub-browser";

const facets = {
  modalities: [
    { value: "text", label: "Text", hint: "Sentences, documents, or chat turns.", importable: true },
    { value: "image", label: "Image", hint: "Pictures. Browse only here.", importable: false }
  ],
  formats: [
    { value: "parquet", label: "Parquet", hint: "Columnar binary format.", importable: true }
  ],
  sizes: [{ value: "1K<n<10K", label: "1K – 10K rows", hint: "A comfortable set.", importable: true }],
  tasks: [
    {
      value: "question-answering",
      label: "Question answering",
      hint: "Question, context, and answer columns.",
      importable: true
    }
  ],
  sorts: [
    { value: "trending", label: "Trending", hint: "What the Hub is featuring.", importable: true },
    { value: "downloads", label: "Most downloaded", hint: "All-time downloads.", importable: true }
  ]
};

function result(overrides: Record<string, unknown> = {}) {
  return {
    hub_id: "rajpurkar/squad",
    author: "rajpurkar",
    downloads: 226_000,
    likes: 577,
    gated: false,
    tags: [],
    updated_at: null,
    pretty_name: "SQuAD",
    modalities: ["text"],
    formats: ["parquet"],
    task_categories: ["question-answering"],
    languages: ["en"],
    license: "cc-by-sa-4.0",
    size_category: "10K<n<100K",
    has_viewer: true,
    importable: true,
    trending_score: 10,
    summary: "Stanford Question Answering Dataset.",
    ...overrides
  };
}

function renderBrowser() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <HubBrowser
        projectId="p-1"
        openDataset={vi.fn()}
        catalogQuery={{ data: [] } as never}
        onImported={vi.fn()}
      />
    </QueryClientProvider>
  );
}

describe("HubBrowser", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, "datasetHubFacets").mockResolvedValue(facets as never);
    vi.spyOn(api, "searchDatasetHub").mockResolvedValue({ results: [result()], error: null } as never);
  });

  it("renders the filter vocabulary with each term's explanation as its tooltip", async () => {
    renderBrowser();
    // The whole point of serving the vocabulary: a chip that only names a term
    // is a quiz, so the sentence has to reach the DOM.
    const chip = await screen.findByRole("button", { name: "Parquet" });
    expect(chip).toHaveAttribute("title", "Columnar binary format.");
  });

  it("warns on a filter Orinth can browse but not import", async () => {
    renderBrowser();
    const chip = await screen.findByRole("button", { name: "Image" });
    expect(chip.getAttribute("title")).toContain("Orinth cannot import this");
  });

  it("imports from the row, without a trip through the detail screen", async () => {
    // The mapped import needed a column mapping and therefore a detail screen.
    // The as-is import needs neither, so the button belongs on the card.
    const ingest = vi
      .spyOn(api, "ingestDatasetHub")
      .mockResolvedValue({ id: "ds-1" } as never);
    vi.spyOn(api, "datasetPrepStatus").mockResolvedValue({ state: "ready" } as never);

    renderBrowser();
    await screen.findByText("rajpurkar/squad");

    fireEvent.change(screen.getByLabelText("Rows to import from rajpurkar/squad"), {
      target: { value: "25000" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Import" }));

    await waitFor(() =>
      expect(ingest).toHaveBeenCalledWith(
        expect.objectContaining({ hub_id: "rajpurkar/squad", max_rows: 25000, project_id: "p-1" })
      )
    );
  });

  it("refuses to import a dataset Orinth has no task for", async () => {
    vi.spyOn(api, "searchDatasetHub").mockResolvedValue({
      results: [result({ importable: false, modalities: ["audio"] })],
      error: null
    } as never);
    renderBrowser();
    await screen.findByText("rajpurkar/squad");
    expect(screen.getByRole("button", { name: "Import" })).toBeDisabled();
  });

  it("sends a toggled facet to the Hub rather than filtering the page locally", async () => {
    renderBrowser();
    fireEvent.click(await screen.findByRole("button", { name: "Text" }));
    await waitFor(() =>
      expect(api.searchDatasetHub).toHaveBeenCalledWith(
        expect.objectContaining({ modality: ["text"] })
      )
    );
  });

  it("does not search on every keystroke", async () => {
    renderBrowser();
    await screen.findByRole("button", { name: "Parquet" });
    const calls = vi.mocked(api.searchDatasetHub).mock.calls.length;

    const input = screen.getByLabelText("Search HuggingFace datasets");
    fireEvent.change(input, { target: { value: "squ" } });
    fireEvent.change(input, { target: { value: "squad" } });

    // Typing alone must not reach the Hub — that is one request per character
    // and a fast route to a 429.
    expect(vi.mocked(api.searchDatasetHub).mock.calls.length).toBe(calls);

    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() =>
      expect(api.searchDatasetHub).toHaveBeenCalledWith(expect.objectContaining({ query: "squad" }))
    );
  });

  it("shows the metadata needed to tell two similar datasets apart", async () => {
    renderBrowser();
    expect(await screen.findByText("rajpurkar/squad")).toBeInTheDocument();
    expect(screen.getByText("Stanford Question Answering Dataset.")).toBeInTheDocument();
    expect(screen.getByText("Viewer")).toBeInTheDocument();
    // 226,000 downloads reads as "226.0k" — a card is scanned, not audited.
    expect(screen.getByText("226.0k")).toBeInTheDocument();
  });

  it("flags a dataset that cannot be previewed instead of letting the user find out by clicking", async () => {
    vi.spyOn(api, "searchDatasetHub").mockResolvedValue({
      results: [result({ has_viewer: false, importable: false, modalities: ["video"] })],
      error: null
    } as never);
    renderBrowser();
    await screen.findByText("rajpurkar/squad");
    expect(screen.queryByText("Viewer")).not.toBeInTheDocument();
    expect(screen.getByText("Browse only")).toBeInTheDocument();
  });

  it("surfaces a Hub failure in the panel rather than swallowing it", async () => {
    vi.spyOn(api, "searchDatasetHub").mockResolvedValue({
      results: [],
      error: "rate limited"
    } as never);
    renderBrowser();
    expect(await screen.findByText(/Hub search failed: rate limited/)).toBeInTheDocument();
  });
});
