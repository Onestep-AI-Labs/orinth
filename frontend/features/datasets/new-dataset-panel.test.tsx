import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { NewDatasetPanel } from "@/features/datasets/new-dataset-panel";

function renderPanel(canImportHub = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <NewDatasetPanel
        projectId="p-1"
        onCreated={vi.fn()}
        canImportHub={canImportHub}
        hubImportPanel={<div>hub panel</div>}
        createPanel={<div>create panel</div>}
      />
    </QueryClientProvider>
  );
}

describe("NewDatasetPanel", () => {
  it("opens on the file drop, which is the one source that needs no decisions", () => {
    renderPanel();
    expect(screen.getByText("Drop your data here")).toBeInTheDocument();
    expect(screen.queryByText("hub panel")).not.toBeInTheDocument();
    expect(screen.queryByText("create panel")).not.toBeInTheDocument();
  });

  it("switches to one source at a time", () => {
    renderPanel();

    fireEvent.click(screen.getByRole("tab", { name: /HuggingFace/ }));
    expect(screen.getByText("hub panel")).toBeInTheDocument();
    expect(screen.queryByText("Drop your data here")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Empty/ }));
    expect(screen.getByText("create panel")).toBeInTheDocument();
    expect(screen.queryByText("hub panel")).not.toBeInTheDocument();
  });

  it("hands documents off to the recipe builder rather than faking it here", () => {
    renderPanel();
    fireEvent.click(screen.getByRole("tab", { name: /Documents/ }));
    expect(screen.getByRole("link", { name: /Build from documents/ })).toHaveAttribute(
      "href",
      "/datasets/recipes"
    );
  });

  it("hides the two LLM-only sources in a project that cannot hold one", () => {
    renderPanel(false);
    expect(screen.queryByRole("tab", { name: /HuggingFace/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /Documents/ })).not.toBeInTheDocument();
    // Files and Empty always remain: every project can build a dataset by hand.
    expect(screen.getByRole("tab", { name: /Files/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Empty/ })).toBeInTheDocument();
  });
});
