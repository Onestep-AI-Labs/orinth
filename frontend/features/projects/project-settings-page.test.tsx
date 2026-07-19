import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProjectSettingsPage } from "@/features/projects/project-settings-page";
import type { ProjectStats, ProjectSummary } from "@/types/api";

const projectStats = vi.fn();
const datasetCatalog = vi.fn();
const updateProject = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    projectStats: (...args: unknown[]) => projectStats(...args),
    datasetCatalog: (...args: unknown[]) => datasetCatalog(...args),
    updateProject: (...args: unknown[]) => updateProject(...args),
    deleteProject: vi.fn()
  }
}));

const projects: ProjectSummary[] = [];

vi.mock("@/components/app-shell", () => ({
  useProject: () => ({
    projectId: "retina-study",
    projects,
    projectsLoading: false,
    setProjectId: vi.fn(),
    refreshProjects: vi.fn().mockResolvedValue(undefined)
  })
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

function fakeProject(overrides: Partial<ProjectSummary> = {}): ProjectSummary {
  return {
    id: "retina-study",
    name: "Retina Study",
    description: "Fundus experiments",
    task_types: ["segmentation"],
    metadata: {},
    archived: false,
    created_at: "2026-01-01T00:00:00",
    updated_at: "2026-01-01T00:00:00",
    ...overrides
  } as ProjectSummary;
}

function fakeStats(overrides: Partial<ProjectStats> = {}): ProjectStats {
  return {
    project_id: "retina-study",
    datasets: 0,
    training_jobs: 0,
    evaluation_jobs: 0,
    inference_jobs: 0,
    inference_runs: 0,
    deletable: true,
    blockers: [],
    ...overrides
  } as ProjectStats;
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ProjectSettingsPage projectId="retina-study" />
    </QueryClientProvider>
  );
}

describe("ProjectSettingsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    projects.length = 0;
    projects.push(fakeProject());
    projectStats.mockResolvedValue(fakeStats());
    datasetCatalog.mockResolvedValue([]);
    updateProject.mockResolvedValue(fakeProject());
  });

  it("seeds the form from the project in context", () => {
    renderPage();
    expect(screen.getByDisplayValue("Retina Study")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Fundus experiments")).toBeInTheDocument();
  });

  it("disables save until something changes", async () => {
    renderPage();
    const save = screen.getByRole("button", { name: /save changes/i });
    expect(save).toBeDisabled();

    await userEvent.type(screen.getByDisplayValue("Retina Study"), " v2");
    expect(save).toBeEnabled();
  });

  it("disables save when the name is emptied", async () => {
    renderPage();
    await userEvent.clear(screen.getByDisplayValue("Retina Study"));
    expect(screen.getByRole("button", { name: /save changes/i })).toBeDisabled();
  });

  it("lists blockers and disables delete when the project owns content", async () => {
    projectStats.mockResolvedValue(
      fakeStats({ datasets: 2, training_jobs: 1, deletable: false, blockers: ["2 datasets", "1 training job"] })
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/2 datasets, 1 training job/)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /delete/i })).toBeDisabled();
  });

  it("shows an archived badge and offers restore for an archived project", async () => {
    projects[0] = fakeProject({ archived: true });
    renderPage();

    expect(screen.getByText("Archived")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /restore/i })).toBeInTheDocument();
  });

  it("opens on the domain the project already works in", () => {
    renderPage();
    // Vision-only project: its own task types are visible without switching tabs.
    expect(screen.getByRole("button", { name: /segmentation/i })).toHaveAttribute(
      "aria-pressed",
      "true"
    );
    expect(screen.queryByRole("button", { name: /summarization/i })).not.toBeInTheDocument();
  });

  it("reaches the other domain so a vision project can adopt an NLP task type", async () => {
    renderPage();
    const save = screen.getByRole("button", { name: /save changes/i });
    expect(save).toBeDisabled();

    // The concrete gap this phase closes: task types were frozen at creation.
    await userEvent.click(screen.getByRole("button", { name: /^NLP/ }));
    await userEvent.click(screen.getByRole("button", { name: /text classification/i }));

    expect(save).toBeEnabled();
    await userEvent.click(save);
    await waitFor(() => expect(updateProject).toHaveBeenCalled());
    const [, payload] = updateProject.mock.calls[0];
    expect(payload.task_types).toEqual(["segmentation", "text_classification"]);
    // Domain is derived on every save, so it cannot claim the domain just left.
    expect(payload.metadata.domain).toBe("mixed");
  });

  it("disables save when every task type is deselected", async () => {
    renderPage();
    await userEvent.click(screen.getByRole("button", { name: /segmentation/i }));
    expect(screen.getByRole("button", { name: /save changes/i })).toBeDisabled();
  });

  it("blocks archive and delete for the default workspace", async () => {
    projects[0] = fakeProject({ id: "default-research-project" });
    projectStats.mockResolvedValue(fakeStats({ project_id: "default-research-project", deletable: false }));
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ProjectSettingsPage projectId="default-research-project" />
      </QueryClientProvider>
    );

    expect(screen.getByRole("button", { name: /archive/i })).toBeDisabled();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /delete/i })).toBeDisabled();
    });
  });
});
