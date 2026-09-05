"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, Suspense, useContext, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  Boxes,
  ChevronsLeft,
  ChevronsRight,
  Database,
  NotebookPen,
  FlaskConical,
  Folder,
  Rocket,
  Settings,
  Settings2
} from "lucide-react";
import { useIsMutating, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { BRAND_NAME, BRAND_PARENT, BRAND_TAGLINE, LogoMark } from "@/components/brand";
import { IconButton, Select } from "@/features/platform/ui";
import { JobProgress, useForegroundJob } from "@/features/platform/foreground-job";
import { Toaster } from "@/features/platform/toast";
import { PlatformTour } from "@/features/platform/tour";
import type { ProjectSummary } from "@/types/api";

const DEFAULT_PROJECT_ID = "default-research-project";

/**
 * `/projects/{id}/settings` is project-scoped and shows the project sidebar;
 * `/projects` (the list) and `/projects/new` are not and must not match.
 */
const PROJECT_SETTINGS_PATH = /^\/projects\/[^/]+\/settings/;

type ProjectContextValue = {
  projectId: string;
  project: ProjectSummary | null;
  projects: ProjectSummary[];
  projectsLoading: boolean;
  setProjectId: (value: string) => void;
  refreshProjects: () => Promise<unknown>;
};

const ProjectContext = createContext<ProjectContextValue | null>(null);

export function useProject() {
  const context = useContext(ProjectContext);
  if (!context) throw new Error("useProject must be used inside AppShell");
  return context;
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [projectId, setProjectId] = useState(DEFAULT_PROJECT_ID);
  const [projectSidebarOpen, setProjectSidebarOpen] = useState(true);
  const activeMutations = useIsMutating();
  // A model load (serving start) is a long blocking mutation; the global
  // overlay stays full-screen but shows a progress bar instead of the circular
  // spinner while it runs.
  const loadingModel = useIsMutating({ mutationKey: ["serving-start"] }) > 0;
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const projects = useMemo(() => projectsQuery.data ?? [], [projectsQuery.data]);
  const project = useMemo(
    () => projects.find((item) => item.id === projectId) ?? null,
    [projects, projectId]
  );
  // Every route that reads or writes something belonging to one project. A
  // surface missing from this list still works, but loses the project sidebar
  // and the switcher with it — which is how `/notebooks` ended up as the one
  // project page you could not tell the project of.
  const isProjectArea =
    ["/datasets", "/notebooks", "/models", "/inference", "/testing", "/training"].some((prefix) =>
      pathname?.startsWith(prefix)
    ) || PROJECT_SETTINGS_PATH.test(pathname ?? "");

  useEffect(() => {
    const saved = window.localStorage.getItem("image-platform-project");
    if (saved) setProjectId(saved);
    const sidebar = window.localStorage.getItem("image-platform-project-sidebar");
    if (sidebar) setProjectSidebarOpen(sidebar !== "closed");
  }, []);

  useEffect(() => {
    if (project?.id) {
      setProjectId(project.id);
      window.localStorage.setItem("image-platform-project", project.id);
    }
  }, [project?.id]);

  useEffect(() => {
    window.localStorage.setItem(
      "image-platform-project-sidebar",
      projectSidebarOpen ? "open" : "closed"
    );
  }, [projectSidebarOpen]);

  const value = {
    projectId,
    project,
    projects,
    projectsLoading: projectsQuery.isLoading,
    setProjectId,
    refreshProjects: () => {
      return projectsQuery.refetch();
    }
  };

  return (
    <ProjectContext.Provider value={value}>
      <main
        className={`app-shell bg-canvas text-ink ${isProjectArea ? "app-shell-project" : ""} ${
          isProjectArea && !projectSidebarOpen ? "app-shell-project-collapsed" : ""
        }`}
      >
        {isProjectArea ? (
          <>
            <GlobalSidebar pathname={pathname || ""} compact />
            <ProjectSidebar
              pathname={pathname || ""}
              projectId={projectId}
              project={project}
              projects={projects}
              open={projectSidebarOpen}
              setOpen={setProjectSidebarOpen}
              setProjectId={setProjectId}
            />
          </>
        ) : (
          <GlobalSidebar pathname={pathname || ""} />
        )}
        <section className="content-shell">{children}</section>
      </main>
      {activeMutations > 0 && <GlobalLoadingOverlay loadingModel={loadingModel} />}
      <Toaster />
      {/* PlatformTour reads useSearchParams (to tell the dataset catalog from an
          open Dataset Studio); a Suspense boundary keeps that CSR-only read from
          forcing the statically-generated docs pages into client rendering. */}
      <Suspense fallback={null}>
        <PlatformTour />
      </Suspense>
    </ProjectContext.Provider>
  );
}

/**
 * The blocking overlay, and what it is allowed to say.
 *
 * "Working" is the fallback, not the design. A mutation that knows how far
 * through it is publishes a `ForegroundJob` and the overlay renders that
 * instead — the same bar, counts, and stage ladder the dataset panel shows.
 * Before this, the overlay covered those readouts: a forty-thousand-row prep
 * run drew a determinate progress bar *behind* a modal that said one word.
 */
function GlobalLoadingOverlay({ loadingModel }: { loadingModel: boolean }) {
  const job = useForegroundJob();
  return (
    // The job readout is its own live region, so the overlay stops being one
    // when it is showing it — two nested `role="status"` elements announce the
    // same change twice.
    <div
      className="global-loading-overlay"
      role={job ? undefined : "status"}
      aria-live={job ? undefined : "polite"}
    >
      <div className={`global-loading-panel ${job ? "global-loading-panel-job" : ""}`}>
        {job ? (
          <JobProgress job={job} />
        ) : loadingModel ? (
          <>
            <div className="global-loading-bar">
              <span />
            </div>
            <strong>Loading model…</strong>
          </>
        ) : (
          <>
            <span className="global-loading-spinner" />
            <strong>Working</strong>
          </>
        )}
      </div>
    </div>
  );
}

function GlobalSidebar({ pathname, compact = false }: { pathname: string; compact?: boolean }) {
  return (
    <aside className={`sidebar sidebar-global ${compact ? "sidebar-global-compact" : ""}`}>
      <Link className="brand-block brand-link" href="/projects" title={`${BRAND_NAME} workspace`} data-tour="brand">
        <span className="brand-mark" aria-hidden="true">
          <LogoMark size={26} />
        </span>
        <div className="brand-copy">
          <h1>{BRAND_NAME}</h1>
          <span>{BRAND_TAGLINE}</span>
        </div>
      </Link>
      <nav className="side-nav">
        {/* Project settings is a project-scoped destination owned by the project
            sidebar, so it must not light up the global Projects entry as well. */}
        <SideLink
          href="/projects"
          active={pathname.startsWith("/projects") && !PROJECT_SETTINGS_PATH.test(pathname)}
          icon={<Folder size={17} />}
          dataTour="nav-projects"
        >
          Projects
        </SideLink>
        <SideLink href="/settings" active={pathname.startsWith("/settings")} icon={<Settings size={17} />} dataTour="nav-settings">
          Settings
        </SideLink>
      </nav>
      <div className="sidebar-spacer" />
      <div className="sidebar-note">By {BRAND_PARENT}</div>
    </aside>
  );
}

function ProjectSidebar({
  pathname,
  projectId,
  project,
  projects,
  open,
  setOpen,
  setProjectId
}: {
  pathname: string;
  projectId: string;
  project: ProjectSummary | null;
  projects: ProjectSummary[];
  open: boolean;
  setOpen: (value: boolean) => void;
  setProjectId: (value: string) => void;
}) {
  const router = useRouter();

  // The settings route is per-project. Without moving the URL, switching projects
  // renames the sidebar while the form underneath keeps editing the old project.
  function switchProject(nextId: string) {
    setProjectId(nextId);
    if (PROJECT_SETTINGS_PATH.test(pathname)) router.replace(`/projects/${nextId}/settings`);
  }

  if (!open) {
    return (
      <aside className="sidebar sidebar-project sidebar-project-collapsed">
        <IconButton aria-label="Open project sidebar" onClick={() => setOpen(true)}>
          <ChevronsRight size={17} />
        </IconButton>
        <nav className="side-nav side-nav-icons">
          <SideLink href="/datasets" active={pathname.startsWith("/datasets")} icon={<Database size={17} />} iconOnly>
            Datasets
          </SideLink>
          <SideLink href="/notebooks" active={pathname.startsWith("/notebooks")} icon={<NotebookPen size={17} />} iconOnly>
            Notebooks
          </SideLink>
          <SideLink href="/models" active={pathname.startsWith("/models")} icon={<Boxes size={17} />} iconOnly>
            Models
          </SideLink>
          <SideLink href="/training" active={pathname.startsWith("/training")} icon={<Activity size={17} />} iconOnly>
            Training
          </SideLink>
          <SideLink href="/testing" active={pathname.startsWith("/testing")} icon={<FlaskConical size={17} />} iconOnly>
            Testing
          </SideLink>
          <SideLink href="/inference" active={pathname.startsWith("/inference")} icon={<Rocket size={17} />} iconOnly>
            Inference
          </SideLink>
        </nav>
        <div className="sidebar-spacer" />
        <div className="project-sidebar-footer">
          <SideLink
            href={`/projects/${projectId}/settings`}
            active={PROJECT_SETTINGS_PATH.test(pathname)}
            icon={<Settings2 size={17} />}
            title="Project settings"
            iconOnly
          >
            Settings
          </SideLink>
        </div>
      </aside>
    );
  }

  return (
    <aside className="sidebar sidebar-project">
      <div className="project-sidebar-top">
        <Link className="project-back-link" href="/projects">
          <ArrowLeft size={16} />
          <span>Projects</span>
        </Link>
        <IconButton aria-label="Close project sidebar" onClick={() => setOpen(false)}>
          <ChevronsLeft size={17} />
        </IconButton>
      </div>
      <div className="project-sidebar-title">
        <strong title={project?.name ?? "Project"}>{project?.name ?? "Project"}</strong>
        <span>Project workspace</span>
      </div>
      <label className="project-switcher">
        <span>Switch project</span>
        <Select value={project?.id ?? projectId} onChange={(event) => switchProject(event.target.value)}>
          {!projects.some((item) => item.id === projectId) && (
            <option value={projectId}>Project</option>
          )}
          {projects
            // Archived projects drop out of the switcher, except the active one —
            // otherwise selecting it would show a workspace the control cannot display.
            .filter((item) => !item.archived || item.id === projectId)
            .map((item) => (
              <option key={item.id} value={item.id}>
                {item.archived ? `${item.name} (archived)` : item.name}
              </option>
            ))}
        </Select>
      </label>
      <nav className="side-nav">
        <SideLink href="/datasets" active={pathname.startsWith("/datasets")} icon={<Database size={17} />} dataTour="nav-datasets">
          Datasets
        </SideLink>
        <SideLink href="/notebooks" active={pathname.startsWith("/notebooks")} icon={<NotebookPen size={17} />} dataTour="nav-notebooks">
          Notebooks
        </SideLink>
        <SideLink href="/models" active={pathname.startsWith("/models")} icon={<Boxes size={17} />} dataTour="nav-models">
          Models
        </SideLink>
        <SideLink href="/training" active={pathname.startsWith("/training")} icon={<Activity size={17} />} dataTour="nav-training">
          Training
        </SideLink>
        <SideLink href="/testing" active={pathname.startsWith("/testing")} icon={<FlaskConical size={17} />} dataTour="nav-testing">
          Testing
        </SideLink>
        <SideLink href="/inference" active={pathname.startsWith("/inference")} icon={<Rocket size={17} />} dataTour="nav-inference">
          Inference
        </SideLink>
      </nav>
      <div className="sidebar-spacer" />
      <div className="project-sidebar-footer">
        <SideLink
          href={`/projects/${projectId}/settings`}
          active={PROJECT_SETTINGS_PATH.test(pathname)}
          icon={<Settings2 size={17} />}
          title="Project settings"
        >
          Settings
        </SideLink>
      </div>
    </aside>
  );
}

function SideLink({
  href,
  active,
  icon,
  iconOnly = false,
  title,
  dataTour,
  children
}: {
  href: string;
  active: boolean;
  icon: React.ReactNode;
  iconOnly?: boolean;
  /** Tooltip override. Defaults to the label, which is all the collapsed rail shows. */
  title?: string;
  /** Anchor id consumed by the guided tour (see features/platform/tour). */
  dataTour?: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      className={`nav-link ${active ? "nav-link-active" : ""} ${iconOnly ? "nav-link-icon-only" : ""}`}
      href={href}
      title={title ?? String(children)}
      data-tour={dataTour}
    >
      {icon}
      {!iconOnly && <span>{children}</span>}
    </Link>
  );
}
