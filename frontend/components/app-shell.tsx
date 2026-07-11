"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Boxes,
  ChevronsLeft,
  ChevronsRight,
  Database,
  FlaskConical,
  ImageIcon,
  ScanEye,
  Settings
} from "lucide-react";
import { useIsMutating, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ProjectSummary } from "@/types/api";

const DEFAULT_PROJECT_ID = "default-research-project";

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
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.projects });
  const projects = useMemo(() => projectsQuery.data ?? [], [projectsQuery.data]);
  const project = useMemo(
    () => projects.find((item) => item.id === projectId) ?? null,
    [projects, projectId]
  );
  const isProjectArea = ["/datasets", "/models", "/inference", "/testing", "/training"].some((prefix) =>
    pathname.startsWith(prefix)
  );

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
        className={`app-shell bg-[#f5f7f9] text-ink ${isProjectArea ? "app-shell-project" : ""} ${
          isProjectArea && !projectSidebarOpen ? "app-shell-project-collapsed" : ""
        }`}
      >
        {isProjectArea ? (
          <>
            <GlobalSidebar pathname={pathname} compact />
            <ProjectSidebar
              pathname={pathname}
              projectId={projectId}
              project={project}
              projects={projects}
              open={projectSidebarOpen}
              setOpen={setProjectSidebarOpen}
              setProjectId={setProjectId}
            />
          </>
        ) : (
          <GlobalSidebar pathname={pathname} />
        )}
        <section className="content-shell">{children}</section>
      </main>
      {activeMutations > 0 && <GlobalLoadingOverlay />}
    </ProjectContext.Provider>
  );
}

function GlobalLoadingOverlay() {
  return (
    <div className="global-loading-overlay" role="status" aria-live="polite">
      <div className="global-loading-panel">
        <span className="global-loading-spinner" />
        <strong>Working</strong>
      </div>
    </div>
  );
}

function GlobalSidebar({ pathname, compact = false }: { pathname: string; compact?: boolean }) {
  return (
    <aside className={`sidebar sidebar-global ${compact ? "sidebar-global-compact" : ""}`}>
      <Link className="brand-block brand-link" href="/" title="Onestep Vision home">
        <span className="brand-mark" aria-hidden="true">
          <Image src="/brand/logo_transparent.png" alt="" width={42} height={42} priority />
        </span>
        <div className="brand-copy">
          <h1>Onestep Vision</h1>
          <span>Image intelligence workspace</span>
        </div>
      </Link>
      <nav className="side-nav">
        <SideLink href="/" active={pathname === "/"} icon={<ScanEye size={17} />}>
          Projects
        </SideLink>
        <SideLink href="/settings" active={pathname.startsWith("/settings")} icon={<Settings size={17} />}>
          Settings
        </SideLink>
      </nav>
      <div className="sidebar-spacer" />
      <div className="sidebar-note">ONESTEP product studio</div>
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
  if (!open) {
    return (
      <aside className="sidebar sidebar-project sidebar-project-collapsed">
        <button className="icon-button" onClick={() => setOpen(true)} title="Open project sidebar" type="button">
          <ChevronsRight size={17} />
        </button>
        <nav className="side-nav side-nav-icons">
          <SideLink href="/datasets" active={pathname.startsWith("/datasets")} icon={<Database size={17} />} iconOnly>
            Datasets
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
          <SideLink href="/inference" active={pathname.startsWith("/inference")} icon={<ImageIcon size={17} />} iconOnly>
            Inference
          </SideLink>
        </nav>
      </aside>
    );
  }

  return (
    <aside className="sidebar sidebar-project">
      <div className="project-sidebar-top">
        <Link className="project-back-link" href="/">
          <ArrowLeft size={16} />
          <span>Projects</span>
        </Link>
        <button className="icon-button" onClick={() => setOpen(false)} title="Close project sidebar" type="button">
          <ChevronsLeft size={17} />
        </button>
      </div>
      <div className="project-sidebar-title">
        <strong title={project?.name ?? "Project"}>{project?.name ?? "Project"}</strong>
        <span>Project workspace</span>
      </div>
      <label className="project-switcher">
        <span>Switch project</span>
        <select value={project?.id ?? projectId} onChange={(event) => setProjectId(event.target.value)}>
          {!projects.some((item) => item.id === projectId) && (
            <option value={projectId}>Project</option>
          )}
          {projects.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>
      </label>
      <nav className="side-nav">
        <SideLink href="/datasets" active={pathname.startsWith("/datasets")} icon={<Database size={17} />}>
          Datasets
        </SideLink>
        <SideLink href="/models" active={pathname.startsWith("/models")} icon={<Boxes size={17} />}>
          Models
        </SideLink>
        <SideLink href="/training" active={pathname.startsWith("/training")} icon={<Activity size={17} />}>
          Training
        </SideLink>
        <SideLink href="/testing" active={pathname.startsWith("/testing")} icon={<FlaskConical size={17} />}>
          Testing
        </SideLink>
        <SideLink href="/inference" active={pathname.startsWith("/inference")} icon={<ImageIcon size={17} />}>
          Inference
        </SideLink>
      </nav>
    </aside>
  );
}

function SideLink({
  href,
  active,
  icon,
  iconOnly = false,
  children
}: {
  href: string;
  active: boolean;
  icon: React.ReactNode;
  iconOnly?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Link className={`nav-link ${active ? "nav-link-active" : ""} ${iconOnly ? "nav-link-icon-only" : ""}`} href={href} title={String(children)}>
      {icon}
      {!iconOnly && <span>{children}</span>}
      {active && !iconOnly && <BarChart3 size={14} />}
    </Link>
  );
}
