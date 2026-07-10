"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Boxes,
  Database,
  FlaskConical,
  ImageIcon,
  Layers3,
  Settings
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ProjectSummary } from "@/types/api";

const DEFAULT_PROJECT_ID = "default-research-project";

type ProjectContextValue = {
  projectId: string;
  project: ProjectSummary | null;
  projects: ProjectSummary[];
  setProjectId: (value: string) => void;
  refreshProjects: () => void;
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
  }, []);

  useEffect(() => {
    if (project?.id) {
      setProjectId(project.id);
      window.localStorage.setItem("image-platform-project", project.id);
    }
  }, [project?.id]);

  const value = {
    projectId,
    project,
    projects,
    setProjectId,
    refreshProjects: () => {
      projectsQuery.refetch();
    }
  };

  return (
    <ProjectContext.Provider value={value}>
      <main className="app-shell bg-[#f5f7f9] text-ink">
        {isProjectArea ? (
          <ProjectSidebar pathname={pathname} project={project} />
        ) : (
          <GlobalSidebar pathname={pathname} />
        )}
        <section className="content-shell">{children}</section>
      </main>
    </ProjectContext.Provider>
  );
}

function GlobalSidebar({ pathname }: { pathname: string }) {
  return (
    <aside className="sidebar sidebar-global">
      <div className="brand-block">
        <div className="grid h-10 w-10 place-items-center rounded-md bg-ink text-white">
          <Layers3 size={21} />
        </div>
        <div>
          <h1>Image Platform</h1>
          <span>Local research workspace</span>
        </div>
      </div>
      <nav className="side-nav">
        <SideLink href="/" active={pathname === "/"} icon={<Layers3 size={17} />}>
          Projects
        </SideLink>
        <SideLink href="/settings" active={pathname.startsWith("/settings")} icon={<Settings size={17} />}>
          Settings
        </SideLink>
      </nav>
      <div className="sidebar-spacer" />
      <div className="sidebar-note">Local image research platform</div>
    </aside>
  );
}

function ProjectSidebar({ pathname, project }: { pathname: string; project: ProjectSummary | null }) {
  return (
    <aside className="sidebar sidebar-project">
      <Link className="project-back-link" href="/">
        <ArrowLeft size={16} />
        <span>Projects</span>
      </Link>
      <div className="project-sidebar-title">
        <strong title={project?.name ?? "Project"}>{project?.name ?? "Project"}</strong>
        <span>Project workspace</span>
      </div>
      <nav className="side-nav">
        <SideLink href="/datasets" active={pathname.startsWith("/datasets")} icon={<Database size={17} />}>
          Datasets
        </SideLink>
        <SideLink href="/models" active={pathname.startsWith("/models")} icon={<Boxes size={17} />}>
          Models
        </SideLink>
        <SideLink href="/inference" active={pathname.startsWith("/inference")} icon={<ImageIcon size={17} />}>
          Inference
        </SideLink>
        <SideLink href="/testing" active={pathname.startsWith("/testing")} icon={<FlaskConical size={17} />}>
          Testing
        </SideLink>
        <SideLink href="/training" active={pathname.startsWith("/training")} icon={<Activity size={17} />}>
          Training
        </SideLink>
      </nav>
    </aside>
  );
}

function SideLink({
  href,
  active,
  icon,
  children
}: {
  href: string;
  active: boolean;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Link className={`nav-link ${active ? "nav-link-active" : ""}`} href={href}>
      {icon}
      <span>{children}</span>
      {active && <BarChart3 size={14} />}
    </Link>
  );
}
