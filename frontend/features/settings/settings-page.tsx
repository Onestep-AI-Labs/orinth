"use client";

import { Settings } from "lucide-react";
import { useProject } from "@/components/app-shell";
import { Metric, PageHeader, PanelTitle } from "@/features/platform/ui";

export function SettingsPage() {
  const { projects } = useProject();
  return (
    <div className="space-y-5">
      <PageHeader title="Settings" subtitle="Platform preferences" icon={<Settings size={20} />} />
      <section className="panel">
        <PanelTitle icon={<Settings size={18} />} title="Platform Overview" />
        <div className="metric-grid mt-4">
          <Metric label="Total projects" value={projects.length} />
          <Metric label="Account settings" value="Coming soon" />
          <Metric label="Usage limits" value="Local" />
          <Metric label="Storage mode" value="Workspace" />
        </div>
      </section>
      <section className="panel">
        <PanelTitle icon={<Settings size={18} />} title="Account Settings" />
        <p className="hint-text">Profile, team access, and authentication controls are coming soon.</p>
      </section>
      <section className="panel">
        <PanelTitle icon={<Settings size={18} />} title="Usage Limits" />
        <div className="metric-grid mt-4">
          <Metric label="Compute policy" value="Local only" />
          <Metric label="Storage quota" value="Not enforced" />
          <Metric label="API keys" value="Not configured" />
        </div>
      </section>
    </div>
  );
}
