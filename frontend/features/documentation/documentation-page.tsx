"use client";

import Link from "next/link";
import {
  Activity,
  BookOpen,
  CheckCircle2,
  Database,
  FileText,
  FlaskConical,
  ImageIcon,
  Layers3,
  Route,
  ScanEye
} from "lucide-react";
import { PageHeader, PanelTitle } from "@/features/platform/ui";

const sections = [
  {
    title: "Dataset Studio",
    status: "Vision + NLP",
    description: "Build project-scoped image or text datasets, upload files, manage labels, inspect EDA, and prepare annotation-ready splits.",
    href: "/datasets",
    icon: <Database size={17} />
  },
  {
    title: "Training",
    status: "Vision + NLP",
    description: "Configure local model runs, prepare assets, track progress, and register completed artifacts back to the project.",
    href: "/training",
    icon: <Activity size={17} />
  },
  {
    title: "Testing",
    status: "Vision + NLP",
    description: "Evaluate project models against compatible test splits, compare results, and inspect task-aware metrics.",
    href: "/testing",
    icon: <FlaskConical size={17} />
  },
  {
    title: "Inference",
    status: "Vision + NLP",
    description: "Run project-scoped models on images or text and review prediction history without crossing workspace boundaries.",
    href: "/inference",
    icon: <ScanEye size={17} />
  },
  {
    title: "NLP Workspace",
    status: "Active",
    description: "Text classification, summarization, and question answering use offline local baselines across dataset, training, testing, and inference.",
    href: "/documentation",
    icon: <FileText size={17} />
  }
];

const tracks = [
  {
    title: "Vision task track",
    status: "Current",
    description: "Classification, object detection, and segmentation workflows are available across dataset, training, testing, and inference modules.",
    icon: <ImageIcon size={18} />
  },
  {
    title: "NLP task track",
    status: "Current",
    description: "Text classification, summarization, question answering, evaluation, and lightweight local NLP baselines are available in the same workspace model.",
    icon: <FileText size={18} />
  }
];

const notes = [
  {
    title: "Project isolation",
    description: "Datasets, available models, inference history, testing jobs, and training jobs follow the selected project.",
    icon: <Layers3 size={17} />
  },
  {
    title: "Research workflow",
    description: "The platform is built for experiment management, comparison, and local engineering review.",
    icon: <Route size={17} />
  },
  {
    title: "Current domain",
    description: "Vision and NLP workflows are active using the same project-first model.",
    icon: <CheckCircle2 size={17} />
  }
];

export function DocumentationPage() {
  return (
    <div className="space-y-5">
      <PageHeader title="Documentation" subtitle="Onestep AI Platform guide" icon={<BookOpen size={20} />} />
      <section className="panel documentation-intro">
        <div>
          <span className="badge badge-ok">General AI research platform</span>
          <h3>One project structure for datasets, training, testing, and inference.</h3>
          <p>
            Onestep AI Platform is organized around project-scoped AI research workflows. The current
            implementation focuses on vision tasks, and the next expansion will bring NLP workflows into
            the same workspace model.
          </p>
        </div>
      </section>
      <section className="panel">
        <PanelTitle icon={<BookOpen size={18} />} title="Workspace Modules" />
        <div className="documentation-grid">
          {sections.map((section) => (
            <article className="documentation-card" key={section.title}>
              <div className="documentation-card-header">
                <span className="documentation-card-icon">{section.icon}</span>
                <div>
                  <strong>{section.title}</strong>
                  <span>{section.status}</span>
                </div>
              </div>
              <p>{section.description}</p>
              <div className="documentation-card-actions">
                <Link className="secondary-button" href={section.href}>
                  Open
                </Link>
              </div>
            </article>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelTitle icon={<Route size={18} />} title="Research Tracks" />
        <div className="documentation-track-grid">
          {tracks.map((track) => (
            <article className="documentation-track" key={track.title}>
              <span className="documentation-card-icon">{track.icon}</span>
              <div>
                <div className="documentation-track-title">
                  <strong>{track.title}</strong>
                  <span className="badge">{track.status}</span>
                </div>
                <p>{track.description}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelTitle icon={<BookOpen size={18} />} title="Reference Notes" />
        <div className="documentation-note-grid">
          {notes.map((note) => (
            <article className="documentation-note" key={note.title}>
              <span className="documentation-note-icon">{note.icon}</span>
              <div>
                <strong>{note.title}</strong>
                <p>{note.description}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
