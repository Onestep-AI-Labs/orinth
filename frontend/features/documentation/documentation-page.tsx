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
  KeyRound,
  Layers3,
  Route,
  ScanEye,
  Settings
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
    description: "Text classification, summarization, and question answering support samples, local baselines, Keras models, and Hugging Face options.",
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
    title: "Local assets",
    description: "Uploads, overlays, datasets, model assets, and training runs stay in ignored workspace storage.",
    icon: <CheckCircle2 size={17} />
  }
];

const workflowSteps = [
  "Create or select a project with the task types you need.",
  "Create, import, clone, or use a bundled sample dataset.",
  "Train a compatible local, Keras, Ultralytics, or Hugging Face model.",
  "Test completed models against task-compatible test splits.",
  "Run inference on images or text and inspect saved history."
];

const developerSteps = [
  "Add a task-first model catalog entry.",
  "Write a subprocess runner and normalized artifacts.",
  "Register a lazy predictor and promotion path.",
  "Keep frontend controls task-aware through option defaults.",
  "Add tests and update the matching spec."
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
            Onestep AI Platform is organized around project-scoped AI research workflows for vision and
            NLP tasks. Use it to prepare local datasets, train runnable models, compare test metrics, and
            inspect inference outputs without turning the workspace into a clinical decision system.
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
        <PanelTitle icon={<Route size={18} />} title="Workflow" />
        <div className="documentation-note-grid">
          {workflowSteps.map((step, index) => (
            <article className="documentation-note" key={step}>
              <span className="documentation-note-icon">{index + 1}</span>
              <div>
                <strong>{step}</strong>
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
        <PanelTitle icon={<Settings size={18} />} title="Settings" />
        <div className="documentation-note-grid">
          <article className="documentation-note">
            <span className="documentation-note-icon"><KeyRound size={17} /></span>
            <div>
              <strong>Hugging Face token</strong>
              <p>Settings can save `HUGGINGFACE_HUB_TOKEN` to the ignored workspace `.env` for authenticated Hub downloads.</p>
            </div>
          </article>
          <article className="documentation-note">
            <span className="documentation-note-icon"><CheckCircle2 size={17} /></span>
            <div>
              <strong>Secret handling</strong>
              <p>The token status is visible, but the token value is never returned by the API.</p>
            </div>
          </article>
        </div>
      </section>
      <section className="panel">
        <PanelTitle icon={<FileText size={18} />} title="Model Extensions" />
        <div className="documentation-note-grid">
          {developerSteps.map((step, index) => (
            <article className="documentation-note" key={step}>
              <span className="documentation-note-icon">{index + 1}</span>
              <div>
                <strong>{step}</strong>
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
