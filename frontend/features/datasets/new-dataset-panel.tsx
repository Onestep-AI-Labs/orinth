"use client";

import { useState, type ReactNode } from "react";
import { Cloud, FilePlus2, FileText, Upload } from "lucide-react";
import Link from "next/link";
import { IngestDropCard } from "@/features/datasets/prep/ingest-drop-card";
import { PanelTitle } from "@/features/platform/ui";

/**
 * The one way a dataset starts.
 *
 * There used to be four, presented as four peer buttons on the catalog toolbar —
 * "Add Dataset", "Browse HuggingFace", "New from documents" — plus a drop card
 * underneath them that did the same job better. Four buttons is four decisions
 * before a single byte moves, and three of the four were the wrong default: the
 * overwhelming case is "I have files", which the drop card handles without
 * asking anything at all.
 *
 * So this is one panel with one question — where is the data — answered by a
 * segmented control rather than by choosing between buttons that look equally
 * important. Files is the default and stays the visual centre. The other three
 * are the same components as before, moved rather than rewritten; nothing was
 * removed, it was demoted.
 */

type Source = "files" | "hub" | "documents" | "blank";

type Tab = { key: Source; label: string; icon: typeof Upload };

export function NewDatasetPanel({
  projectId,
  onCreated,
  hubImportPanel,
  createPanel,
  canImportHub
}: {
  projectId: string;
  onCreated: (datasetId: string) => void;
  hubImportPanel: ReactNode;
  createPanel: ReactNode;
  canImportHub: boolean;
}) {
  const [source, setSource] = useState<Source>("files");

  // HuggingFace and the document recipes are both LLM-dataset flows, so a
  // project that cannot hold one should not be offered either. Hiding them here
  // rather than disabling them keeps the row honest about what is available.
  const tabs: Tab[] = [
    { key: "files", label: "Files", icon: Upload },
    ...(canImportHub
      ? ([
          { key: "hub", label: "HuggingFace", icon: Cloud },
          { key: "documents", label: "Documents", icon: FileText }
        ] as Tab[])
      : []),
    { key: "blank", label: "Empty", icon: FilePlus2 }
  ];

  return (
    <section className="panel" data-tour="datasets-new">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <PanelTitle icon={<Upload size={18} />} title="New dataset" />
        {/* The system's existing segmented control (DESIGN.md §7), not a new one:
            this is the same "pick one view" gesture the task grid and the models
            tabs already use. */}
        <div className="segmented-control" role="tablist" aria-label="Dataset source">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.key}
                type="button"
                role="tab"
                aria-selected={source === tab.key}
                className={source === tab.key ? "segmented-active" : ""}
                onClick={() => setSource(tab.key)}
                data-tour={tab.key === "hub" ? "datasets-import" : undefined}
              >
                <Icon size={15} /> {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {source === "files" && <IngestDropCard projectId={projectId} onCreated={onCreated} />}
      {source === "hub" && hubImportPanel}
      {source === "documents" && (
        <div className="source-handoff">
          <p className="form-caption">
            PDFs, Word files and plain text become a training set by being read and turned into
            question-and-answer records. That needs a model and a few choices about what to
            generate, so it has a screen of its own.
          </p>
          <Link className="primary-button" href="/datasets/recipes">
            <FileText size={16} /> Build from documents
          </Link>
        </div>
      )}
      {source === "blank" && createPanel}
    </section>
  );
}
