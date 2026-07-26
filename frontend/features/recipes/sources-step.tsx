"use client";

import { useRef } from "react";
import { AlertTriangle, ArrowRight, FileText, Trash2, Upload } from "lucide-react";
import { Badge, EmptyState, InlineSpinner, MutationError, PanelTitle } from "@/features/platform/ui";
import { useAddSourcesMutation, useDeleteSourceMutation } from "@/features/recipes/hooks";
import type { RecipeRead } from "@/types/api";

const ACCEPT = ".pdf,.docx,.txt,.md,.csv,.jsonl";

export function RecipeSourcesStep({ recipe, onNext }: { recipe: RecipeRead; onNext: () => void }) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const addMutation = useAddSourcesMutation(recipe.id);
  const deleteMutation = useDeleteSourceMutation(recipe.id);
  const usableSources = recipe.sources.filter((source) => !source.excluded).length;

  function upload(files: FileList | null) {
    if (!files || files.length === 0) return;
    const form = new FormData();
    Array.from(files).forEach((file) => form.append("files", file));
    addMutation.mutate(form);
  }

  return (
    <section className="panel">
      <div className="records-toolbar">
        <PanelTitle icon={<FileText size={18} />} title="Sources" />
        <div className="records-toolbar-actions">
          {addMutation.isPending && <InlineSpinner label="Extracting" />}
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT}
            multiple
            hidden
            onChange={(event) => {
              upload(event.target.files);
              event.target.value = "";
            }}
          />
          <button
            className="primary-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={addMutation.isPending}
          >
            <Upload size={16} /> Add files
          </button>
        </div>
      </div>
      <p className="mt-1 text-sm text-ink-subtle">
        PDF, DOCX, TXT, MD, CSV, or JSONL, up to 20 MB each. Text is extracted on upload; scanned or
        image-only PDFs are excluded with a warning (no OCR).
      </p>

      {recipe.sources.length === 0 ? (
        <EmptyState
          centered
          label="No sources yet."
          icon={<Upload size={28} />}
          description="Add the documents you want to turn into training records."
        />
      ) : (
        <div className="table-wrap mt-4">
          <table className="records-table">
            <thead>
              <tr>
                <th>File</th>
                <th className="records-col-num">Characters</th>
                <th className="records-col-num">Pages</th>
                <th>Status</th>
                <th className="records-col-actions" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {recipe.sources.map((source) => (
                <tr key={source.id} className="records-row">
                  <td>
                    <div className="recipe-source-file">
                      <span>{source.filename}</span>
                      <Badge tone="neutral">{source.media_type}</Badge>
                    </div>
                    {source.warnings.length > 0 && (
                      <span className="recipe-source-warning">
                        <AlertTriangle size={13} /> {source.warnings.join(" · ")}
                      </span>
                    )}
                  </td>
                  <td className="records-col-num">{source.characters.toLocaleString()}</td>
                  <td className="records-col-num">{source.pages ?? "—"}</td>
                  <td>
                    <Badge tone={source.excluded ? "warn" : "ok"}>
                      {source.excluded ? "excluded" : "ready"}
                    </Badge>
                  </td>
                  <td className="records-col-actions">
                    <button
                      className="icon-button"
                      title="Remove source"
                      onClick={() => deleteMutation.mutate(source.id)}
                      disabled={deleteMutation.isPending}
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <MutationError mutations={[addMutation, deleteMutation]} />

      <div className="recipe-step-footer">
        <span className="text-sm text-ink-subtle">
          {usableSources} of {recipe.sources.length} sources usable for generation
        </span>
        <button className="primary-button" onClick={onNext} disabled={usableSources === 0}>
          Continue <ArrowRight size={16} />
        </button>
      </div>
    </section>
  );
}
