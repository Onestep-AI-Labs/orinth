"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft, FilePlus2, FileText, Sparkles, Trash2 } from "lucide-react";
import { useProject } from "@/components/app-shell";
import { Badge, EmptyState, Field, MutationError, PageHeader, PanelTitle, StatusBadge, useConfirmationDialog } from "@/features/platform/ui";
import { RECIPE_PRESETS, type RecipePreset } from "@/features/recipes/presets";
import { useCreateRecipeMutation, useDeleteRecipeMutation, useRecipesQuery } from "@/features/recipes/hooks";
import type { RecipeRead } from "@/types/api";

export function RecipesPage() {
  const router = useRouter();
  const { projectId, project } = useProject();
  const [name, setName] = useState("");
  const recipesQuery = useRecipesQuery(projectId);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const createMutation = useCreateRecipeMutation(projectId, (recipe) =>
    router.push(`/datasets/recipes/${recipe.id}`)
  );
  const deleteMutation = useDeleteRecipeMutation(projectId);

  const allowsLlm = (project?.task_types ?? []).includes("llm_finetune");
  const recipes = recipesQuery.data ?? [];

  function createFromPreset(preset: RecipePreset | null) {
    createMutation.mutate({
      project_id: projectId,
      name: name.trim() || preset?.name || "Untitled recipe",
      output_format: preset?.output_format ?? "instruction_jsonl",
      generation: {
        mode: "auto",
        prompt_flavor: preset?.prompt_flavor ?? "qa",
        chunk_size: preset?.chunk_size ?? 3000,
        chunk_overlap: preset?.chunk_overlap ?? 200,
        records_per_chunk: preset?.records_per_chunk ?? 3,
        model: null
      }
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Data recipes"
        subtitle={project?.name ?? "Project"}
        icon={<FileText size={20} />}
        actions={
          <button className="secondary-button" onClick={() => router.push("/datasets")}>
            <ArrowLeft size={16} /> Back to datasets
          </button>
        }
      />

      {!allowsLlm && (
        <section className="panel">
          <EmptyState
            centered
            label="This project does not allow LLM fine-tuning datasets."
            icon={<Sparkles size={28} />}
            description="Add the 'llm_finetune' task in project settings to build datasets from documents."
          />
        </section>
      )}

      {allowsLlm && (
        <>
          <section className="panel">
            <PanelTitle icon={<Sparkles size={18} />} title="Start from a template" />
            <p className="mt-1 text-sm text-ink-subtle">
              Templates prefill the output format, chunking, and generation style. Everything stays editable
              afterward. Generated records are research data — never clinical guidance.
            </p>
            <div className="mt-4 max-w-sm">
              <Field label="Recipe name">
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="e.g. Cardiology handbook QA"
                />
              </Field>
            </div>
            <div className="recipe-template-grid mt-4">
              {RECIPE_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  className="recipe-template-card"
                  onClick={() => createFromPreset(preset)}
                  disabled={createMutation.isPending}
                >
                  <span className="recipe-template-icon"><FileText size={17} /></span>
                  <strong>{preset.name}</strong>
                  <span className="recipe-template-desc">{preset.description}</span>
                  <span className="recipe-template-badges">
                    {preset.concepts.map((concept) => (
                      <Badge key={concept} tone="neutral">{concept}</Badge>
                    ))}
                  </span>
                </button>
              ))}
              <button
                type="button"
                className="recipe-template-card recipe-template-card-blank"
                onClick={() => createFromPreset(null)}
                disabled={createMutation.isPending}
              >
                <span className="recipe-template-icon"><FilePlus2 size={17} /></span>
                <strong>Start blank</strong>
                <span className="recipe-template-desc">
                  An empty instruction recipe. Configure sources and generation yourself.
                </span>
              </button>
            </div>
            <MutationError mutations={[createMutation]} />
          </section>

          <section className="panel">
            <PanelTitle icon={<FileText size={18} />} title="Your recipes" />
            {recipes.length === 0 ? (
              <EmptyState
                centered
                label="No recipes yet."
                icon={<FileText size={28} />}
                description="Pick a template above to turn your documents into a training dataset."
              />
            ) : (
              <div className="recipe-list mt-4">
                {recipes.map((recipe) => (
                  <RecipeRow
                    key={recipe.id}
                    recipe={recipe}
                    onOpen={() => router.push(`/datasets/recipes/${recipe.id}`)}
                    onDelete={() =>
                      confirm({
                        title: "Delete recipe?",
                        message: "This removes the recipe and all its sources and generated records. This cannot be undone.",
                        confirmLabel: "Delete recipe",
                        onConfirm: () => deleteMutation.mutate(recipe.id)
                      })
                    }
                  />
                ))}
              </div>
            )}
            <MutationError mutations={[deleteMutation]} />
          </section>
        </>
      )}
      {confirmationDialog}
    </div>
  );
}

function RecipeRow({ recipe, onOpen, onDelete }: { recipe: RecipeRead; onOpen: () => void; onDelete: () => void }) {
  return (
    <div className="recipe-row">
      <button className="recipe-row-main" type="button" onClick={onOpen}>
        <div className="recipe-row-title">
          <strong>{recipe.name}</strong>
          <span>{recipe.output_format === "chat_jsonl" ? "Chat" : "Instruction"}</span>
        </div>
        <div className="recipe-row-meta">
          <span>{recipe.sources.length} sources</span>
          <span>{recipe.record_count} records</span>
        </div>
      </button>
      <div className="recipe-row-actions">
        <StatusBadge status={recipe.status} />
        <button className="icon-button" type="button" onClick={onDelete} title="Delete recipe">
          <Trash2 size={15} />
        </button>
      </div>
    </div>
  );
}
