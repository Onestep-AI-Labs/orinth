"use client";

import { useState } from "react";
import { Database, PackageCheck } from "lucide-react";
import { Field, MutationError, PanelTitle } from "@/features/platform/ui";
import { useCommitRecipeMutation } from "@/features/recipes/hooks";
import type { RecipeRead } from "@/types/api";

export function RecipeCommitStep({
  recipe,
  onCommitted
}: {
  recipe: RecipeRead;
  onCommitted: (datasetId: string) => void;
}) {
  const [name, setName] = useState(recipe.name);
  const commitMutation = useCommitRecipeMutation(recipe.id, (response) => onCommitted(response.dataset.id));

  return (
    <section className="panel">
      <PanelTitle icon={<PackageCheck size={18} />} title="Commit to a dataset" />
      <p className="mt-1 text-sm text-ink-subtle">
        Creates a new <strong>llm_finetune</strong> dataset ({recipe.output_format === "chat_jsonl" ? "chat" : "instruction"})
        with these {recipe.record_count} records in the unassigned inbox, ready for the standard split flow. The recipe
        stays editable — you can commit again to create another dataset.
      </p>

      <div className="mt-4 max-w-sm">
        <Field label="Dataset name">
          <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Dataset name" />
        </Field>
      </div>

      <MutationError mutations={[commitMutation]} />

      <div className="recipe-step-footer">
        <span className="text-sm text-ink-subtle">{recipe.record_count} records will be committed.</span>
        <button
          className="primary-button"
          onClick={() => commitMutation.mutate(name.trim() || recipe.name)}
          disabled={commitMutation.isPending || recipe.record_count === 0}
        >
          <Database size={16} /> Commit dataset
        </button>
      </div>
    </section>
  );
}
