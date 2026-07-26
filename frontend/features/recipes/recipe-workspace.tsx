"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { ArrowLeft, Check, FileText } from "lucide-react";
import { PageHeader, StatusBadge, TableSkeleton } from "@/features/platform/ui";
import { useRecipeQuery } from "@/features/recipes/hooks";
import { RecipeSourcesStep } from "@/features/recipes/sources-step";
import { RecipeGenerateStep } from "@/features/recipes/generate-step";
import { RecipeReviewStep } from "@/features/recipes/review-step";
import { RecipeCommitStep } from "@/features/recipes/commit-step";

type StepKey = "sources" | "generate" | "review" | "commit";

const STEPS: { key: StepKey; label: string }[] = [
  { key: "sources", label: "Sources" },
  { key: "generate", label: "Generate" },
  { key: "review", label: "Review" },
  { key: "commit", label: "Commit" }
];

export function RecipeWorkspace({ recipeId }: { recipeId: string }) {
  const router = useRouter();
  const recipeQuery = useRecipeQuery(recipeId);
  const recipe = recipeQuery.data;

  const initialStep = useMemo<StepKey>(() => {
    if (!recipe) return "sources";
    if (recipe.record_count > 0) return "review";
    if (recipe.sources.length > 0) return "generate";
    return "sources";
    // Only seed from the first load; the user drives it after that.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recipe?.id]);

  const [step, setStep] = useState<StepKey>("sources");
  const [seeded, setSeeded] = useState(false);
  if (recipe && !seeded) {
    setStep(initialStep);
    setSeeded(true);
  }

  if (recipeQuery.isLoading || !recipe) {
    return (
      <div className="space-y-5">
        <PageHeader title="Recipe" subtitle="Loading" icon={<FileText size={20} />} />
        <section className="panel"><TableSkeleton rows={4} /></section>
      </div>
    );
  }

  const stepIndex = STEPS.findIndex((entry) => entry.key === step);

  return (
    <div className="space-y-5">
      <PageHeader
        title={recipe.name}
        subtitle={recipe.output_format === "chat_jsonl" ? "Chat recipe" : "Instruction recipe"}
        icon={<FileText size={20} />}
        actions={
          <div className="flex items-center gap-2">
            <StatusBadge status={recipe.status} />
            <button className="secondary-button" onClick={() => router.push("/datasets/recipes")}>
              <ArrowLeft size={16} /> All recipes
            </button>
          </div>
        }
      />

      <nav className="recipe-stepper" aria-label="Recipe steps">
        {STEPS.map((entry, index) => (
          <button
            key={entry.key}
            type="button"
            className={`recipe-step ${step === entry.key ? "recipe-step-active" : ""} ${
              index < stepIndex ? "recipe-step-done" : ""
            }`}
            onClick={() => setStep(entry.key)}
          >
            <span className="recipe-step-index">
              {index < stepIndex ? <Check size={14} /> : index + 1}
            </span>
            {entry.label}
          </button>
        ))}
      </nav>

      {step === "sources" && (
        <RecipeSourcesStep recipe={recipe} onNext={() => setStep("generate")} />
      )}
      {step === "generate" && (
        <RecipeGenerateStep recipe={recipe} onReview={() => setStep("review")} />
      )}
      {step === "review" && (
        <RecipeReviewStep recipe={recipe} onCommit={() => setStep("commit")} />
      )}
      {step === "commit" && (
        <RecipeCommitStep recipe={recipe} onCommitted={(datasetId) => router.push(`/datasets?dataset=${datasetId}`)} />
      )}
    </div>
  );
}
