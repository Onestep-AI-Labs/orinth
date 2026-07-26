import { Suspense } from "react";
import { RecipeWorkspace } from "@/components/platform-pages";

export default function Page({ params }: { params: { recipeId: string } }) {
  return (
    <Suspense fallback={null}>
      <RecipeWorkspace recipeId={params.recipeId} />
    </Suspense>
  );
}
