import { Suspense } from "react";
import { RecipesPage } from "@/components/platform-pages";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <RecipesPage />
    </Suspense>
  );
}
