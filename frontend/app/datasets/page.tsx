import { Suspense } from "react";
import { DatasetPage } from "@/components/platform-pages";

export default function Page() {
  return (
    <Suspense fallback={null}>
      <DatasetPage />
    </Suspense>
  );
}
