import { ModelDetailPage } from "@/components/platform-pages";

export default function Page({ params }: { params: { modelId: string } }) {
  return <ModelDetailPage modelId={params.modelId} />;
}
