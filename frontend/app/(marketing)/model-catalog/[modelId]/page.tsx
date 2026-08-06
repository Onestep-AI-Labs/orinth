import type { Metadata } from "next";
import { ModelDetailPage } from "@/features/marketing/model-detail-page";
import { SYSTEM_MODELS } from "@/features/marketing/models-data";

interface PageProps {
  params: {
    modelId: string;
  };
}

export function generateStaticParams() {
  return SYSTEM_MODELS.map((model) => ({
    modelId: model.id
  }));
}

export function generateMetadata({ params }: PageProps): Metadata {
  const model = SYSTEM_MODELS.find((m) => m.id === params.modelId);
  return {
    title: model ? `${model.name} · Models Catalog · Onestep AI Platform` : "Model Details · Onestep AI Platform",
    description: model ? model.description : "Details of system model in Onestep AI Platform."
  };
}

export default function Page({ params }: PageProps) {
  return <ModelDetailPage modelId={params.modelId} />;
}
