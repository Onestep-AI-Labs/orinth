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
  const title = model ? `${model.name} — ${model.family} Model Specs | Onestep AI Platform` : "Model Specs — Onestep AI Platform";
  const description = model ? model.description : "Technical details and inference specifications of system model in Onestep AI Platform.";
  const keywords = model
    ? [model.name, model.family, model.taskLabel, model.format, "Onestep AI Platform", "local AI model", "model specifications"]
    : ["Onestep AI Platform", "model details"];

  return {
    title,
    description,
    keywords,
    openGraph: {
      title,
      description,
      siteName: "Onestep AI Platform",
      type: "website"
    },
    twitter: {
      card: "summary_large_image",
      title,
      description
    }
  };
}

export default function Page({ params }: PageProps) {
  return <ModelDetailPage modelId={params.modelId} />;
}
