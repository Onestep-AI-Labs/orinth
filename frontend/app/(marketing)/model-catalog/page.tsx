import type { Metadata } from "next";
import { ModelsPage } from "@/features/marketing/models-page";

export const metadata: Metadata = {
  title: "Available Models Catalog · Onestep AI Platform",
  description: "Browse available Computer Vision, NLP, LLM, and Architecture Studio models in Onestep AI Platform."
};

export default function Page() {
  return <ModelsPage />;
}
