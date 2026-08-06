import type { Metadata } from "next";
import { ModelsPage } from "@/features/marketing/models-page";

export const metadata: Metadata = {
  title: "Models Catalog — Onestep AI Platform | Vision, NLP & LLMs",
  description: "Explore reference weights, pretrained baselines, fine-tuned transformers, GGUF chat models, and custom neural architectures available in Onestep AI Platform.",
  keywords: [
    "AI model catalog",
    "YOLOv11 pretrained weights",
    "GGUF local models",
    "Keras UNet weights",
    "Hugging Face transformers model zoo",
    "computer vision model catalog",
    "local LLM models"
  ],
  openGraph: {
    title: "Models Catalog — Onestep AI Platform | Vision, NLP & LLMs",
    description: "Explore reference weights, pretrained baselines, fine-tuned transformers, and GGUF chat models.",
    siteName: "Onestep AI Platform",
    type: "website"
  },
  twitter: {
    card: "summary_large_image",
    title: "Models Catalog — Onestep AI Platform",
    description: "Explore pretrained vision baselines, Hugging Face transformers, and GGUF chat models."
  }
};

export default function Page() {
  return <ModelsPage />;
}
