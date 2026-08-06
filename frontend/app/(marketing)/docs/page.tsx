import type { Metadata } from "next";
import { DocumentationPage } from "@/features/marketing/documentation-page";

export const metadata: Metadata = {
  title: "Documentations — Onestep AI Platform | System Architecture & REST API",
  description: "Technical documentation, REST API specifications, model runner architecture, Dataset Studio guides, and local deployment options for Onestep AI Platform.",
  keywords: [
    "Onestep AI documentation",
    "REST API reference",
    "local model deployment",
    "YOLOv11 API",
    "GGUF serving guide",
    "Dataset Studio documentation",
    "machine learning API endpoints",
    "Python FastAPI ML backend"
  ],
  openGraph: {
    title: "Documentations — Onestep AI Platform | System Architecture & REST API",
    description: "Technical documentation, REST API specifications, model runner architecture, and local deployment options.",
    siteName: "Onestep AI Platform",
    type: "website"
  },
  twitter: {
    card: "summary_large_image",
    title: "Documentations — Onestep AI Platform",
    description: "Technical documentation, REST API specifications, and local model runner architecture."
  }
};

export default function Page() {
  return <DocumentationPage />;
}
