import type { Metadata } from "next";
import { DocumentationPage } from "@/features/marketing/documentation-page";

export const metadata: Metadata = {
  title: "Documentations — Onestep AI Platform",
  description: "Complete system documentation, REST API specs, and local model runner architecture for Onestep AI Platform."
};

export default function Page() {
  return <DocumentationPage />;
}
