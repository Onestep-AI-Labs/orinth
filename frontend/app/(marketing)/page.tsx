import type { Metadata } from "next";
import { LandingPage } from "@/features/marketing/landing-page";

export const metadata: Metadata = {
  title: "Onestep AI Platform — Run, Train & Evaluate AI Models Locally",
  description: "Open-source, privacy-first local workspace for Computer Vision, NLP, and LLM intelligence. Run YOLOv11, U-Net, Transformers, and GGUF chat 100% on your own machine.",
  keywords: [
    "Onestep AI Platform",
    "local AI platform",
    "open source AI workspace",
    "run AI models locally",
    "YOLOv11 local inference",
    "Keras UNet segmentation",
    "Hugging Face fine-tuning",
    "local GGUF LLM serving",
    "dataset annotation studio",
    "privacy-first machine learning",
    "computer vision NLP platform"
  ],
  authors: [{ name: "Onestep AI Platform Team" }],
  openGraph: {
    title: "Onestep AI Platform — Run, Train & Evaluate AI Models Locally",
    description: "Open-source, privacy-first local workspace for Computer Vision, NLP, and LLM intelligence. Zero cloud dependency.",
    url: "https://github.com/L007/onestep-ai-platform",
    siteName: "Onestep AI Platform",
    locale: "en_US",
    type: "website",
    images: [
      {
        url: "/brand/logo_transparent.png",
        width: 512,
        height: 512,
        alt: "Onestep AI Platform Logo"
      }
    ]
  },
  twitter: {
    card: "summary_large_image",
    title: "Onestep AI Platform — Run, Train & Evaluate AI Models Locally",
    description: "Open-source, privacy-first local workspace for Computer Vision, NLP, and LLM intelligence.",
    images: ["/brand/logo_transparent.png"]
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1
    }
  }
};

export default function Page() {
  return <LandingPage />;
}
