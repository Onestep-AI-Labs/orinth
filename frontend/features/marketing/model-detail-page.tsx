"use client";

import Link from "next/link";
import { notFound } from "next/navigation";
import { ButtonLink } from "@/features/platform/ui";
import { CodeBlock } from "./code-block";
import { MarketingNav } from "./marketing-nav";
import { SYSTEM_MODELS } from "./models-data";
import { MODELS_HREF, SIGN_IN_HREF } from "./routes";

interface ModelDetailPageProps {
  modelId: string;
}

export function ModelDetailPage({ modelId }: ModelDetailPageProps) {
  const model = SYSTEM_MODELS.find((m) => m.id === modelId);

  if (!model) {
    notFound();
  }

  return (
    <div className="landing">
      <MarketingNav />

      <main>
        <div className="landing-container" style={{ paddingTop: "2rem", paddingBottom: "4rem" }}>
          <div className="model-detail-breadcrumbs">
            <Link href="/">Home</Link>
            <span>/</span>
            <Link href={MODELS_HREF}>Models</Link>
            <span>/</span>
            <strong>{model.name}</strong>
          </div>

          <header className="model-detail-header">
            <div>
              <div className="model-detail-badges">
                <span className="model-card-badge">{model.taskLabel}</span>
                <span className="model-card-format">{model.format}</span>
                <span className="model-detail-size">Disk size: {model.size}</span>
              </div>
              <h1 className="model-detail-title">{model.name}</h1>
              <p className="model-detail-family">Family: <strong>{model.family}</strong></p>
              <p className="landing-lede" style={{ marginTop: "1rem" }}>{model.description}</p>
            </div>

            <div className="model-detail-actions">
              <ButtonLink href={SIGN_IN_HREF}>Try in Workspace</ButtonLink>
              <ButtonLink variant="secondary" href={MODELS_HREF}>Back to Models Catalog</ButtonLink>
            </div>
          </header>

          <section className="model-detail-section">
            <h2 className="landing-h2" style={{ fontSize: "1.35rem", marginBottom: "1rem" }}>Model Overview</h2>
            <p className="landing-lede">{model.longDescription}</p>
          </section>

          <section className="model-detail-section">
            <h2 className="landing-h2" style={{ fontSize: "1.35rem", marginBottom: "1rem" }}>Key Features & Capabilities</h2>
            <ul className="model-detail-features">
              {model.features.map((feat) => (
                <li key={feat}>{feat}</li>
              ))}
            </ul>
          </section>

          <section className="model-detail-section">
            <h2 className="landing-h2" style={{ fontSize: "1.35rem", marginBottom: "1rem" }}>Benchmark Metrics</h2>
            <div className="model-detail-metrics-grid">
              {Object.entries(model.metrics).map(([metricName, value]) => (
                <div key={metricName} className="model-detail-metric-card">
                  <span className="model-card-metric-label">{metricName}</span>
                  <strong className="model-detail-metric-value">{value}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="model-detail-section">
            <h2 className="landing-h2" style={{ fontSize: "1.35rem", marginBottom: "1rem" }}>Input & Output Specifications</h2>
            <div className="landing-table-wrap">
              <table className="landing-table">
                <tbody>
                  <tr>
                    <th scope="row" style={{ width: "200px" }}>Input Format</th>
                    <td>{model.inputFormat}</td>
                  </tr>
                  <tr>
                    <th scope="row">Output Format</th>
                    <td>{model.outputFormat}</td>
                  </tr>
                  <tr>
                    <th scope="row">Weight Storage</th>
                    <td><code>{model.format}</code> — Local workspace artifact storage</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section className="model-detail-section">
            <h2 className="landing-h2" style={{ fontSize: "1.35rem", marginBottom: "1rem" }}>Usage Code Snippet</h2>
            <p className="landing-lede" style={{ marginBottom: "1rem" }}>
              Run inference programmatically using Onestep AI Platform ML runners:
            </p>
            <CodeBlock code={model.codeExample} label={`${model.name} Python Usage`} />
          </section>

          <section className="landing-close" style={{ marginTop: "4rem" }}>
            <div className="landing-container">
              <h2 className="landing-h2">Test or Fine-tune {model.name} in Onestep AI Platform.</h2>
              <p className="landing-lede">
                Open the platform to run inference, evaluate metrics on custom datasets, or fine-tune models locally.
              </p>
              <div className="landing-cta-row">
                <ButtonLink href={SIGN_IN_HREF}>Open the platform</ButtonLink>
              </div>
            </div>
          </section>
        </div>
      </main>

      <footer>
        <div className="landing-container landing-footer-simple">
          <p className="landing-footer-text">
            Onestep AI Platform — Research and engineering workspace for Vision &amp; NLP intelligence.
          </p>
        </div>
      </footer>
    </div>
  );
}
