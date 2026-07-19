import Image from "next/image";
import Link from "next/link";
import { ButtonLink } from "@/features/platform/ui";
import { SIGN_IN_HREF } from "./routes";

const REPO_HREF = "https://github.com/L007/onestep-ai-platform";

/**
 * Every claim on this page is traceable to README.md or docs/ai/rules.md. No
 * invented metrics, no testimonials, no logo wall — there are no users to quote
 * yet, and fabricating them is the fastest way to make an honest project read
 * as dishonest. Captures are the real app, unretouched.
 */
const STAGES = [
  {
    n: "01",
    title: "Organize",
    copy: "Create projects that scope image and text work separately. A project owns its datasets, its runs, and the models promoted out of them.",
    tags: ["Projects", "Vision + NLP"],
    image: "/brand/1_project_list.png",
    caption: "Project list"
  },
  {
    n: "02",
    title: "Prepare",
    copy: "Upload, label, annotate, split, and version datasets without rewriting the originals. Dataset Studio handles segmentation masks, classification labels, and text records.",
    tags: ["Annotation", "Splits", "Versioning"],
    image: "/brand/5_dataset_studio_nlp_clasification_task.png",
    caption: "Dataset Studio — text classification"
  },
  {
    n: "03",
    title: "Train",
    copy: "Start task-compatible training jobs from a prepared dataset version. Runs execute as subprocesses, so a long fine-tune never blocks the API.",
    tags: ["Subprocess runs", "Model zoo"],
    image: "/brand/6_training_details.png",
    caption: "Training run detail"
  },
  {
    n: "04",
    title: "Test",
    copy: "Score a trained model against a held-out split with task-aware metrics, then read the per-item results sitting behind each number.",
    tags: ["Task-aware metrics", "Per-item review"],
    image: "/brand/5_available_trained_model_list.png",
    caption: "Trained models"
  },
  {
    n: "05",
    title: "Inspect",
    copy: "Run inference on a single item and read the output with its overlays, class confidence, and full run history.",
    tags: ["Overlays", "Run history"],
    image: "/brand/7_inference_image_segmentation_task.png",
    caption: "Inference — image segmentation"
  }
];

const MODEL_FAMILIES = [
  {
    family: "YOLOv11",
    tasks: "Instance segmentation",
    notes: (
      <>
        Ultralytics weights. Default confidence <code>0.65</code>, inference IoU <code>0.7</code>.
      </>
    )
  },
  {
    family: "U-Net + Inception",
    tasks: "Segmentation, classification",
    notes: (
      <>
        Keras. <code>256×256</code> segmentation input, <code>299×299</code> classifier input.
      </>
    )
  },
  {
    family: "Hugging Face Transformers",
    tasks: "Text classification, summarization, question answering",
    notes: (
      <>
        Fine-tuned locally. Gated repos need <code>HUGGINGFACE_HUB_TOKEN</code>.
      </>
    )
  }
];

export function LandingPage() {
  return (
    <div className="landing">
      <header className="landing-nav">
        <Link className="landing-wordmark" href="/">
          <Image src="/brand/logo_transparent.png" alt="" width={30} height={30} priority />
          <strong>Onestep AI Platform</strong>
        </Link>
        <div className="landing-nav-actions">
          <a
            className="landing-nav-link landing-nav-link-secondary"
            href={REPO_HREF}
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
          <ButtonLink href={SIGN_IN_HREF} size="sm">
            Open the platform
          </ButtonLink>
        </div>
      </header>

      <main>
        <div className="landing-container">
          <section className="landing-hero">
            <div>
              <span className="landing-eyebrow">Open source · Apache 2.0</span>
              <h1 className="landing-h1">From raw data to measured model behavior.</h1>
              <p className="landing-lede">
                Onestep AI Platform is a local studio for computer vision and NLP. Label, prepare,
                train, test, and inspect models in one workspace instead of five disconnected
                notebooks.
              </p>
              <div className="landing-cta-row">
                <ButtonLink href={SIGN_IN_HREF}>Open the platform</ButtonLink>
                <ButtonLink variant="secondary" href={REPO_HREF} target="_blank" rel="noreferrer">
                  View the source
                </ButtonLink>
              </div>
              <p className="landing-note">
                Runs on your own machine. Datasets, weights, and predictions stay in local storage.
                Built for research and engineering work — it measures model behavior, and is not a
                diagnostic device.
              </p>
            </div>
            <figure className="landing-figure">
              <Image
                src="/brand/3_dataset_studio_image_segmentation_task.png"
                alt="Dataset Studio showing an image segmentation task with annotated regions"
                width={1915}
                height={927}
                priority
                sizes="(max-width: 60rem) 100vw, 45vw"
              />
              <figcaption>Dataset Studio — image segmentation</figcaption>
            </figure>
          </section>
        </div>

        <section className="landing-section">
          <div className="landing-container">
            <span className="landing-eyebrow">The workflow</span>
            <h2 className="landing-h2">Five stages, and each one hands off to the next.</h2>
            <p className="landing-lede">
              Continuity is the organizing idea: an empty dataset points at upload, a finished
              training run points at testing, a scored model points at inference.
            </p>

            <ol className="landing-stages">
              {STAGES.map((stage) => (
                <li className="landing-stage" key={stage.n}>
                  <div className="landing-stage-rail">
                    <span className="landing-stage-number">{stage.n}</span>
                  </div>
                  <div className="landing-stage-body">
                    <div>
                      <h3 className="landing-stage-title">{stage.title}</h3>
                      <p className="landing-stage-copy">{stage.copy}</p>
                      <ul className="landing-stage-tags">
                        {stage.tags.map((tag) => (
                          <li key={tag}>{tag}</li>
                        ))}
                      </ul>
                    </div>
                    <figure className="landing-figure">
                      <Image
                        src={stage.image}
                        alt={`${stage.title} stage — ${stage.caption}`}
                        width={1915}
                        height={927}
                        loading="lazy"
                        sizes="(max-width: 60rem) 100vw, 40vw"
                      />
                      <figcaption>{stage.caption}</figcaption>
                    </figure>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section className="landing-section">
          <div className="landing-container">
            <span className="landing-eyebrow">What it runs</span>
            <h2 className="landing-h2">Three model families, one normalized response.</h2>
            <p className="landing-lede">
              Inference output is normalized across families, so the review surface keeps its shape
              when the model underneath changes.
            </p>
            <div className="landing-table-wrap">
              <table className="landing-table">
                <thead>
                  <tr>
                    <th scope="col">Family</th>
                    <th scope="col">Tasks</th>
                    <th scope="col">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_FAMILIES.map((row) => (
                    <tr key={row.family}>
                      <th scope="row">{row.family}</th>
                      <td>{row.tasks}</td>
                      <td>{row.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className="landing-section">
          <div className="landing-container landing-split">
            <div>
              <h2 className="landing-h2">Two commands to a running stack.</h2>
              <p className="landing-lede">
                A FastAPI and SQLite backend, a Next.js frontend, and <code>make</code> targets that
                start both together with interleaved logs. Full setup lives in the{" "}
                <a
                  className="landing-inline-link"
                  href={`${REPO_HREF}#quickstart`}
                  target="_blank"
                  rel="noreferrer"
                >
                  README
                </a>
                .
              </p>
            </div>
            <pre className="landing-code">
              <code>
                {`cp .env.example .env
cp frontend/.env.example frontend/.env.local

`}
                <span className="landing-code-comment">
                  {`# verify uv, pnpm, Python 3.11, Node
`}
                </span>
                {`make doctor

`}
                <span className="landing-code-comment">
                  {`# backend :8000 + frontend :3000
`}
                </span>
                {`make dev`}
              </code>
            </pre>
          </div>
        </section>

        <section className="landing-close">
          <div className="landing-container">
            <h2 className="landing-h2">Start with a project.</h2>
            <p className="landing-lede">
              Create one, point it at a dataset, and the workspace carries you through to a scored
              model.
            </p>
            <div className="landing-cta-row">
              <ButtonLink href={SIGN_IN_HREF}>Open the platform</ButtonLink>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div className="landing-container landing-footer">
          <span>Onestep AI Platform — research and engineering workspace for vision and NLP.</span>
          <div className="landing-footer-links">
            <a href={REPO_HREF} target="_blank" rel="noreferrer">
              GitHub
            </a>
            <Link href="/documentation">Documentation</Link>
            <a href={`${REPO_HREF}/blob/main/LICENSE`} target="_blank" rel="noreferrer">
              Apache 2.0
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
