"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  FlaskConical,
  ImageIcon,
  Play,
  RefreshCw,
  Upload
} from "lucide-react";
import { QueryClient, QueryClientProvider, useMutation, useQuery } from "@tanstack/react-query";
import { api, mediaUrl } from "@/lib/api";
import type { EvaluationJob, InferenceResult, ModelInfo, TrainingJob } from "@/types/api";

type TabKey = "inference" | "testing" | "training";

export default function Page() {
  const [client] = useState(() => new QueryClient());
  return (
    <QueryClientProvider client={client}>
      <Workspace />
    </QueryClientProvider>
  );
}

function Workspace() {
  const [activeTab, setActiveTab] = useState<TabKey>("inference");
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: api.models });

  return (
    <main className="min-h-screen bg-[#f5f7f9] text-ink">
      <header className="border-b border-line bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-md bg-ink text-white">
              <BrainCircuit size={21} />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-normal">Dental Segmentation</h1>
              <p className="text-sm text-slate-500">Granuloma and kista workflow</p>
            </div>
          </div>
          <ModelStatus models={modelsQuery.data ?? []} />
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-5 py-5">
        <nav className="mb-5 flex w-fit rounded-md border border-line bg-white p-1">
          <TabButton active={activeTab === "inference"} onClick={() => setActiveTab("inference")}>
            <ImageIcon size={16} /> Inference
          </TabButton>
          <TabButton active={activeTab === "testing"} onClick={() => setActiveTab("testing")}>
            <FlaskConical size={16} /> Testing
          </TabButton>
          <TabButton active={activeTab === "training"} onClick={() => setActiveTab("training")}>
            <Activity size={16} /> Training
          </TabButton>
        </nav>

        {activeTab === "inference" && <InferenceView models={modelsQuery.data ?? []} />}
        {activeTab === "testing" && <TestingView models={modelsQuery.data ?? []} />}
        {activeTab === "training" && <TrainingView />}
      </div>
    </main>
  );
}

function TabButton({
  active,
  children,
  onClick
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button className={`tab-button ${active ? "tab-button-active" : ""}`} onClick={onClick}>
      {children}
    </button>
  );
}

function ModelStatus({ models }: { models: ModelInfo[] }) {
  const available = models.filter((model) => model.available).length;
  return (
    <div className="hidden items-center gap-2 text-sm text-slate-600 sm:flex">
      <CheckCircle2 size={17} className="text-teal" />
      <span>
        {available}/{models.length || 2} models available
      </span>
    </div>
  );
}

function InferenceView({ models }: { models: ModelInfo[] }) {
  const [selectedModel, setSelectedModel] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.65);
  const [iou, setIou] = useState(0.7);
  const [result, setResult] = useState<InferenceResult | null>(null);

  const historyQuery = useQuery({ queryKey: ["inference-history"], queryFn: api.inferenceHistory });
  const inferenceMutation = useMutation({
    mutationFn: api.runInference,
    onSuccess: (data) => {
      setResult(data);
      historyQuery.refetch();
    }
  });

  useEffect(() => {
    if (!selectedModel && models.length > 0) {
      setSelectedModel(models.find((model) => model.available)?.id ?? models[0].id);
    }
  }, [models, selectedModel]);

  const canRun = Boolean(file && selectedModel && !inferenceMutation.isPending);

  async function runInference() {
    if (!file || !selectedModel) return;
    const form = new FormData();
    form.append("file", file);
    form.append("model_id", selectedModel);
    form.append("confidence_threshold", String(confidence));
    form.append("iou_threshold", String(iou));
    await inferenceMutation.mutateAsync(form);
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[380px_1fr]">
      <section className="panel">
        <PanelTitle icon={<Upload size={18} />} title="Inference" />
        <div className="field">
          <label>Model</label>
          <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
            {models.map((model) => (
              <option key={model.id} value={model.id} disabled={!model.available}>
                {model.name}
                {model.available ? "" : " (missing files)"}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Radiograph</label>
          <input
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </div>
        <SliderField
          label="Confidence"
          value={confidence}
          min={0}
          max={1}
          step={0.01}
          onChange={setConfidence}
        />
        <SliderField label="IoU" value={iou} min={0} max={1} step={0.01} onChange={setIou} />
        <button className="primary-button mt-2 w-full" disabled={!canRun} onClick={runInference}>
          <Play size={17} /> Run inference
        </button>
        {inferenceMutation.error && <p className="error-text">{inferenceMutation.error.message}</p>}
      </section>

      <section className="panel min-h-[520px]">
        <PanelTitle icon={<ImageIcon size={18} />} title="Result" />
        {result ? <InferenceResultView result={result} /> : <EmptyState />}
      </section>

      <section className="panel lg:col-span-2">
        <div className="mb-3 flex items-center justify-between">
          <PanelTitle icon={<RefreshCw size={18} />} title="Recent Inference" />
          <button className="icon-button" onClick={() => historyQuery.refetch()} title="Refresh">
            <RefreshCw size={16} />
          </button>
        </div>
        <InferenceHistory rows={historyQuery.data ?? []} onSelect={setResult} />
      </section>
    </div>
  );
}

function InferenceResultView({ result }: { result: InferenceResult }) {
  const overlay = mediaUrl(result.overlay_url);
  return (
    <div className="grid gap-5 xl:grid-cols-[1fr_380px]">
      <div className="image-frame">
        {overlay ? <img src={overlay} alt="Prediction overlay" /> : null}
      </div>
      <div>
        <div className="metric-grid">
          <Metric label="Image label" value={result.image_level_label} />
          <Metric label="Detections" value={String(result.detections.length)} />
          <Metric label="Model" value={result.model_id} />
        </div>
        <DetectionTable result={result} />
      </div>
    </div>
  );
}

function DetectionTable({ result }: { result: InferenceResult }) {
  return (
    <div className="table-wrap mt-4">
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Conf.</th>
            <th>Area</th>
            <th>Box</th>
          </tr>
        </thead>
        <tbody>
          {result.detections.map((detection, index) => (
            <tr key={`${detection.class_name}-${index}`}>
              <td>
                <span className={`class-dot ${detection.class_name}`} />
                {detection.class_name}
              </td>
              <td>{detection.confidence.toFixed(2)}</td>
              <td>{Math.round(detection.mask_area)}</td>
              <td>
                {Math.round(detection.bbox.width)} x {Math.round(detection.bbox.height)}
              </td>
            </tr>
          ))}
          {result.detections.length === 0 && (
            <tr>
              <td colSpan={4}>Normal</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function InferenceHistory({
  rows,
  onSelect
}: {
  rows: InferenceResult[];
  onSelect: (result: InferenceResult) => void;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Model</th>
            <th>Label</th>
            <th>Detections</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} onClick={() => onSelect(row)} className="click-row">
              <td>{new Date(row.created_at).toLocaleString()}</td>
              <td>{row.model_id}</td>
              <td>{row.image_level_label}</td>
              <td>{row.detections.length}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={4}>No inference history yet</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function TestingView({ models }: { models: ModelInfo[] }) {
  const [modelId, setModelId] = useState("");
  const [datasetKey, setDatasetKey] = useState("");
  const [limit, setLimit] = useState<number | "">("");
  const datasetsQuery = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const jobsQuery = useQuery({
    queryKey: ["testing-jobs"],
    queryFn: api.testingJobs,
    refetchInterval: 4000
  });
  const mutation = useMutation({
    mutationFn: api.createTestingJob,
    onSuccess: () => jobsQuery.refetch()
  });

  useEffect(() => {
    if (!modelId && models.length > 0) setModelId(models.find((model) => model.available)?.id ?? models[0].id);
  }, [modelId, models]);
  useEffect(() => {
    const datasets = datasetsQuery.data ?? [];
    if (!datasetKey && datasets.length > 0) {
      setDatasetKey(datasets.find((dataset) => dataset.available)?.key ?? datasets[0].key);
    }
  }, [datasetKey, datasetsQuery.data]);

  async function runTesting() {
    if (!modelId || !datasetKey) return;
    await mutation.mutateAsync({
      model_id: modelId,
      dataset_key: datasetKey,
      limit: limit === "" ? null : limit
    });
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
      <section className="panel">
        <PanelTitle icon={<FlaskConical size={18} />} title="Testing Job" />
        <div className="field">
          <label>Model</label>
          <select value={modelId} onChange={(event) => setModelId(event.target.value)}>
            {models.map((model) => (
              <option key={model.id} value={model.id} disabled={!model.available}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Dataset</label>
          <select value={datasetKey} onChange={(event) => setDatasetKey(event.target.value)}>
            {(datasetsQuery.data ?? []).map((dataset) => (
              <option key={dataset.key} value={dataset.key} disabled={!dataset.available}>
                {dataset.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Limit</label>
          <input
            type="number"
            min={1}
            max={500}
            value={limit}
            onChange={(event) =>
              setLimit(event.target.value === "" ? "" : Number(event.target.value))
            }
          />
        </div>
        <button className="primary-button w-full" onClick={runTesting} disabled={mutation.isPending}>
          <Play size={17} /> Run test
        </button>
        {mutation.error && <p className="error-text">{mutation.error.message}</p>}
      </section>

      <section className="panel">
        <PanelTitle icon={<BarChart3 size={18} />} title="Testing Results" />
        <JobList jobs={jobsQuery.data ?? []} />
      </section>
    </div>
  );
}

function TrainingView() {
  const [modelFamily, setModelFamily] = useState<"yolo" | "unet_inception">("yolo");
  const [epochs, setEpochs] = useState(50);
  const [imageSize, setImageSize] = useState(512);
  const [batchSize, setBatchSize] = useState(16);
  const jobsQuery = useQuery({
    queryKey: ["training-jobs"],
    queryFn: api.trainingJobs,
    refetchInterval: 4000
  });
  const createMutation = useMutation({
    mutationFn: api.createTrainingJob,
    onSuccess: () => jobsQuery.refetch()
  });
  const promoteMutation = useMutation({
    mutationFn: api.promoteTrainingJob,
    onSuccess: () => jobsQuery.refetch()
  });

  async function runTraining() {
    await createMutation.mutateAsync({
      model_family: modelFamily,
      epochs,
      image_size: imageSize,
      batch_size: batchSize
    });
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
      <section className="panel">
        <PanelTitle icon={<Activity size={18} />} title="Training Job" />
        <div className="field">
          <label>Family</label>
          <select
            value={modelFamily}
            onChange={(event) =>
              setModelFamily(event.target.value as "yolo" | "unet_inception")
            }
          >
            <option value="yolo">YOLOv11</option>
            <option value="unet_inception">U-Net + Inception</option>
          </select>
        </div>
        <div className="field">
          <label>Epochs</label>
          <input type="number" min={1} max={1000} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} />
        </div>
        <div className="field">
          <label>Image size</label>
          <input type="number" min={128} max={2048} value={imageSize} onChange={(event) => setImageSize(Number(event.target.value))} />
        </div>
        <div className="field">
          <label>Batch</label>
          <input type="number" min={-1} max={256} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
        </div>
        <button className="primary-button w-full" onClick={runTraining} disabled={createMutation.isPending}>
          <Play size={17} /> Start training
        </button>
        {createMutation.error && <p className="error-text">{createMutation.error.message}</p>}
      </section>

      <section className="panel">
        <PanelTitle icon={<Activity size={18} />} title="Training Runs" />
        <TrainingJobList
          jobs={jobsQuery.data ?? []}
          onPromote={(job) => promoteMutation.mutate(job.id)}
        />
      </section>
    </div>
  );
}

function JobList({ jobs }: { jobs: EvaluationJob[] }) {
  return (
    <div className="space-y-4">
      {jobs.map((job) => (
        <div className="run-row" key={job.id}>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge status={job.status} />
              <strong>{job.model_id}</strong>
              <span className="text-sm text-slate-500">{job.dataset_key}</span>
            </div>
            {job.error && <p className="error-text">{job.error}</p>}
          </div>
          <MetricsPreview job={job} />
        </div>
      ))}
      {jobs.length === 0 && <EmptyState />}
    </div>
  );
}

function MetricsPreview({ job }: { job: EvaluationJob }) {
  const overall = job.metrics?.image?.overall;
  const pixel = job.metrics?.pixel;
  if (!overall && !pixel) return <span className="text-sm text-slate-500">{job.status}</span>;
  return (
    <div className="metric-grid min-w-[320px]">
      <Metric label="Accuracy" value={overall?.accuracy ?? "-"} />
      <Metric label="Dice" value={pixel?.dice ?? "-"} />
      <Metric label="Samples" value={job.metrics?.samples ?? "-"} />
    </div>
  );
}

function TrainingJobList({
  jobs,
  onPromote
}: {
  jobs: TrainingJob[];
  onPromote: (job: TrainingJob) => void;
}) {
  return (
    <div className="space-y-4">
      {jobs.map((job) => (
        <div className="run-row" key={job.id}>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge status={job.status} />
              <strong>{job.model_family}</strong>
              <span className="text-sm text-slate-500">{job.id.slice(0, 8)}</span>
            </div>
            {job.artifacts?.log && <p className="text-sm text-slate-500">{job.artifacts.log}</p>}
            {job.error && <p className="error-text">{job.error}</p>}
          </div>
          <div className="flex items-center gap-2">
            {job.promoted_model_id && <span className="badge badge-ok">{job.promoted_model_id}</span>}
            {job.status === "completed" && job.artifacts?.best_model && !job.promoted_model_id && (
              <button className="secondary-button" onClick={() => onPromote(job)}>
                Promote
              </button>
            )}
          </div>
        </div>
      ))}
      {jobs.length === 0 && <EmptyState />}
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="field">
      <label>
        {label} <span>{value.toFixed(2)}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="mb-4 flex items-center gap-2">
      <span className="text-slate-500">{icon}</span>
      <h2 className="text-base font-semibold">{title}</h2>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const ok = status === "completed";
  const failed = status === "failed";
  return <span className={`badge ${ok ? "badge-ok" : failed ? "badge-fail" : ""}`}>{status}</span>;
}

function EmptyState() {
  return (
    <div className="empty-state">
      <ImageIcon size={28} />
      <span>No result selected</span>
    </div>
  );
}
