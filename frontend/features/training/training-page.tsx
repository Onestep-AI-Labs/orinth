"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Activity, Cpu, Database, Lock, Play, SlidersHorizontal, Upload } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { TrainingJobTable } from "@/features/training/training-components";
import { AdvancedSettings, advancedDefaults, type AdvancedValues } from "@/features/training/advanced-settings";
import { allowedTaskTypesForProject, compactNumber, formatBytes, isLlmTask, isNlpTask, listPollInterval } from "@/features/platform/utils";
import { Badge, Button, ButtonLink, CardGridSkeleton, EmptyState, Field, HistoryHeader, MutationError, NumberInput, PageHeader, PanelTitle, TableSkeleton, TaskSelect, useConfirmationDialog } from "@/features/platform/ui";
import type { TaskType, TrainingJob } from "@/types/api";

// Phase 17: the architecture studio's Train button deep-links here with the
// graph to train. The id rides the open-ended `hyperparameters` dict, matching
// how the backend reads it — see `app/ml/architecture/train_catalog.py`.
const ARCHITECTURE_OPTION_ID = "architecture_graph";

export function TrainingPage() {
  const { projectId, project } = useProject();
  const searchParams = useSearchParams();
  const linkedArchitectureId = searchParams.get("architecture_id") ?? "";
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [modelOptionId, setModelOptionId] = useState("");
  const [architectureId, setArchitectureId] = useState("");
  const [modelName, setModelName] = useState("");
  const [customHfId, setCustomHfId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [epochs, setEpochs] = useState(50);
  const [imageSize, setImageSize] = useState(512);
  const [batchSize, setBatchSize] = useState(16);
  const [optimizer, setOptimizer] = useState("AdamW");
  const [learningRate, setLearningRate] = useState(0.002);
  const [maxLength, setMaxLength] = useState(160);
  const [targetMaxLength, setTargetMaxLength] = useState(64);
  const [vocabSize, setVocabSize] = useState(12000);
  const [device, setDevice] = useState("");
  const [finetuneMethod, setFinetuneMethod] = useState("lora");
  const [advancedValues, setAdvancedValues] = useState<AdvancedValues>({});
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const datasetsQuery = useQuery({ queryKey: ["training-datasets", projectId], queryFn: () => api.datasetCatalog(projectId) });
  const optionsQuery = useQuery({
    queryKey: ["training-options", taskType],
    queryFn: () => api.trainingOptions(taskType)
  });
  const jobsQuery = useQuery({
    queryKey: ["training-jobs", projectId],
    queryFn: () => api.trainingJobs(projectId),
    refetchInterval: (query) => listPollInterval(query.state.data as TrainingJob[] | undefined)
  });
  const router = useRouter();
  const createMutation = useMutation({
    mutationFn: api.createTrainingJob,
    onSuccess: async (job, variables) => {
      setModelName("");
      // LLM runs go straight to the live training detail view (the platform's
      // equivalent of Studio switching to its Current Run tab on start).
      if (variables.task_type === "llm_finetune") {
        router.push(`/training/${job.id}`);
        return;
      }
      await jobsQuery.refetch();
    }
  });
  const prepareMutation = useMutation({
    mutationFn: () => api.prepareModelAsset(modelOptionId, true, customHf ? customHfId.trim() : undefined)
  });
  const deleteMutation = useMutation({
    mutationFn: ({ ids, clearAll }: { ids: string[]; clearAll?: boolean }) =>
      api.deleteTrainingJobs(ids, projectId, clearAll),
    onSuccess: async () => {
      setSelectedIds([]);
      await jobsQuery.refetch();
    }
  });
  const options = useMemo(() => optionsQuery.data ?? [], [optionsQuery.data]);
  const option = options.find((item) => item.id === modelOptionId);
  const isArchitectureOption = modelOptionId === ARCHITECTURE_OPTION_ID;
  const architecturesQuery = useQuery({
    queryKey: ["architectures", projectId],
    queryFn: () => api.architectures(projectId),
    enabled: isArchitectureOption
  });
  const architectures = architecturesQuery.data ?? [];
  // Starting a graph run without a graph fails server-side; block it here
  // so the user sees why before submitting.
  const architectureMissing = isArchitectureOption && !architectureId;
  const taskOptions = useMemo(() => allowedTaskTypesForProject(project), [project]);
  const nlp = isNlpTask(taskType);
  const llm = isLlmTask(taskType);
  // What a run would actually do on this machine (accelerator, and for LLM the
  // Unsloth-vs-PEFT backend). Fetched whenever a device choice is relevant —
  // vision and LLM — but never for NLP, which has no device control.
  const environmentQuery = useQuery({
    queryKey: ["llm-environment"],
    queryFn: api.llmEnvironment,
    enabled: !nlp
  });
  const detectedDevice = environmentQuery.data?.device;
  const customHf = llm && Boolean(option?.defaults.custom_hf);
  const customHfMissing = customHf && !customHfId.trim();
  // Accurate base-model detail from the Hugging Face API (real size, files,
  // gating). The ref is the catalog/local option id (backend resolves it) or,
  // for the custom option, the typed hub id — debounced so typing doesn't spam
  // the Hub.
  const baseRef = customHf ? customHfId.trim() : llm ? option?.id : undefined;
  const [debouncedRef, setDebouncedRef] = useState<string | undefined>(undefined);
  useEffect(() => {
    if (!llm || !baseRef) {
      setDebouncedRef(undefined);
      return;
    }
    const handle = setTimeout(() => setDebouncedRef(baseRef), customHf ? 600 : 0);
    return () => clearTimeout(handle);
  }, [llm, baseRef, customHf]);
  const modelInfoQuery = useQuery({
    queryKey: ["llm-model-info", debouncedRef],
    queryFn: () => api.llmModelInfo(debouncedRef as string),
    enabled: llm && Boolean(debouncedRef)
  });
  const modelInfo = modelInfoQuery.data;
  // Prefer the live HF gating status; fall back to the catalog's static flag.
  const gatedLicense = llm && (Boolean(modelInfo?.gated) || Boolean(option?.defaults.gated_license));
  const nlpBaseline = nlp && option?.source === "local" && ["nlp_tfidf_classifier", "nlp_extractive_summarizer", "nlp_keyword_qa"].includes(option.id);
  const nlpNeural = nlp && !nlpBaseline;
  const showBatch = !nlp || nlpNeural;
  const showOptimization = !nlp || nlpNeural || option?.id === "nlp_tfidf_classifier";
  const showMaxLength = nlpNeural || Boolean(option?.defaults.max_length);
  const showTargetMaxLength = nlp && taskType === "summarization" && Boolean(option?.defaults.target_max_length);
  const showVocabSize =
    nlp &&
    option?.source === "local" &&
    (option.family === "nlp_keras_seq2seq" || Boolean(option.family?.startsWith("nlp_keras_")));
  const datasets = useMemo(
    () => (datasetsQuery.data ?? []).filter((dataset) => dataset.task_type === taskType),
    [datasetsQuery.data, taskType]
  );

  useEffect(() => {
    if (!taskOptions.includes(taskType)) {
      setTaskType(taskOptions[0] ?? "classification");
    }
  }, [taskOptions, taskType]);

  useEffect(() => {
    // Arriving from the studio picks the option for the user: they already
    // chose the architecture, so making them find it again is friction.
    if (!linkedArchitectureId) return;
    setArchitectureId(linkedArchitectureId);
    setTaskType("classification");
    setModelOptionId(ARCHITECTURE_OPTION_ID);
  }, [linkedArchitectureId]);

  useEffect(() => {
    // Clear a model that this task does not offer, but never pick one. The
    // base model determines the hyperparameter defaults below, so an
    // auto-selected one silently decides how the run is configured.
    if (modelOptionId && !options.some((item) => item.id === modelOptionId)) {
      setModelOptionId("");
    }
  }, [modelOptionId, options]);

  useEffect(() => {
    if (!option) return;
    setEpochs((value) => Number(option.defaults.epochs ?? value));
    setImageSize((value) => Number(option.defaults.image_size ?? value));
    setBatchSize((value) => Number(option.defaults.batch_size ?? value));
    setOptimizer((value) => String(option.defaults.optimizer ?? value));
    setLearningRate((value) => Number(option.defaults.learning_rate ?? value));
    setMaxLength((value) => Number(option.defaults.max_length ?? value));
    setTargetMaxLength((value) => Number(option.defaults.target_max_length ?? value));
    setVocabSize((value) => Number(option.defaults.vocab_size ?? value));
  }, [option]);

  // Rehydrate the advanced accordion from the selected option's catalog
  // defaults, discarding edits — the same behaviour the basic fields have. An
  // empty set (or no option) clears it, so the accordion never carries a stale
  // family's fields.
  //
  // For LLM runs the 4-bit/precision defaults are auto-detected from the
  // device: 4-bit QLoRA and bf16 only apply on CUDA, so on MPS/CPU the shown
  // defaults become "off"/fp32 — matching what the run will actually do instead
  // of showing knobs the runner would silently downgrade.
  useEffect(() => {
    const defaults = advancedDefaults(option?.advanced_parameters ?? []);
    if (llm && detectedDevice && detectedDevice !== "cuda") {
      if ("load_in_4bit" in defaults) defaults.load_in_4bit = false;
      if ("precision" in defaults) defaults.precision = "fp32";
    }
    setAdvancedValues(defaults);
  }, [option, llm, detectedDevice]);

  // Transformer fine-tuning lives in a narrow learning-rate band. At 1e-3 the
  // updates are large enough to wreck the pretrained weights, which shows up as
  // a validation curve that oscillates and climbs while training loss falls.
  // LoRA tolerates larger rates than full fine-tuning, so LLM runs get their
  // own 1e-3 threshold — the BERT-era 1e-4 rule would flag every default run.
  const recommendedLearningRate =
    option?.defaults.learning_rate !== undefined ? Number(option.defaults.learning_rate) : null;
  const learningRateWarning = llm
    ? learningRate > 0.001
      ? `${learningRate} is above the LoRA fine-tuning band; use ${recommendedLearningRate ?? 0.0002} or lower.`
      : null
    : option?.source === "huggingface" && learningRate > 0.0001
      ? `${learningRate} is far too high for fine-tuning; use ${recommendedLearningRate ?? 0.00005} or lower.`
      : null;
  const showRecommendedLearningRate =
    recommendedLearningRate !== null && learningRate !== recommendedLearningRate;

  useEffect(() => {
    if (datasets.length === 0) {
      setDatasetId("");
      return;
    }
    if (!datasets.some((dataset) => dataset.id === datasetId)) setDatasetId(datasets[0].id);
  }, [datasetId, datasets]);

  async function runTraining() {
    // No falling back to options[0]: training against a model the user did not
    // choose produces a run whose configuration they cannot account for.
    const selectedOption = option;
    if (!selectedOption || !datasetId) return;
    // A custom Hugging Face base needs its hub id before the run can resolve one.
    if (customHf && !customHfId.trim()) return;
    // Advanced values ride the same open-ended dict as the existing NLP knobs;
    // every catalog default is submitted so the run is reproducible from the
    // job record alone. Runners allowlist-filter these, so stale keys are safe.
    const hyperparameters: Record<string, unknown> = { ...advancedValues };
    if (nlp && showMaxLength) hyperparameters.max_length = maxLength;
    if (nlp && showTargetMaxLength) hyperparameters.target_max_length = targetMaxLength;
    if (nlp && showVocabSize) hyperparameters.vocab_size = vocabSize;
    // Fine-tuning method is chosen via its own prominent select, not the
    // advanced accordion; it rides the same hyperparameters dict.
    if (llm) hyperparameters.finetune_method = finetuneMethod;
    if (selectedOption.id === ARCHITECTURE_OPTION_ID) hyperparameters.architecture_id = architectureId;
    await createMutation.mutateAsync({
      project_id: projectId,
      task_type: taskType,
      model_family: selectedOption.family,
      model_option_id: selectedOption.id,
      model_name: modelName.trim() || null,
      ...(customHf ? { base_model: customHfId.trim() } : {}),
      epochs,
      ...(nlp || llm ? {} : { image_size: imageSize }),
      batch_size: batchSize,
      dataset_id: datasetId,
      optimizer,
      learning_rate: learningRate,
      hyperparameters,
      ...(nlp || llm
        ? {}
        : {
            device,
            cache: "disk" as const,
            workers: 0,
            patience: 50
          })
    });
  }

  function confirmDeleteTrainingRows() {
    if (selectedIds.length === 0) return;
    confirm({
      title: "Delete training jobs?",
      message: `This will delete ${selectedIds.length} selected terminal training job${selectedIds.length === 1 ? "" : "s"} and owned artifacts.`,
      confirmLabel: "Delete jobs",
      onConfirm: () => deleteMutation.mutate({ ids: selectedIds })
    });
  }

  function confirmClearTrainingJobs() {
    confirm({
      title: "Clear all training jobs?",
      message: "This will delete all terminal training jobs for the active project and owned artifacts.",
      confirmLabel: "Clear all",
      onConfirm: () => deleteMutation.mutate({ ids: [], clearAll: true })
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Training" subtitle="Configure model runs" icon={<Activity size={20} />} />
      {/* Configure anatomy: a full-width Model header over a config rail
          (Dataset + Run) beside the wider Parameters panel, collapsing to one
          column when narrow. No field changes ownership between basic and
          advanced — this is a layout regrouping only. */}
      <div className="training-config">
        <section className="panel training-model-section">
          <PanelTitle icon={<Activity size={18} />} title="Model" dataTour="training-model" />
          {(optionsQuery.isLoading || datasetsQuery.isLoading) && <CardGridSkeleton count={1} />}
          <div className="form-grid form-grid-three">
            <Field label="Task">
              <TaskSelect value={taskType} onChange={setTaskType} options={taskOptions} />
            </Field>
            <Field label="Base model">
              <select value={modelOptionId} onChange={(event) => setModelOptionId(event.target.value)}>
                <option value="">Choose a model…</option>
                {llm ? (
                  <>
                    <optgroup label="Hub models">
                      {options.filter((item) => item.source !== "local").map((item) => (
                        <option key={item.id} value={item.id} disabled={!item.runnable}>
                          {item.name}
                          {item.defaults.gated_license ? " — license required" : ""}
                          {item.runnable ? "" : " (gated)"}
                        </option>
                      ))}
                    </optgroup>
                    {options.some((item) => item.source === "local") && (
                      <optgroup label="Local base models">
                        {options.filter((item) => item.source === "local").map((item) => (
                          <option key={item.id} value={item.id} disabled={!item.runnable}>
                            {item.name}
                            {item.runnable ? "" : " (gated)"}
                          </option>
                        ))}
                      </optgroup>
                    )}
                  </>
                ) : (
                  options.map((item) => (
                    <option key={item.id} value={item.id} disabled={!item.runnable}>
                      {item.name}
                      {item.runnable ? "" : " (gated)"}
                    </option>
                  ))
                )}
              </select>
            </Field>
            <Field label="Model name" hint="optional">
              <input value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="Display name" />
            </Field>
          </div>
          {option && <p className="form-caption">{option.description}</p>}
          {isArchitectureOption && (
            <Field
              label="Architecture"
              hint={architectures.length === 0 ? "none saved yet" : "from the studio"}
            >
              {architectures.length === 0 ? (
                <EmptyState
                  label="No saved architectures"
                  description="Build one on the canvas first, then come back to train it."
                  action={
                    <ButtonLink variant="secondary" href="/models/architectures">
                      Open the studio
                    </ButtonLink>
                  }
                />
              ) : (
                <select
                  value={architectureId}
                  onChange={(event) => setArchitectureId(event.target.value)}
                >
                  <option value="">Choose an architecture…</option>
                  {architectures.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name} — v{item.version}, {item.node_count} nodes
                    </option>
                  ))}
                </select>
              )}
            </Field>
          )}
          {customHf && (
            <Field label="Hugging Face model id" hint="Transformers checkpoint">
              <input
                value={customHfId}
                onChange={(event) => setCustomHfId(event.target.value)}
                placeholder="e.g. Qwen/Qwen3.5-1.8B-Instruct"
              />
            </Field>
          )}
          {gatedLicense && (
            <Badge tone="warn">
              <Lock size={12} /> Gated license — accept it on huggingface.co before preparing
            </Badge>
          )}
          {llm && (
            <p className="form-caption">
              Trainable base: a Hugging Face Transformers model (safetensors/PyTorch + config.json +
              tokenizer), from the hub or a local <code>llm_hf</code> upload. GGUF and TFLite are
              inference/export formats and can’t be used as a training base.
            </p>
          )}
          {llm && debouncedRef && (
            <p className="form-caption model-detail-line">
              {modelInfoQuery.isFetching && !modelInfo ? (
                "Checking model details on Hugging Face…"
              ) : modelInfo?.exists ? (
                <>
                  {modelInfo.size_bytes != null && (
                    <span>Download size {formatBytes(modelInfo.size_bytes)}</span>
                  )}
                  {modelInfo.file_count != null && <span>{modelInfo.file_count} files</span>}
                  {modelInfo.downloads != null && (
                    <span>{compactNumber(modelInfo.downloads)} downloads</span>
                  )}
                  {modelInfo.gated && <span className="model-detail-gated">license required</span>}
                </>
              ) : modelInfo ? (
                <span className="model-detail-error">{modelInfo.error}</span>
              ) : null}
            </p>
          )}
          {llm && environmentQuery.data && (
            <div className="llm-env-banner">
              <span className="llm-env-banner-headline">
                <Cpu size={14} /> Device {environmentQuery.data.device.toUpperCase()} · backend{" "}
                {environmentQuery.data.recommended_backend}
              </span>
              {environmentQuery.data.notes.map((note) => (
                <p key={note}>{note}</p>
              ))}
            </div>
          )}
        </section>
        <div className="training-config-grid">
          <div className="training-config-rail">
            <section className="panel">
              <PanelTitle icon={<Database size={18} />} title="Dataset" dataTour="training-dataset" />
              {datasetsQuery.isLoading ? (
                <CardGridSkeleton count={1} />
              ) : datasets.length === 0 ? (
                <EmptyState
                  icon={<Database size={26} />}
                  label="No dataset for this task"
                  description="Create or import a dataset before starting a run."
                  action={
                    <ButtonLink variant="secondary" size="sm" href="/datasets">
                      Go to datasets
                    </ButtonLink>
                  }
                />
              ) : (
                <Field label="Dataset" hint={`${datasets.length} available`}>
                  <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
                    {datasets.map((dataset) => (
                      <option value={dataset.id} key={dataset.id}>
                        {dataset.name}
                      </option>
                    ))}
                  </select>
                </Field>
              )}
            </section>
            <section className="panel training-run-card">
              <PanelTitle icon={<Play size={18} />} title="Run" dataTour="training-run" />
              {!nlp && !llm && (
                <Field label="Device" hint="auto-detected">
                  {/* Auto-detected accelerator, offered as a dropdown rather than
                      free text so an invalid device string can't reach training.
                      "Auto" lets the runner pick cuda > mps > cpu. */}
                  <select value={device} onChange={(event) => setDevice(event.target.value)}>
                    <option value="">
                      Auto{detectedDevice ? ` — ${detectedDevice.toUpperCase()}` : ""}
                    </option>
                    {detectedDevice === "cuda" && <option value="cuda:0">CUDA (cuda:0)</option>}
                    {detectedDevice === "mps" && <option value="mps">MPS (Apple GPU)</option>}
                    <option value="cpu">CPU</option>
                  </select>
                </Field>
              )}
              <div className="action-row action-row-split">
                <Button
                  variant="secondary"
                  onClick={() => prepareMutation.mutate()}
                  disabled={!option?.needs_download || prepareMutation.isPending || customHfMissing}
                >
                  <Upload size={16} /> Prepare
                </Button>
                <Button
                  variant="primary"
                  className="flex-1"
                  onClick={runTraining}
                  disabled={createMutation.isPending || !option?.runnable || !datasetId || customHfMissing || architectureMissing}
                >
                  <Play size={17} /> Start training
                </Button>
              </div>
              {prepareMutation.data && (
                <p className="form-caption">
                  {prepareMutation.data.status}: {prepareMutation.data.message ?? prepareMutation.data.path}
                </p>
              )}
              <MutationError mutations={[createMutation, prepareMutation]} />
            </section>
          </div>
          <section className="panel training-parameters-card">
            <PanelTitle icon={<SlidersHorizontal size={18} />} title="Parameters" dataTour="training-parameters" />
            {/* One grid for every numeric parameter: the visible set changes with
                task and model, and separate per-row grids left ragged half-width
                and full-width fields stacked against each other. */}
            {llm && (
              <div className="form-grid">
                <Field label="Fine-tuning method">
                  <select value={finetuneMethod} onChange={(event) => setFinetuneMethod(event.target.value)}>
                    <option value="lora">LoRA adapter — light, recommended</option>
                    <option value="qlora">QLoRA (4-bit) — least VRAM (CUDA)</option>
                    <option value="full">Full fine-tune — updates every weight (heavy)</option>
                    <option value="continued_pretrain">Continued pretraining — LoRA on full text</option>
                  </select>
                  {finetuneMethod === "full" && (
                    <span className="field-hint field-hint-start">
                      Full fine-tuning needs far more memory; on a Mac/CPU prefer LoRA or QLoRA.
                    </span>
                  )}
                  {finetuneMethod === "qlora" && detectedDevice !== "cuda" && (
                    <span className="field-hint field-hint-start">
                      4-bit QLoRA needs CUDA; on {(detectedDevice ?? "this device").toUpperCase()} it trains as plain LoRA.
                    </span>
                  )}
                </Field>
              </div>
            )}
            <div className="form-grid form-grid-two">
              <Field label="Epochs">
                <NumberInput min={1} max={1000} value={epochs} onChange={setEpochs} />
              </Field>
              {!nlp && !llm && (
                <Field label="Image size">
                  <NumberInput min={128} max={2048} value={imageSize} onChange={setImageSize} />
                </Field>
              )}
              {showBatch && (
                <Field label="Batch size">
                  <NumberInput min={1} max={256} value={batchSize} onChange={setBatchSize} />
                </Field>
              )}
              {showMaxLength && (
                <Field label="Max length">
                  <NumberInput min={8} max={2048} value={maxLength} onChange={setMaxLength} />
                </Field>
              )}
              {showTargetMaxLength && (
                <Field label="Target length">
                  <NumberInput min={8} max={512} value={targetMaxLength} onChange={setTargetMaxLength} />
                </Field>
              )}
              {showVocabSize && (
                <Field label="Vocab size">
                  <NumberInput min={100} max={100000} value={vocabSize} onChange={setVocabSize} />
                </Field>
              )}
            </div>
            {showOptimization && (
              <div className="form-grid form-grid-two">
                {/* LLM runs are AdamW-only (the SFT runner owns the optimizer);
                    a select with one working value would be decoration. */}
                {!llm && (
                  <Field label="Optimizer">
                    <select value={optimizer} onChange={(event) => setOptimizer(event.target.value)}>
                      <option value="AdamW">AdamW</option>
                      <option value="adam">Adam</option>
                      <option value="sgd">SGD</option>
                      <option value="liblinear">Liblinear</option>
                      <option value="keyword">Keyword</option>
                    </select>
                  </Field>
                )}
                <Field label={option?.id === "nlp_tfidf_classifier" ? "Regularisation (C)" : "Learning rate"}>
                  <NumberInput
                    min={0}
                    step={option?.source === "huggingface" ? 0.00001 : 0.0001}
                    value={learningRate}
                    onChange={setLearningRate}
                  />
                  {/* Silent at the default, since the field is already filled with
                      the recommended value — restating it there is noise that only
                      forces the label to wrap. Speaks up once you deviate. */}
                  {learningRateWarning ? (
                    <span className="field-warning">{learningRateWarning}</span>
                  ) : showRecommendedLearningRate ? (
                    <span className="field-hint field-hint-start">Recommended {recommendedLearningRate}</span>
                  ) : null}
                </Field>
              </div>
            )}
            <AdvancedSettings
              params={(option?.advanced_parameters ?? []).filter((spec) => spec.key !== "finetune_method")}
              values={advancedValues}
              onChange={(key, value) => setAdvancedValues((current) => ({ ...current, [key]: value }))}
            />
          </section>
        </div>
        <section className="panel">
          <HistoryHeader
            title="Training Jobs"
            selectedCount={selectedIds.length}
            onRefresh={() => jobsQuery.refetch()}
            onDelete={confirmDeleteTrainingRows}
            onClear={confirmClearTrainingJobs}
            dataTour="training-jobs"
          />
          <TrainingJobTable jobs={jobsQuery.data ?? []} selectedIds={selectedIds} setSelectedIds={setSelectedIds} />
          {jobsQuery.isLoading && <TableSkeleton rows={5} />}
          <MutationError mutations={[deleteMutation]} />
        </section>
      </div>
      {confirmationDialog}
    </div>
  );
}
