"use client";

/* eslint-disable @next/next/no-img-element */

import { useCallback, useEffect, useState } from "react";
import { BarChart3, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, Copy, Database, FilePlus2, ImageIcon, Trash2, X } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { api, apiAssetUrl } from "@/lib/api";
import { SPLIT_FILTERS, SPLITS, TRAINING_SPLITS } from "@/features/platform/constants";
import { formatDatasetFormat, formatDatasetTask, isLlmTask, isNlpTask, labelColor, labelFill, pointsAttr, taskDescription } from "@/features/platform/utils";
import { CardGridSkeleton, EmptyState, Field, InlineSpinner, Metric, MutationError, PanelTitle, Select, StatusBadge, useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetEdaSummary, DatasetFormat, DatasetItemSummary, DatasetPreprocessConfig, DatasetSplitConfig, DatasetSplitFilter, DatasetSummary, DatasetVersionSummary, SplitKey, TaskType } from "@/types/api";

export function DatasetCreatePanel({
  newDatasetName,
  setNewDatasetName,
  taskType,
  setTaskType,
  allowedTaskTypes,
  labelDraft,
  setLabelDraft,
  onCreate,
  pending
}: {
  newDatasetName: string;
  setNewDatasetName: (value: string) => void;
  taskType: TaskType;
  setTaskType: (value: TaskType) => void;
  allowedTaskTypes: TaskType[];
  labelDraft: string;
  setLabelDraft: (value: string) => void;
  onCreate: (override?: { format?: DatasetFormat }) => void;
  pending: boolean;
}) {
  const labels = labelDraft
    .split(",")
    .map((label) => label.trim())
    .filter(Boolean);
  const visibleLabels = labels;
  const [labelInput, setLabelInput] = useState("");
  // Instruction | chat is chosen here because the task alone does not fix the
  // record shape for LLM fine-tuning (the phase-10 exception to derived format).
  const [llmSchema, setLlmSchema] = useState<DatasetFormat>("instruction_jsonl");
  // Domain is picked before the task list, so a mixed project shows only the
  // relevant choices instead of every task spanning unrelated kinds of data.
  const [domain, setDomain] = useState<"vision" | "nlp" | "llm" | null>(null);
  const visionChoices = allowedTaskTypes.filter((task) => !isNlpTask(task) && !isLlmTask(task));
  const nlpChoices = allowedTaskTypes.filter((task) => isNlpTask(task));
  const llmChoices = allowedTaskTypes.filter((task) => isLlmTask(task));
  const domainChoices =
    domain === "vision" ? visionChoices : domain === "nlp" ? nlpChoices : domain === "llm" ? llmChoices : [];
  const llm = isLlmTask(taskType);
  const nlp = isNlpTask(taskType);
  const format: DatasetFormat = llm
    ? llmSchema
    : nlp
      ? "text_folder"
      : taskType === "classification"
        ? "image_folder"
        : "yolo";
  const chosen = domain !== null && domainChoices.includes(taskType);
  // Summarization/QA carry their content in the annotation itself and LLM
  // records carry their own supervision; their label list is fixed or empty.
  const needsLabels =
    taskType !== "summarization" && taskType !== "question_answering" && !llm;
  const activeDomains = [visionChoices, nlpChoices, llmChoices].filter((choices) => choices.length > 0).length;
  const multiDomain = activeDomains > 1;

  const pickDomain = useCallback(
    (next: "vision" | "nlp" | "llm") => {
      setDomain(next);
      // Park the task on the new domain's first option so the derived format
      // and label fields below never describe the domain the user just left.
      const choices = next === "vision" ? visionChoices : next === "nlp" ? nlpChoices : llmChoices;
      if (choices.length && !choices.includes(taskType)) setTaskType(choices[0]);
    },
    [llmChoices, nlpChoices, setTaskType, taskType, visionChoices]
  );

  useEffect(() => {
    // A single-domain project has no choice to make, so skip straight to its
    // tasks rather than making the user click a lone tab.
    if (domain !== null || multiDomain) return;
    if (visionChoices.length > 0) pickDomain("vision");
    else if (nlpChoices.length > 0) pickDomain("nlp");
    else if (llmChoices.length > 0) pickDomain("llm");
  }, [domain, llmChoices.length, multiDomain, nlpChoices.length, pickDomain, visionChoices.length]);

  function syncLabels(nextLabels: string[]) {
    setLabelDraft(nextLabels.join(", "));
  }

  function addLabels() {
    // Accept either separator so pasting a list from elsewhere works without
    // the user having to know which one this field wants.
    const next = labelInput
      .split(/[;,]/)
      .map((label) => label.trim())
      .filter(Boolean);
    if (next.length === 0) return;
    const merged = [...visibleLabels];
    next.forEach((label) => {
      if (!merged.some((item) => item.toLowerCase() === label.toLowerCase())) {
        merged.push(label);
      }
    });
    syncLabels(merged);
    setLabelInput("");
  }

  function removeLabel(index: number) {
    syncLabels(visibleLabels.filter((_label, labelIndex) => labelIndex !== index));
  }

  return (
    <div className="create-panel">
      <Field label="Dataset name">
        <input value={newDatasetName} onChange={(event) => setNewDatasetName(event.target.value)} placeholder="Name this dataset" />
      </Field>
      <div className="field">
        <label>Dataset type</label>
        {allowedTaskTypes.length === 0 ? (
          <CardGridSkeleton count={3} />
        ) : (
          <>
            {/* Only the domains this project declared — a vision-only project
                has no use for an NLP tab it can never choose. */}
            {multiDomain ? (
              <div className="segmented-control mb-3">
                {visionChoices.length > 0 ? (
                  <button
                    type="button"
                    className={domain === "vision" ? "segmented-active" : ""}
                    onClick={() => pickDomain("vision")}
                  >
                    Vision
                  </button>
                ) : null}
                {nlpChoices.length > 0 ? (
                  <button
                    type="button"
                    className={domain === "nlp" ? "segmented-active" : ""}
                    onClick={() => pickDomain("nlp")}
                  >
                    NLP
                  </button>
                ) : null}
                {llmChoices.length > 0 ? (
                  <button
                    type="button"
                    className={domain === "llm" ? "segmented-active" : ""}
                    onClick={() => pickDomain("llm")}
                  >
                    LLM
                  </button>
                ) : null}
              </div>
            ) : null}
            {domain === null ? (
              <p className="hint-text">Choose a domain to see the tasks this project allows.</p>
            ) : (
              <div className="task-choice-grid">
                {domainChoices.map((task) => (
                  <button
                    type="button"
                    className={`task-choice ${taskType === task ? "task-choice-active" : ""}`}
                    key={task}
                    onClick={() => setTaskType(task)}
                  >
                    <strong>{formatDatasetTask(task)}</strong>
                    <span>{taskDescription(task)}</span>
                  </button>
                ))}
              </div>
            )}
            {chosen && !llm ? (
              <span className="field-hint field-hint-start">
                Stored as {formatDatasetFormat(format)}
              </span>
            ) : null}
          </>
        )}
      </div>
      {chosen && llm ? (
        <div className="field">
          <label>Record schema</label>
          {/* The task alone does not fix the record shape, so this is chosen
              explicitly (phase-10 exception to derived format). */}
          <div className="task-choice-grid">
            <button
              type="button"
              className={`task-choice ${llmSchema === "instruction_jsonl" ? "task-choice-active" : ""}`}
              onClick={() => setLlmSchema("instruction_jsonl")}
            >
              <strong>Instruction</strong>
              <span>instruction / input / output rows</span>
            </button>
            <button
              type="button"
              className={`task-choice ${llmSchema === "chat_jsonl" ? "task-choice-active" : ""}`}
              onClick={() => setLlmSchema("chat_jsonl")}
            >
              <strong>Chat</strong>
              <span>system / user / assistant messages</span>
            </button>
          </div>
          <span className="field-hint field-hint-start">Stored as {formatDatasetFormat(format)}</span>
        </div>
      ) : null}
      {chosen && needsLabels ? (
        <>
          <div className="field">
            <label>Class labels</label>
            <div className="label-add-row">
              <input
                value={labelInput}
                onChange={(event) => setLabelInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    addLabels();
                  }
                }}
                placeholder="Type a label and press Enter — or paste several separated by ;"
              />
              <button className="secondary-button" type="button" onClick={addLabels} disabled={!labelInput.trim()}>
                Add label
              </button>
            </div>
          </div>
          {visibleLabels.length === 0 ? (
            <p className="hint-text">Add at least one class label before creating the dataset.</p>
          ) : (
            <div className="label-chip-row">
              {visibleLabels.map((label, index) => (
                <span className="label-chip" key={`${label}-${index}`}>
                  <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
                  {label}
                  <button type="button" onClick={() => removeLabel(index)} title="Remove label">
                    <X size={12} />
                  </button>
                </span>
              ))}
            </div>
          )}
        </>
      ) : null}
      <button
        className="primary-button"
        onClick={() => onCreate({ format })}
        disabled={pending || !newDatasetName.trim() || !chosen || (needsLabels && visibleLabels.length === 0)}
      >
        <FilePlus2 size={16} /> Create dataset
      </button>
    </div>
  );
}

export function PreprocessPanel({
  config,
  setConfig,
  editable,
  onPreview,
  previewUrl,
  previewText,
  previewPending,
  nlp = false
}: {
  config: DatasetPreprocessConfig;
  setConfig: (value: DatasetPreprocessConfig) => void;
  editable: boolean;
  onPreview: () => void;
  previewUrl: string | null;
  previewText?: string | null;
  previewPending: boolean;
  nlp?: boolean;
}) {
  const [expanded, setExpanded] = useState(true);

  function setTransform(name: string, checked: boolean) {
    setConfig({
      ...config,
      transforms: checked
        ? [...new Set([...config.transforms, name])]
        : config.transforms.filter((item) => item !== name)
    });
  }
  return (
    <div className="preprocess-panel accordion-panel">
      <div className="accordion-header">
        <button className="accordion-title" type="button" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
          <ChevronDown className={`accordion-icon ${expanded ? "" : "accordion-icon-collapsed"}`} size={17} />
          <span>Preprocess</span>
        </button>
        <label className="toggle-row">
          <input
            type="checkbox"
            checked={config.enabled}
            disabled={!editable}
            onChange={(event) => setConfig({ ...config, enabled: event.target.checked })}
          />
          <span>Enabled</span>
        </label>
      </div>
      {expanded && (
        config.enabled ? (
          <div className="accordion-body">
            <Field label="Preset">
              <Select
                value={config.preset}
                disabled={!editable}
                onChange={(event) => setConfig({ ...config, preset: event.target.value as DatasetPreprocessConfig["preset"] })}
              >
                <option value="none">None</option>
                {nlp ? (
                  <>
                    <option value="nlp_clean">Clean text</option>
                    <option value="nlp_augment">Augment text</option>
                  </>
                ) : (
                  <>
                    <option value="light">Light</option>
                    <option value="inspection">Inspection</option>
                  </>
                )}
              </Select>
            </Field>
            {!nlp && (
              <div className="grid grid-cols-2 gap-3">
                <Field label="Width">
                  <input
                    type="number"
                    min={32}
                    value={config.resize_width ?? ""}
                    disabled={!editable}
                    onChange={(event) => setConfig({ ...config, resize_width: event.target.value ? Number(event.target.value) : null })}
                  />
                </Field>
                <Field label="Height">
                  <input
                    type="number"
                    min={32}
                    value={config.resize_height ?? ""}
                    disabled={!editable}
                    onChange={(event) => setConfig({ ...config, resize_height: event.target.value ? Number(event.target.value) : null })}
                  />
                </Field>
              </div>
            )}
            <div className="choice-list compact">
              {(nlp
                ? [
                    ["lowercase", "Lowercase"],
                    ["remove_punctuation", "Remove punctuation"],
                    ["remove_stopwords", "Remove stop words"],
                    ["normalize_whitespace", "Normalize whitespace"],
                    ["synonym_replacement", "Synonym replacement"],
                    ["word_swap", "Word swap"],
                    ["word_deletion", "Word deletion"]
                  ]
                : [
                    ["horizontal_flip", "Horizontal flip"],
                    ["vertical_flip", "Vertical flip"],
                    ["brightness_contrast", "Brightness/contrast"],
                    ["gaussian_blur", "Gaussian blur"]
                  ]).map(([value, label]) => (
                <label key={value}>
                  <input
                    type="checkbox"
                    checked={config.transforms.includes(value)}
                    disabled={!editable}
                    onChange={(event) => setTransform(value, event.target.checked)}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
            <Field label="Augmentation mode">
              <Select
                value={config.augmentation_mode}
                disabled={!editable}
                onChange={(event) => setConfig({ ...config, augmentation_mode: event.target.value as DatasetPreprocessConfig["augmentation_mode"] })}
              >
                <option value="random">Random during training</option>
                <option value="materialize">Generate version copies</option>
              </Select>
            </Field>
            {config.augmentation_mode === "materialize" && (
              <Field label={`Generated copies per train ${nlp ? "text" : "image"}`}>
                <input
                  type="number"
                  min={0}
                  max={20}
                  value={config.copies_per_image}
                  disabled={!editable}
                  onChange={(event) => setConfig({ ...config, copies_per_image: Number(event.target.value) })}
                />
              </Field>
            )}
            <button className="secondary-button w-full" onClick={onPreview} disabled={previewPending}>
              {previewPending ? <InlineSpinner label="Previewing" /> : <><ImageIcon size={16} /> Preview</>}
            </button>
            {previewUrl && <div className="image-frame compact"><img src={previewUrl} alt="Preprocess preview" /></div>}
            {previewText && <div className="text-item-preview compact">{previewText}</div>}
          </div>
        ) : (
          <div className="accordion-empty">Preprocess is disabled.</div>
        )
      )}
    </div>
  );
}

export function SplitConfigPanel({
  config,
  setConfig,
  editable,
  dataset,
  onProceed,
  pending
}: {
  config: DatasetSplitConfig;
  setConfig: (value: DatasetSplitConfig) => void;
  editable: boolean;
  dataset: DatasetSummary;
  onProceed: () => void;
  pending: boolean;
}) {
  const [expanded, setExpanded] = useState(true);
  const total = config.train + config.valid + config.test;
  const unassigned = dataset.splits.unassigned?.item_count ?? dataset.splits.unassigned?.image_count ?? 0;
  // LLM records carry no label, so "stratify by label" has nothing to balance
  // on — hide the control and keep the sent config unstratified for them.
  const llm = isLlmTask(dataset.task_type);
  function setRatio(key: "train" | "valid" | "test", value: number) {
    setConfig({ ...config, [key]: Math.max(0, Math.min(1, value)) });
  }
  return (
    <div className="preprocess-panel accordion-panel">
      <div className="accordion-header">
        <button className="accordion-title" type="button" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
          <ChevronDown className={`accordion-icon ${expanded ? "" : "accordion-icon-collapsed"}`} size={17} />
          <span>Dataset Split</span>
        </button>
        <StatusBadge status={`${unassigned} inbox`} />
      </div>
      {expanded && (
        <div className="accordion-body">
          <div className="grid grid-cols-3 gap-3">
            <Field label="Train">
              <input type="number" min={0} max={1} step={0.05} value={config.train} disabled={!editable} onChange={(event) => setRatio("train", Number(event.target.value))} />
            </Field>
            <Field label="Valid">
              <input type="number" min={0} max={1} step={0.05} value={config.valid} disabled={!editable} onChange={(event) => setRatio("valid", Number(event.target.value))} />
            </Field>
            <Field label="Test">
              <input type="number" min={0} max={1} step={0.05} value={config.test} disabled={!editable} onChange={(event) => setRatio("test", Number(event.target.value))} />
            </Field>
          </div>
          <div className="summary-strip">
            {TRAINING_SPLITS.map((splitName) => (
              <span className="split-pill" key={splitName}>
                <strong>{splitName}</strong>
                <span>{Math.round((config[splitName as "train" | "valid" | "test"] / Math.max(total, 0.0001)) * 100)}%</span>
              </span>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Seed">
              <input type="number" value={config.seed} disabled={!editable} onChange={(event) => setConfig({ ...config, seed: Number(event.target.value) })} />
            </Field>
            <div className="choice-list compact">
              {!llm && (
                <label>
                  <input type="checkbox" checked={config.stratify} disabled={!editable} onChange={(event) => setConfig({ ...config, stratify: event.target.checked })} />
                  <span>Stratify labels</span>
                </label>
              )}
              <label>
                <input type="checkbox" checked={config.resplit_all} disabled={!editable} onChange={(event) => setConfig({ ...config, resplit_all: event.target.checked })} />
                <span>Resplit existing</span>
              </label>
            </div>
          </div>
          <button className="primary-button w-full" onClick={onProceed} disabled={!editable || pending}>
            {pending ? <InlineSpinner label="Processing" /> : <><CheckCircle2 size={16} /> Proceed</>}
          </button>
          <p className="hint-text">
            Proceed saves preprocessing and distributes inbox items into train, valid, and test.
          </p>
        </div>
      )}
    </div>
  );
}

export function VersionPanel({
  versions,
  loading,
  versionName,
  setVersionName,
  config,
  onCreate,
  pending,
  editable
}: {
  versions: DatasetVersionSummary[];
  loading: boolean;
  versionName: string;
  setVersionName: (value: string) => void;
  config: DatasetPreprocessConfig;
  onCreate: () => void;
  pending: boolean;
  editable: boolean;
}) {
  return (
    <div className="version-panel">
      <h3 className="section-title mb-0">Version Artifact</h3>
      <Field label="Version name">
        <input value={versionName} onChange={(event) => setVersionName(event.target.value)} placeholder="Optional version name" disabled={!editable} />
      </Field>
      <button className="secondary-button w-full" onClick={onCreate} disabled={!editable || pending}>
        {pending ? <InlineSpinner label="Creating" /> : <><Copy size={16} /> Create version</>}
      </button>
      <p className="hint-text">
        {config.augmentation_mode === "materialize"
          ? `Creates ${config.copies_per_image} generated train copies per item in storage.`
          : "Random mode stores config for training without generated augmentation copies."}
      </p>
      {loading ? (
        <CardGridSkeleton count={1} />
      ) : versions.length > 0 ? (
        <div className="version-list">
          {versions.slice(0, 4).map((version) => (
            <div className="version-row" key={version.id}>
              <strong title={version.name}>{version.name}</strong>
              <span>{version.item_count || version.image_count || version.text_count} items / {version.generated_count} generated</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="hint-text">No version artifacts yet.</p>
      )}
    </div>
  );
}

export function LabelFilterChips({
  dataset,
  value,
  onChange
}: {
  dataset: DatasetSummary;
  value: string;
  onChange: (value: string) => void;
}) {
  const chips = [
    { value: "", label: "All" },
    { value: "__unlabeled__", label: "Unlabeled" },
    ...dataset.labels.map((label) => ({ value: label, label }))
  ];
  return (
    <div className="filter-chip-row">
      {chips.map((chip, index) => (
        <button
          className={`filter-chip ${value === chip.value ? "filter-chip-active" : ""}`}
          key={chip.value || "all"}
          onClick={() => onChange(chip.value)}
          type="button"
        >
          {index > 1 && <span className="class-dot" style={{ backgroundColor: labelColor(index - 2) }} />}
          {chip.label}
        </button>
      ))}
    </div>
  );
}

export function EdaPanel({
  eda,
  loading,
  dataset,
  split,
  setSplit
}: {
  eda: DatasetEdaSummary | undefined;
  loading: boolean;
  dataset: DatasetSummary;
  split: DatasetSplitFilter;
  setSplit: (split: DatasetSplitFilter) => void;
}) {
  const totalImages = eda ? Object.values(eda.split_counts).reduce((total, count) => total + count, 0) : 0;
  const llm = isLlmTask(dataset.task_type);
  const nlp = isNlpTask(dataset.task_type) || llm;
  const roleCounts = eda?.role_counts ?? {};
  const distribution = llm ? roleCounts : (eda?.class_counts ?? {});
  const maxClassCount = eda ? Math.max(1, ...Object.values(distribution)) : 1;
  const maxSplitCount = eda ? Math.max(1, ...Object.values(eda.split_counts)) : 1;
  const coverage = eda && eda.image_count > 0 ? ((eda.image_count - eda.unlabeled_count) / eda.image_count) * 100 : 0;
  const avgAnnotations = eda && eda.image_count > 0 ? eda.annotation_count / eda.image_count : 0;
  return (
    <div className="eda-panel eda-panel-full">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <PanelTitle icon={<BarChart3 size={18} />} title="Exploratory Data Analysis" />
        <StatusBadge status={formatDatasetTask(dataset.task_type)} />
      </div>
      <DatasetSummaryBar dataset={dataset} split={split} setSplit={setSplit} />
      {loading ? (
        <CardGridSkeleton count={4} />
      ) : eda ? (
        <>
          {/* Information, not a warning. A dataset large enough to be sampled
              has nothing wrong with it, and putting this in the warning list
              made a healthy 100,000-row dataset look like it had a problem. */}
          {eda.sampled_items > 0 && (
            <p className="form-caption eda-sampled-note">
              Counts below are estimated from {eda.sampled_items.toLocaleString()} items sampled
              evenly across the split. Split totals are exact.
            </p>
          )}
          <div className="eda-metric-grid">
            <Metric label={llm ? "Records" : nlp ? "Texts" : "Images"} value={llm ? eda.item_count : nlp ? (eda.text_count || eda.item_count) : eda.image_count} />
            {llm ? (
              <>
                <Metric label="Empty outputs" value={eda.unlabeled_count} />
                <Metric label="Duplicates" value={eda.duplicate_count} />
              </>
            ) : (
              <>
                <Metric label="Annotations" value={eda.annotation_count} />
                <Metric label="Unlabeled" value={eda.unlabeled_count} />
                <Metric label="Coverage" value={`${coverage.toFixed(1)}%`} />
                <Metric label="Avg ann/img" value={avgAnnotations.toFixed(2)} />
              </>
            )}
            <Metric label="All splits" value={totalImages} />
          </div>
          <div className="eda-chart-grid">
            <div className="eda-chart-card">
              <h3>{llm ? "Message Roles" : "Class Balance"}</h3>
              <div className="eda-bars">
                {Object.keys(distribution).length === 0 && (
                  <p className="hint-text">{llm ? "No chat roles (instruction dataset)." : "No labels yet."}</p>
                )}
                {Object.entries(distribution).map(([label, count], index) => (
                  <div className="eda-bar-row" key={label}>
                    <span title={label}>
                      <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
                      {label}
                    </span>
                    <div className="eda-bar-track">
                      <div className="eda-bar-fill" style={{ width: `${(count / maxClassCount) * 100}%`, backgroundColor: labelColor(index) }} />
                    </div>
                    <strong>{count}</strong>
                  </div>
                ))}
              </div>
            </div>
            <div className="eda-chart-card">
              <h3>Split Distribution</h3>
              <div className="eda-bars">
                {SPLITS.map((splitName, index) => {
                  const count = eda.split_counts[splitName] ?? 0;
                  return (
                    <div className="eda-bar-row" key={splitName}>
                      <span>{splitName}</span>
                      <div className="eda-bar-track">
                        <div className="eda-bar-fill" style={{ width: `${(count / maxSplitCount) * 100}%`, backgroundColor: labelColor(index) }} />
                      </div>
                      <strong>{count}</strong>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="eda-chart-card">
              <h3>{nlp ? "Length Distribution" : "Image Geometry"}</h3>
              <div className="eda-stats-list">
                {nlp ? (
                  <>
                    <span>Mean tokens <strong>{eda.text_length.mean_tokens ?? "-"}</strong></span>
                    <span>Token range <strong>{eda.text_length.min_tokens ?? "-"} - {eda.text_length.max_tokens ?? "-"}</strong></span>
                    <span>Mean chars <strong>{eda.text_length.mean_chars ?? "-"}</strong></span>
                    <span>Char range <strong>{eda.text_length.min_chars ?? "-"} - {eda.text_length.max_chars ?? "-"}</strong></span>
                  </>
                ) : (
                  <>
                    <span>Mean size <strong>{eda.image_size.mean_width ?? "-"} x {eda.image_size.mean_height ?? "-"}</strong></span>
                    <span>Width range <strong>{eda.image_size.min_width ?? "-"} - {eda.image_size.max_width ?? "-"}</strong></span>
                    <span>Height range <strong>{eda.image_size.min_height ?? "-"} - {eda.image_size.max_height ?? "-"}</strong></span>
                    <span>Aspect ratio <strong>{eda.aspect_ratio.mean ?? "-"}</strong></span>
                  </>
                )}
              </div>
            </div>
            <div className="eda-chart-card">
              <h3>Dataset Health</h3>
              <div className="eda-stats-list">
                {llm ? (
                  <>
                    <span>Empty outputs <strong>{eda.unlabeled_count}</strong></span>
                    <span>Duplicate records <strong>{eda.duplicate_count}</strong></span>
                  </>
                ) : (
                  <>
                    <span>Missing annotations <strong>{eda.missing_annotation_count}</strong></span>
                    <span>Labeled images <strong>{eda.image_count - eda.unlabeled_count}</strong></span>
                  </>
                )}
                <span>Task <strong>{formatDatasetTask(dataset.task_type)}</strong></span>
                <span>Format <strong>{formatDatasetFormat(dataset.format)}</strong></span>
              </div>
            </div>
          </div>
          {eda.warnings.length > 0 && (
            <div className="warning-list">
              {eda.warnings.map((warning) => <span key={warning}>{warning}</span>)}
            </div>
          )}
        </>
      ) : (
        <EmptyState label="No EDA yet" />
      )}
    </div>
  );
}

export function DatasetSummaryBar({
  dataset,
  split,
  setSplit
}: {
  dataset: DatasetSummary;
  split: DatasetSplitFilter;
  setSplit: (split: DatasetSplitFilter) => void;
}) {
  const unit = isLlmTask(dataset.task_type) ? "rec" : isNlpTask(dataset.task_type) ? "txt" : "img";
  const countFor = (splitName: string) => dataset.splits[splitName]?.item_count ?? dataset.splits[splitName]?.image_count ?? 0;
  const allCount = SPLITS.reduce((total, splitName) => total + countFor(splitName), 0);
  return (
    <div className="summary-strip">
      {SPLIT_FILTERS.map((splitName) => {
        const count = splitName === "all" ? allCount : countFor(splitName);
        return (
          <button
            className={`split-pill ${split === splitName ? "split-pill-active" : ""}`}
            key={splitName}
            onClick={() => setSplit(splitName)}
          >
            <strong>{splitName}</strong>
            <span>{count} {unit}</span>
          </button>
        );
      })}
    </div>
  );
}

export function DatasetPagination({
  total,
  offset,
  limit,
  onLimitChange,
  onPageChange
}: {
  total: number;
  offset: number;
  limit: number;
  onLimitChange: (limit: number) => void;
  onPageChange: (page: number) => void;
}) {
  const currentPage = Math.floor(offset / Math.max(limit, 1));
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + limit, total);
  const canPrevious = currentPage > 0;
  const canNext = to < total;
  return (
    <div className="dataset-pagination">
      <label className="dataset-pagination-size">
        <span>Images per page:</span>
        <Select value={limit} onChange={(event) => onLimitChange(Number(event.target.value))}>
          {[25, 50, 100, 200].map((size) => (
            <option value={size} key={size}>{size}</option>
          ))}
        </Select>
      </label>
      <div className="dataset-pagination-nav">
        <strong>{from} - {to} of {total}</strong>
        <button className="icon-button" onClick={() => onPageChange(currentPage - 1)} disabled={!canPrevious} title="Previous page" type="button">
          <ChevronLeft size={16} />
        </button>
        <button className="icon-button" onClick={() => onPageChange(currentPage + 1)} disabled={!canNext} title="Next page" type="button">
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}

export function LabelManager({ dataset, onChanged }: { dataset: DatasetSummary; onChanged: () => void | Promise<void> }) {
  const [newLabel, setNewLabel] = useState("");
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const addLabel = useMutation({
    mutationFn: () => api.addDatasetLabel(dataset.id, newLabel),
    onSuccess: async () => {
      setNewLabel("");
      await onChanged();
    }
  });
  const deleteLabel = useMutation({
    mutationFn: (index: number) => api.deleteDatasetLabel(dataset.id, index, false),
    onSuccess: async () => {
      await onChanged();
    }
  });

  function confirmDeleteLabel(label: string, index: number) {
    confirm({
      title: "Delete label?",
      message: `This will remove "${label}" from "${dataset.name}" when it is not used by existing annotations.`,
      confirmLabel: "Delete label",
      onConfirm: () => deleteLabel.mutate(index)
    });
  }

  return (
    <div className="space-y-3">
      <div className="label-list">
        {dataset.labels.map((label, index) => (
          <div className="label-row" key={`${label}-${index}`}>
            <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
            <strong>{label}</strong>
            <button
              className="icon-button"
              onClick={() => confirmDeleteLabel(label, index)}
              disabled={!dataset.editable || deleteLabel.isPending}
              title="Delete label"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
      <div className="compact-create">
        <input value={newLabel} onChange={(event) => setNewLabel(event.target.value)} />
        <button
          className="secondary-button"
          onClick={() => addLabel.mutate()}
          disabled={!dataset.editable || !newLabel.trim() || addLabel.isPending}
        >
          Add
        </button>
      </div>
      <MutationError mutations={[addLabel, deleteLabel]} />
      {confirmationDialog}
    </div>
  );
}

export function DatasetThumb({
  item,
  active,
  selected,
  onClick,
  onSelected
}: {
  item: DatasetItemSummary;
  active: boolean;
  selected: boolean;
  onClick: () => void;
  onSelected: (checked: boolean) => void;
}) {
  const imageUrl = apiAssetUrl(item.image_url);
  const labelText = item.label ?? (item.classes.join(", ") || "Unlabeled");
  // An llm_finetune record has no image and no class: the two excerpt fields and
  // its token cost are what there is to see, which is what the Records table
  // shows too. Without this it fell through to the image branch and drew an
  // empty thumbnail frame.
  if (item.media_type === "record") {
    return (
      <div className={`record-row ${active ? "record-row-active" : ""}`}>
        <button className="record-row-main" onClick={onClick} type="button">
          <span className="record-row-excerpt" title={item.text_preview || item.filename}>
            {item.text_preview || item.filename}
          </span>
          <span className="record-row-excerpt record-row-output" title={item.output_preview || ""}>
            {item.output_preview || "—"}
          </span>
          <span className="record-row-meta">
            <span className="record-row-split">{item.split}</span>
            <span className="record-row-tokens">{item.token_estimate} tok</span>
          </span>
        </button>
      </div>
    );
  }
  if (item.media_type === "text") {
    return (
      <div className={`text-table-row ${active ? "text-table-row-active" : ""}`}>
        <label className="text-row-check" title={selected ? "Deselect text" : "Select text"}>
          <input
            type="checkbox"
            checked={selected}
            onChange={(event) => onSelected(event.target.checked)}
          />
        </label>
        <button className="text-row-main" onClick={onClick} type="button">
          <span className="text-row-file">
            <strong title={item.filename}>{item.filename}</strong>
            <small>{item.split}</small>
          </span>
          <span className="text-row-preview" title={item.text_preview || item.filename}>{item.text_preview || item.filename}</span>
          <span className="text-row-label" title={labelText}>{labelText}</span>
          <span className="text-row-ann">{item.annotation_count} ann</span>
        </button>
      </div>
    );
  }
  return (
    <div className={`thumb ${active ? "thumb-active" : ""}`}>
      <label className="thumb-check">
        <input
          type="checkbox"
          checked={selected}
          onChange={(event) => onSelected(event.target.checked)}
        />
      </label>
      <button className="thumb-main" onClick={onClick} type="button">
        <span className="thumb-image-frame">
          {imageUrl && <img src={imageUrl} alt={item.filename} />}
          {item.annotations.length > 0 && (
            <svg className="thumb-annotation-svg" viewBox={`0 0 ${item.width} ${item.height}`} preserveAspectRatio="none" aria-hidden="true">
              {item.annotations.map((annotation, index) =>
                annotation.kind === "box" && annotation.bbox ? (
                  <rect
                    key={`${annotation.class_name}-${index}`}
                    x={annotation.bbox.x}
                    y={annotation.bbox.y}
                    width={annotation.bbox.width}
                    height={annotation.bbox.height}
                    style={{ stroke: labelColor(annotation.class_id), fill: labelFill(annotation.class_id) }}
                  />
                ) : annotation.kind === "polygon" ? (
                  <polygon
                    key={`${annotation.class_name}-${index}`}
                    points={pointsAttr(annotation.polygon)}
                    style={{ stroke: labelColor(annotation.class_id), fill: labelFill(annotation.class_id) }}
                  />
                ) : null
              )}
            </svg>
          )}
        </span>
      </button>
      <div className="thumb-meta">
        <strong title={item.filename}>{item.filename}</strong>
        <span title={labelText}>{labelText}</span>
        <small>{item.split} / {item.annotation_count} ann</small>
      </div>
    </div>
  );
}
