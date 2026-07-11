"use client";

/* eslint-disable @next/next/no-img-element */

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Database,
  FilePlus2,
  FileImage,
  FlaskConical,
  FolderOpen,
  ImageIcon,
  MoreVertical,
  Play,
  RefreshCw,
  Save,
  Settings,
  StopCircle,
  Trash2,
  Upload,
  X
} from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, apiAssetUrl, mediaUrl } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { AnnotationEditor as DatasetAnnotationEditor } from "@/features/datasets/annotation-editor";
import { UploadDropCard as DatasetUploadDropCard } from "@/features/datasets/upload-drop-card";
import { SPLIT_FILTERS, SPLITS, TERMINAL_STATUSES, TRAINING_SPLITS } from "@/features/platform/constants";
import {
  activePollInterval,
  areTasksCompatible,
  defaultPreprocessConfig,
  defaultSplitConfig,
  formatDatasetFormat,
  formatDatasetTask,
  formatMetric,
  formatSeconds,
  isActiveStatus,
  labelColor,
  listPollInterval,
  pointsAttr,
  preprocessFromDataset,
  splitConfigFromDataset,
  taskDescription
} from "@/features/platform/utils";
import type {
  DatasetEdaSummary,
  DatasetItemDetail,
  DatasetItemPage,
  DatasetItemSummary,
  DatasetPreprocessConfig,
  DatasetSplitFilter,
  DatasetSplitConfig,
  DatasetSummary,
  DatasetVersionSummary,
  EvaluationJob,
  EvaluationPerImageRow,
  InferenceJob,
  InferenceResult,
  JobProgress,
  ModelInfo,
  ProjectSummary,
  SplitKey,
  TaskType,
  TrainingJob,
  TrainingModelOption
} from "@/types/api";

const IMAGE_TASK_TYPES: TaskType[] = ["classification", "object_detection", "segmentation"];
const DEFAULT_PROJECT_ID = "default-research-project";
const PROJECT_DESCRIPTION_LIMIT = 160;
type EvaluationDisplayKind = "classification" | "vision";
type MetricEntry = { key: string; label: string; value: number };
type ConfirmationTone = "danger" | "warning";
type ConfirmationDialogOptions = {
  title: string;
  message: string;
  confirmLabel: string;
  tone?: ConfirmationTone;
  onConfirm: () => void | Promise<void>;
};

function useConfirmationDialog() {
  const [dialog, setDialog] = useState<ConfirmationDialogOptions | null>(null);
  const confirm = useCallback((options: ConfirmationDialogOptions) => {
    setDialog(options);
  }, []);
  const close = useCallback(() => setDialog(null), []);
  const confirmationDialog = dialog ? (
    <ConfirmationDialog
      {...dialog}
      onCancel={close}
      onConfirm={async () => {
        await dialog.onConfirm();
        close();
      }}
    />
  ) : null;

  return { confirm, confirmationDialog };
}

function ConfirmationDialog({
  title,
  message,
  confirmLabel,
  tone = "danger",
  onCancel,
  onConfirm
}: ConfirmationDialogOptions & { onCancel: () => void; onConfirm: () => void | Promise<void> }) {
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onCancel();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  return (
    <div className="confirmation-overlay" role="presentation" onMouseDown={onCancel}>
      <section
        className={`confirmation-dialog confirmation-dialog-${tone}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirmation-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="confirmation-dialog-header">
          <span className="confirmation-dialog-icon" aria-hidden="true">
            <AlertTriangle size={19} />
          </span>
          <div>
            <h2 id="confirmation-dialog-title">{title}</h2>
            <p>{message}</p>
          </div>
        </div>
        <div className="confirmation-dialog-actions">
          <button className="secondary-button" type="button" onClick={onCancel}>
            Cancel
          </button>
          <button className={tone === "danger" ? "danger-button" : "primary-button"} type="button" onClick={onConfirm} autoFocus>
            {confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}

export function ProjectLandingPage() {
  const router = useRouter();
  const { projectId, projects, projectsLoading, setProjectId, refreshProjects } = useProject();
  const [openMenuId, setOpenMenuId] = useState("");
  const [renameProjectId, setRenameProjectId] = useState("");
  const [renameProjectDraft, setRenameProjectDraft] = useState("");
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const updateProject = useMutation({
    mutationFn: ({ targetProjectId, name }: { targetProjectId: string; name: string }) =>
      api.updateProject(targetProjectId, { name }),
    onSuccess: async () => {
      setOpenMenuId("");
      setRenameProjectId("");
      setRenameProjectDraft("");
      await refreshProjects();
    }
  });
  const deleteProject = useMutation({
    mutationFn: api.deleteProject,
    onSuccess: async (_result, deletedId) => {
      if (deletedId === projectId) setProjectId(DEFAULT_PROJECT_ID);
      setOpenMenuId("");
      setRenameProjectId("");
      await refreshProjects();
    }
  });

  function openProject(projectId: string) {
    setProjectId(projectId);
    router.push("/datasets");
  }

  function openProjectRename(project: ProjectSummary) {
    setRenameProjectId(project.id);
    setRenameProjectDraft(project.name);
    setOpenMenuId("");
  }

  function saveProjectRename() {
    if (!renameProjectId || !renameProjectDraft.trim()) return;
    updateProject.mutate({ targetProjectId: renameProjectId, name: renameProjectDraft.trim() });
  }

  function confirmDeleteProject(project: ProjectSummary) {
    confirm({
      title: "Delete project?",
      message: `This will delete "${project.name}" if it has no owned datasets or history. This action cannot be undone.`,
      confirmLabel: "Delete project",
      onConfirm: () => deleteProject.mutate(project.id)
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Projects" subtitle="Choose a workspace" icon={<Database size={20} />} />
      <section className="panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle icon={<Database size={18} />} title="Project List" />
          <Link className="primary-button" href="/projects/new">
            <FilePlus2 size={16} /> New Project
          </Link>
        </div>
        {projectsLoading ? (
          <CardGridSkeleton count={3} />
        ) : (
          <div className="project-grid">
            {projects.map((project) => (
              <article className="project-card" key={project.id}>
                <button className="project-card-main" onClick={() => openProject(project.id)} type="button">
                  <div className="project-card-title">
                    <strong title={project.name}>{project.name}</strong>
                    <span>{project.id === DEFAULT_PROJECT_ID ? "Default workspace" : "Project workspace"}</span>
                  </div>
                  <p className="project-card-description">
                    {project.description ?? "Local image workspace"}
                  </p>
                  <div className="label-chip-row tag-row-compact">
                    {project.task_types.map((task) => (
                      <span className="label-chip" key={task}>{formatDatasetTask(task)}</span>
                    ))}
                  </div>
                </button>
                <div className="card-menu">
                  <button
                    className="icon-button"
                    onClick={() => setOpenMenuId((value) => (value === project.id ? "" : project.id))}
                    title="Project options"
                    type="button"
                    aria-expanded={openMenuId === project.id}
                  >
                    <MoreVertical size={16} />
                  </button>
                  {openMenuId === project.id && (
                    <div className="option-menu" role="menu">
                      <button type="button" onClick={() => openProject(project.id)}>
                        <FolderOpen size={15} /> Open
                      </button>
                      <button type="button" onClick={() => openProjectRename(project)}>
                        <Save size={15} /> Rename
                      </button>
                      <button
                        className="danger-menu-item"
                        type="button"
                        onClick={() => confirmDeleteProject(project)}
                        disabled={project.id === DEFAULT_PROJECT_ID || deleteProject.isPending}
                      >
                        <Trash2 size={15} /> Delete
                      </button>
                    </div>
                  )}
                </div>
                {renameProjectId === project.id && (
                  <div className="card-rename-popover">
                    <Field label="Project name">
                      <input
                        value={renameProjectDraft}
                        onChange={(event) => setRenameProjectDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") saveProjectRename();
                          if (event.key === "Escape") setRenameProjectId("");
                        }}
                        autoFocus
                      />
                    </Field>
                    <div className="card-rename-actions">
                      <button className="primary-button" onClick={saveProjectRename} disabled={!renameProjectDraft.trim() || updateProject.isPending}>
                        <Save size={16} /> Save
                      </button>
                      <button className="secondary-button" onClick={() => setRenameProjectId("")}>
                        <X size={16} /> Cancel
                      </button>
                    </div>
                  </div>
                )}
              </article>
            ))}
            {projects.length === 0 && <EmptyState label="No projects" />}
          </div>
        )}
        <MutationError mutations={[updateProject, deleteProject]} />
      </section>
      {confirmationDialog}
    </div>
  );
}

export function ProjectCreatePage() {
  const router = useRouter();
  const { setProjectId, refreshProjects } = useProject();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [selectedTasks, setSelectedTasks] = useState<TaskType[]>(IMAGE_TASK_TYPES);
  const trimmedName = name.trim();
  const trimmedDescription = description.trim();
  const canCreateProject = Boolean(trimmedName && trimmedDescription && selectedTasks.length > 0);
  const createProject = useMutation({
    mutationFn: api.createProject,
    onSuccess: async (project) => {
      setProjectId(project.id);
      await refreshProjects();
      router.push("/datasets");
    }
  });

  function toggleTaskCard(task: TaskType) {
    setSelectedTasks((tasks) => {
      if (tasks.includes(task)) {
        const next = tasks.filter((item) => item !== task);
        return next.length ? next : tasks;
      }
      return [...tasks, task];
    });
  }

  function submitProject() {
    if (!canCreateProject) return;
    createProject.mutate({
      name: trimmedName,
      description: trimmedDescription,
      task_types: selectedTasks,
      metadata: { domain: "image" }
    });
  }

  function taskIcon(task: TaskType) {
    if (task === "classification") return <FileImage size={20} />;
    if (task === "object_detection") return <Database size={20} />;
    return <BarChart3 size={20} />;
  }

  return (
    <div className="create-project-page">
      <section className="create-project-hero">
        <Link className="project-back-link" href="/">
          <ArrowLeft size={16} />
          <span>Projects</span>
        </Link>
        <div className="create-project-hero-copy">
          <span className="brand-kicker">Onestep AI Platform</span>
          <h2>Create an image intelligence workspace</h2>
          <p>Shape a focused project for datasets, annotations, training runs, model tests, and inspection workflows.</p>
        </div>
      </section>
      <section className="panel create-project-panel">
        <div className="create-project-form">
          <Field label="Project name">
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Example: Dental Radiograph Workspace" />
          </Field>
          <Field label="Short description">
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Example: Dental X-ray segmentation experiments for YOLO and U-Net models"
              maxLength={PROJECT_DESCRIPTION_LIMIT}
              rows={3}
            />
            <span className="field-hint">{description.length}/{PROJECT_DESCRIPTION_LIMIT}</span>
          </Field>
        </div>
        <div className="field">
          <label>Project type</label>
          <div className="task-choice-grid task-choice-grid-premium">
            {IMAGE_TASK_TYPES.map((task) => {
              const active = selectedTasks.includes(task);
              return (
                <button
                  type="button"
                  aria-pressed={active}
                  className={`task-choice task-choice-premium task-choice-${task.replaceAll("_", "-")} ${
                    active ? "task-choice-active" : ""
                  }`}
                  key={task}
                  onClick={() => toggleTaskCard(task)}
                >
                  <span className="task-choice-icon">{taskIcon(task)}</span>
                  <span className="task-choice-copy">
                    <strong>{formatDatasetTask(task)}</strong>
                    <small>{taskDescription(task)}</small>
                    <em>{active ? "Included" : "Add type"}</em>
                  </span>
                </button>
              );
            })}
          </div>
        </div>
        <div className="create-project-actions">
          <button className="primary-button" onClick={submitProject} disabled={createProject.isPending || !canCreateProject}>
            <FilePlus2 size={16} /> Create Project
          </button>
        </div>
        <MutationError mutations={[createProject]} />
      </section>
    </div>
  );
}

export function SettingsPage() {
  const { projectId, project, projects } = useProject();
  return (
    <div className="space-y-5">
      <PageHeader title="Settings" subtitle="Workspace preferences" icon={<Settings size={20} />} />
      <section className="panel">
        <PanelTitle icon={<Settings size={18} />} title="Workspace" />
        <div className="metric-grid mt-4">
          <Metric label="Projects" value={projects.length} />
          <Metric label="Active project" value={project?.name ?? projectId} />
        </div>
      </section>
    </div>
  );
}

export function DatasetPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { projectId, project } = useProject();
  const datasetParam = searchParams.get("dataset") ?? "";
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [showDuplicate, setShowDuplicate] = useState(false);
  const [showDatasetOptions, setShowDatasetOptions] = useState(false);
  const [catalogMenuDatasetId, setCatalogMenuDatasetId] = useState("");
  const [catalogRenameDatasetId, setCatalogRenameDatasetId] = useState("");
  const [showRename, setShowRename] = useState(false);
  const [detailTab, setDetailTab] = useState<"images" | "annotate" | "eda" | "config">("images");
  const [split, setSplit] = useState<DatasetSplitFilter>("all");
  const [imagePage, setImagePage] = useState(0);
  const [imagesPerPage, setImagesPerPage] = useState(50);
  const [classFilter, setClassFilter] = useState("");
  const [selectedItemId, setSelectedItemId] = useState("");
  const [selectedItemSplit, setSelectedItemSplit] = useState<SplitKey | "">("");
  const [selectedItemIds, setSelectedItemIds] = useState<string[]>([]);
  const [newDatasetName, setNewDatasetName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [labelDraft, setLabelDraft] = useState("object");
  const [cloneName, setCloneName] = useState("");
  const [editName, setEditName] = useState("");
  const [catalogRenameDraft, setCatalogRenameDraft] = useState("");
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadClassId, setUploadClassId] = useState(0);
  const [bulkClassId, setBulkClassId] = useState(0);
  const [preprocessConfig, setPreprocessConfig] = useState<DatasetPreprocessConfig>(defaultPreprocessConfig());
  const [splitConfig, setSplitConfig] = useState<DatasetSplitConfig>(defaultSplitConfig());
  const [moveTarget, setMoveTarget] = useState<SplitKey>("train");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [versionName, setVersionName] = useState("");
  const [uploadErrors, setUploadErrors] = useState<Array<Record<string, string>>>([]);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const allowedDatasetTasks = useMemo(() => {
    const tasks = (project?.task_types ?? []).filter((task): task is TaskType => IMAGE_TASK_TYPES.includes(task));
    return tasks.length ? tasks : IMAGE_TASK_TYPES;
  }, [project?.task_types]);
  const openDataset = useCallback(
    (datasetId: string) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set("dataset", datasetId);
      setSelectedDatasetId(datasetId);
      router.push(`/datasets?${params.toString()}`);
    },
    [router, searchParams]
  );
  const showDatasetCatalog = useCallback(() => {
    setSelectedDatasetId("");
    router.push("/datasets");
  }, [router]);

  const catalogQuery = useQuery({
    queryKey: ["dataset-catalog", projectId],
    queryFn: () => api.datasetCatalog(projectId)
  });
  const datasets = useMemo(() => catalogQuery.data ?? [], [catalogQuery.data]);
  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId),
    [datasets, selectedDatasetId]
  );
  const itemsQuery = useQuery({
    queryKey: ["dataset-items", selectedDataset?.id, split, classFilter, imagePage, imagesPerPage],
    queryFn: () =>
      api.datasetItems(selectedDataset?.id ?? "", {
        split,
        class_name: classFilter && classFilter !== "__unlabeled__" ? classFilter : undefined,
        unlabeled: classFilter === "__unlabeled__",
        limit: imagesPerPage,
        offset: imagePage * imagesPerPage
      }),
    enabled: Boolean(selectedDataset?.id)
  });
  const itemPage = useMemo<DatasetItemPage>(
    () => itemsQuery.data ?? { items: [], total: 0, limit: imagesPerPage, offset: imagePage * imagesPerPage },
    [imagePage, imagesPerPage, itemsQuery.data]
  );
  const items = useMemo(() => itemPage.items, [itemPage.items]);
  const selectedItemSummary = useMemo(
    () => items.find((item) => item.id === selectedItemId && item.split === selectedItemSplit) ?? null,
    [items, selectedItemId, selectedItemSplit]
  );
  const selectedItemDetailSplit = selectedItemSummary?.split ?? selectedItemSplit;
  const detailQuery = useQuery({
    queryKey: ["dataset-item", selectedDataset?.id, selectedItemDetailSplit, selectedItemId],
    queryFn: () => api.datasetItem(selectedDataset?.id ?? "", selectedItemDetailSplit, selectedItemId),
    enabled: Boolean(selectedDataset?.id && selectedItemId && selectedItemDetailSplit && selectedItemSummary)
  });
  const versionsQuery = useQuery({
    queryKey: ["dataset-versions", selectedDataset?.id],
    queryFn: () => api.datasetVersions(selectedDataset?.id ?? ""),
    enabled: Boolean(selectedDataset?.id)
  });
  const edaQuery = useQuery({
    queryKey: ["dataset-eda", selectedDataset?.id, split],
    queryFn: () => api.datasetEda(selectedDataset?.id ?? "", split),
    enabled: Boolean(selectedDataset?.id)
  });

  const createMutation = useMutation({
    mutationFn: api.createDataset,
    onSuccess: async (dataset) => {
      openDataset(dataset.id);
      setShowCreate(false);
      setNewDatasetName("");
      await catalogQuery.refetch();
    }
  });
  const updateDatasetMutation = useMutation({
    mutationFn: ({ datasetId, name, preprocess }: { datasetId: string; name?: string; preprocess?: DatasetPreprocessConfig }) =>
      api.updateDataset(datasetId, { name, preprocess }),
    onSuccess: async (dataset) => {
      setEditName(dataset.name);
      setPreprocessConfig(preprocessFromDataset(dataset));
      setShowRename(false);
      setShowDatasetOptions(false);
      setCatalogRenameDatasetId("");
      setCatalogRenameDraft("");
      setCatalogMenuDatasetId("");
      await catalogQuery.refetch();
    }
  });
  const cloneMutation = useMutation({
    mutationFn: ({ datasetId, name }: { datasetId: string; name?: string }) =>
      api.cloneDataset(datasetId, { name, project_id: projectId }),
    onSuccess: async (dataset) => {
      openDataset(dataset.id);
      setCloneName("");
      setShowDuplicate(false);
      setShowDatasetOptions(false);
      setCatalogMenuDatasetId("");
      await catalogQuery.refetch();
    }
  });
  const uploadMutation = useMutation({
    mutationFn: ({ datasetId, form }: { datasetId: string; form: FormData }) =>
      api.uploadDatasetImages(datasetId, form),
    onSuccess: async (result) => {
      setUploadFiles([]);
      setUploadErrors(result.errors ?? []);
      const lastItem = result.uploaded.at(-1);
      if (lastItem) {
        setSelectedItemId(lastItem.id);
        setSelectedItemSplit(lastItem.split);
      }
      await Promise.all([
        itemsQuery.refetch(),
        catalogQuery.refetch(),
        edaQuery.refetch()
      ]);
    }
  });
  const deleteItemsMutation = useMutation({
    mutationFn: ({ datasetId, ids }: { datasetId: string; ids: string[] }) =>
      api.deleteDatasetItems(datasetId, { split: split === "all" ? "unassigned" : split, ids }),
    onSuccess: async () => {
      setSelectedItemIds([]);
      setSelectedItemId("");
      setSelectedItemSplit("");
      await Promise.all([
        itemsQuery.refetch(),
        catalogQuery.refetch(),
        edaQuery.refetch()
      ]);
    }
  });
  const bulkLabelMutation = useMutation({
    mutationFn: ({ datasetId, ids, classId }: { datasetId: string; ids: string[]; classId: number }) =>
      api.updateDatasetItemLabels(datasetId, split === "all" ? "unassigned" : split, { ids, class_id: classId }),
    onSuccess: async () => {
      await Promise.all([
        itemsQuery.refetch(),
        detailQuery.refetch(),
        catalogQuery.refetch(),
        edaQuery.refetch()
      ]);
    }
  });
  const previewMutation = useMutation({
    mutationFn: ({ datasetId, item }: { datasetId: string; item: DatasetItemDetail }) =>
      api.previewDatasetPreprocess(datasetId, item.split, item.id, preprocessConfig),
    onSuccess: (preview) => setPreviewUrl(apiAssetUrl(preview.image_url))
  });
  const createVersionMutation = useMutation({
    mutationFn: ({ datasetId, name, config }: { datasetId: string; name?: string; config: DatasetPreprocessConfig }) =>
      api.createDatasetVersion(datasetId, {
        name,
        config,
        splits: TRAINING_SPLITS,
        augmentation_splits: ["train"]
      }),
    onSuccess: async () => {
      setVersionName("");
      await versionsQuery.refetch();
    }
  });
  const deleteDatasetMutation = useMutation({
    mutationFn: api.deleteDataset,
    onSuccess: async () => {
      showDatasetCatalog();
      setSelectedItemId("");
      setSelectedItemSplit("");
      setShowDatasetOptions(false);
      setCatalogMenuDatasetId("");
      setCatalogRenameDatasetId("");
      await catalogQuery.refetch();
    }
  });
  const processMutation = useMutation({
    mutationFn: ({ datasetId, preprocess, split }: { datasetId: string; preprocess: DatasetPreprocessConfig; split: DatasetSplitConfig }) =>
      api.processDataset(datasetId, { preprocess, split }),
    onSuccess: async (result) => {
      setPreprocessConfig(preprocessFromDataset(result.dataset));
      setSplitConfig(splitConfigFromDataset(result.dataset));
      setSplit("train");
      setSelectedItemIds([]);
      await Promise.all([
        catalogQuery.refetch(),
        itemsQuery.refetch(),
        edaQuery.refetch()
      ]);
    }
  });
  const moveItemsMutation = useMutation({
    mutationFn: ({ datasetId, target }: { datasetId: string; target: SplitKey }) =>
      api.moveDatasetItems(datasetId, {
        source_split: split === "all" ? "unassigned" : split,
        target_split: target,
        ids: selectedItemIds
      }),
    onSuccess: async () => {
      setSelectedItemIds([]);
      await Promise.all([
        itemsQuery.refetch(),
        catalogQuery.refetch(),
        edaQuery.refetch()
      ]);
    }
  });

  useEffect(() => {
    setSelectedDatasetId(datasetParam);
  }, [datasetParam]);

  useEffect(() => {
    if (!catalogQuery.isLoading && selectedDatasetId && !selectedDataset) {
      showDatasetCatalog();
    }
  }, [catalogQuery.isLoading, selectedDataset, selectedDatasetId, showDatasetCatalog]);

  useEffect(() => {
    if (items.length > 0 && !items.some((item) => item.id === selectedItemId && item.split === selectedItemSplit)) {
      setSelectedItemId(items[0].id);
      setSelectedItemSplit(items[0].split);
    }
    if (items.length === 0) {
      setSelectedItemId("");
      setSelectedItemSplit("");
    }
  }, [items, selectedItemId, selectedItemSplit]);

  useEffect(() => {
    if (!selectedDataset) return;
    setEditName(selectedDataset.name);
    setPreprocessConfig(preprocessFromDataset(selectedDataset));
    setSplitConfig(splitConfigFromDataset(selectedDataset));
    setUploadClassId(0);
    setBulkClassId(0);
    setSelectedItemIds([]);
    setSelectedItemId("");
    setSelectedItemSplit("");
    setPreviewUrl(null);
    setShowDatasetOptions(false);
    setShowRename(false);
    setShowDuplicate(false);
    setCatalogRenameDatasetId("");
    setDetailTab("images");
    setSplit("all");
    setImagePage(0);
  }, [selectedDataset]);

  useEffect(() => {
    setSelectedItemIds([]);
    setImagePage(0);
  }, [split, classFilter, selectedDatasetId]);

  useEffect(() => {
    setImagePage(0);
  }, [imagesPerPage]);

  useEffect(() => {
    if (itemPage.total === 0 && imagePage !== 0) {
      setImagePage(0);
      return;
    }
    const lastPage = Math.max(0, Math.ceil(itemPage.total / imagesPerPage) - 1);
    if (imagePage > lastPage) setImagePage(lastPage);
  }, [imagePage, imagesPerPage, itemPage.total]);

  useEffect(() => {
    if (!allowedDatasetTasks.includes(taskType)) {
      setTaskType(allowedDatasetTasks[0] ?? "classification");
    }
  }, [allowedDatasetTasks, taskType]);

  useEffect(() => {
    if (moveTarget === split) {
      setMoveTarget(SPLITS.find((splitName) => splitName !== split) ?? "train");
    }
  }, [moveTarget, split]);

  function createDataset() {
    const labels = labelDraft
      .split(",")
      .map((label) => label.trim())
      .filter(Boolean);
    if (!newDatasetName.trim()) return;
    createMutation.mutate({
      project_id: projectId,
      name: newDatasetName.trim(),
      task_type: taskType,
      format: taskType === "classification" ? "image_folder" : "yolo",
      labels: labels.length ? labels : ["object"]
    });
  }

  function uploadImages() {
    if (!selectedDataset?.editable || uploadFiles.length === 0) return;
    const form = new FormData();
    form.append("split", split === "all" ? "unassigned" : split);
    uploadFiles.forEach((file) => form.append("files", file));
    if (selectedDataset.task_type === "classification") {
      form.append("class_id", String(uploadClassId));
    }
    uploadMutation.mutate({ datasetId: selectedDataset.id, form });
  }

  function selectDatasetItem(item: DatasetItemSummary, openAnnotate = false) {
    setSelectedItemId(item.id);
    setSelectedItemSplit(item.split);
    if (openAnnotate) setDetailTab("annotate");
  }

  function createVersion() {
    if (!selectedDataset) return;
    createVersionMutation.mutate({
      datasetId: selectedDataset.id,
      name: versionName.trim() || undefined,
      config: preprocessConfig
    });
  }

  function saveDatasetSettings() {
    if (!selectedDataset?.editable) return;
    updateDatasetMutation.mutate({
      datasetId: selectedDataset.id,
      name: editName.trim() || selectedDataset.name,
      preprocess: preprocessConfig
    });
  }

  function openCatalogDatasetRename(dataset: DatasetSummary) {
    if (!dataset.editable) return;
    setCatalogRenameDatasetId(dataset.id);
    setCatalogRenameDraft(dataset.name);
    setCatalogMenuDatasetId("");
  }

  function saveCatalogDatasetRename() {
    if (!catalogRenameDatasetId || !catalogRenameDraft.trim()) return;
    updateDatasetMutation.mutate({
      datasetId: catalogRenameDatasetId,
      name: catalogRenameDraft.trim()
    });
  }

  function confirmDeleteDataset(dataset: DatasetSummary) {
    confirm({
      title: "Delete dataset?",
      message: `This will delete "${dataset.name}" and its editable dataset files. This action cannot be undone.`,
      confirmLabel: "Delete dataset",
      onConfirm: () => deleteDatasetMutation.mutate(dataset.id)
    });
  }

  function confirmDeleteSelectedImages() {
    if (!selectedDataset || selectedItemIds.length === 0) return;
    confirm({
      title: "Remove selected images?",
      message: `This will delete ${selectedItemIds.length} selected image${selectedItemIds.length === 1 ? "" : "s"} from "${selectedDataset.name}". This action cannot be undone.`,
      confirmLabel: "Remove images",
      onConfirm: () => deleteItemsMutation.mutate({ datasetId: selectedDataset.id, ids: selectedItemIds })
    });
  }

  function confirmMoveSelectedImages() {
    if (!selectedDataset || selectedItemIds.length === 0) return;
    confirm({
      title: "Move selected images?",
      message: `This will move ${selectedItemIds.length} selected image${selectedItemIds.length === 1 ? "" : "s"} from ${split} to ${moveTarget}.`,
      confirmLabel: "Move images",
      tone: "warning",
      onConfirm: () => moveItemsMutation.mutate({ datasetId: selectedDataset.id, target: moveTarget })
    });
  }

  function confirmBulkLabelImages() {
    if (!selectedDataset || selectedItemIds.length === 0) return;
    const label = selectedDataset.labels[bulkClassId] ?? "selected label";
    confirm({
      title: "Edit selected labels?",
      message: `This will set ${selectedItemIds.length} selected image${selectedItemIds.length === 1 ? "" : "s"} to "${label}".`,
      confirmLabel: "Edit labels",
      tone: "warning",
      onConfirm: () => bulkLabelMutation.mutate({ datasetId: selectedDataset.id, ids: selectedItemIds, classId: bulkClassId })
    });
  }

  function confirmProcessDataset() {
    if (!selectedDataset) return;
    confirm({
      title: "Proceed with dataset processing?",
      message: `This saves preprocessing settings and distributes inbox images in "${selectedDataset.name}" into train, valid, and test.`,
      confirmLabel: "Proceed",
      tone: "warning",
      onConfirm: () => processMutation.mutate({ datasetId: selectedDataset.id, preprocess: preprocessConfig, split: splitConfig })
    });
  }

  if (catalogQuery.isLoading) {
    return <PageSkeleton title="Loading datasets" />;
  }

  if (!selectedDataset) {
    return (
      <div className="space-y-5">
        <PageHeader title="Datasets" subtitle={project?.name ?? "Project"} icon={<Database size={20} />} />
        <section className="panel">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <PanelTitle icon={<Database size={18} />} title="Dataset Catalog" />
            <div className="flex flex-wrap gap-2">
              {catalogQuery.isFetching && <InlineSpinner label="Refreshing" />}
              <button className="secondary-button" onClick={() => catalogQuery.refetch()}>
                <RefreshCw size={16} /> Refresh
              </button>
              <button className="primary-button" onClick={() => setShowCreate((value) => !value)}>
                <FilePlus2 size={16} /> Add Dataset
              </button>
            </div>
          </div>
          {showCreate && (
            <DatasetCreatePanel
              newDatasetName={newDatasetName}
              setNewDatasetName={setNewDatasetName}
              taskType={taskType}
              setTaskType={setTaskType}
              allowedTaskTypes={allowedDatasetTasks}
              labelDraft={labelDraft}
              setLabelDraft={setLabelDraft}
              onCreate={createDataset}
              pending={createMutation.isPending}
            />
          )}
          <div className="dataset-catalog-grid">
            {datasets.map((dataset) => {
              const totalImages = SPLITS.reduce((total, splitName) => total + (dataset.splits[splitName]?.image_count ?? 0), 0);
              return (
                <article
                  className={`dataset-card dataset-card-${dataset.task_type.replaceAll("_", "-")} ${
                    dataset.editable ? "dataset-card-editable" : "dataset-card-readonly"
                  }`}
                  key={dataset.id}
                >
                  <button className="dataset-card-main" type="button" onClick={() => openDataset(dataset.id)}>
                    <div className="dataset-card-header">
                      <span className="dataset-card-icon"><Database size={17} /></span>
                      <div className="dataset-card-title">
                        <strong title={dataset.name}>{dataset.name}</strong>
                        <span title={`${formatDatasetTask(dataset.task_type)} / ${formatDatasetFormat(dataset.format)}`}>
                          {formatDatasetTask(dataset.task_type)} / {formatDatasetFormat(dataset.format)}
                        </span>
                      </div>
                    </div>
                    <div className="dataset-card-summary">
                      <span><strong>{totalImages}</strong> images</span>
                      <span><strong>{dataset.labels.length}</strong> labels</span>
                    </div>
                    <div className="dataset-card-stats">
                      {TRAINING_SPLITS.map((splitName) => (
                        <span key={splitName}>{splitName}: {dataset.splits[splitName]?.image_count ?? 0}</span>
                      ))}
                    </div>
                  </button>
                  <StatusBadge status={dataset.editable ? "editable" : "read-only"} />
                  <div className="card-menu">
                    <button
                      className="icon-button"
                      onClick={() => setCatalogMenuDatasetId((value) => (value === dataset.id ? "" : dataset.id))}
                      title="Dataset options"
                      type="button"
                      aria-expanded={catalogMenuDatasetId === dataset.id}
                    >
                      <MoreVertical size={16} />
                    </button>
                    {catalogMenuDatasetId === dataset.id && (
                      <div className="option-menu" role="menu">
                        <button type="button" onClick={() => openDataset(dataset.id)}>
                          <FolderOpen size={15} /> Open
                        </button>
                        <button
                          type="button"
                          onClick={() => openCatalogDatasetRename(dataset)}
                          disabled={!dataset.editable}
                        >
                          <Save size={15} /> Rename
                        </button>
                        <button
                          type="button"
                          onClick={() => cloneMutation.mutate({ datasetId: dataset.id })}
                          disabled={cloneMutation.isPending}
                        >
                          <Copy size={15} /> Duplicate
                        </button>
                        <button
                          className="danger-menu-item"
                          type="button"
                          onClick={() => confirmDeleteDataset(dataset)}
                          disabled={!dataset.editable || deleteDatasetMutation.isPending}
                        >
                          <Trash2 size={15} /> Delete
                        </button>
                      </div>
                    )}
                  </div>
                  {catalogRenameDatasetId === dataset.id && (
                    <div className="card-rename-popover">
                      <Field label="Dataset name">
                        <input
                          value={catalogRenameDraft}
                          onChange={(event) => setCatalogRenameDraft(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") saveCatalogDatasetRename();
                            if (event.key === "Escape") setCatalogRenameDatasetId("");
                          }}
                          autoFocus
                        />
                      </Field>
                      <div className="card-rename-actions">
                        <button className="primary-button" onClick={saveCatalogDatasetRename} disabled={!catalogRenameDraft.trim() || updateDatasetMutation.isPending}>
                          <Save size={16} /> Save
                        </button>
                        <button className="secondary-button" onClick={() => setCatalogRenameDatasetId("")}>
                          <X size={16} /> Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
            {datasets.length === 0 && <EmptyState label="No datasets" />}
          </div>
          <MutationError mutations={[createMutation, updateDatasetMutation, cloneMutation, deleteDatasetMutation]} />
        </section>
        {confirmationDialog}
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="dataset-detail-header">
        <PageHeader title={selectedDataset.name} subtitle="Dataset workspace" icon={<Database size={20} />} />
        <div className="dataset-header-actions">
          <button className="secondary-button" onClick={showDatasetCatalog}>
            Back to catalog
          </button>
          <div className="dataset-options">
            <button
              className="icon-button"
              onClick={() => setShowDatasetOptions((value) => !value)}
              title="Dataset options"
              type="button"
              aria-expanded={showDatasetOptions}
            >
              <MoreVertical size={16} />
            </button>
            {showDatasetOptions && (
              <div className="option-menu" role="menu">
                <button
                  type="button"
                  onClick={() => {
                    setShowRename(true);
                    setShowDuplicate(false);
                    setShowDatasetOptions(false);
                  }}
                >
                  <Save size={15} /> Rename
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowDuplicate(true);
                    setShowRename(false);
                    setShowDatasetOptions(false);
                  }}
                >
                  <Copy size={15} /> Duplicate
                </button>
                <button
                  className="danger-menu-item"
                  type="button"
                  onClick={() => confirmDeleteDataset(selectedDataset)}
                  disabled={!selectedDataset.editable || deleteDatasetMutation.isPending}
                >
                  <Trash2 size={15} /> Delete
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
      {showRename && (
        <section className="inline-dialog dataset-action-panel">
          <Field label="Dataset name">
            <input value={editName} onChange={(event) => setEditName(event.target.value)} disabled={!selectedDataset.editable} />
          </Field>
          <div className="flex flex-wrap gap-2">
            <button className="primary-button" onClick={saveDatasetSettings} disabled={!selectedDataset.editable || updateDatasetMutation.isPending}>
              <Save size={16} /> Save
            </button>
            <button className="secondary-button" onClick={() => setShowRename(false)}>
              Cancel
            </button>
          </div>
        </section>
      )}
      {showDuplicate && (
        <section className="inline-dialog dataset-action-panel">
          <Field label="Duplicate name">
            <input value={cloneName} onChange={(event) => setCloneName(event.target.value)} placeholder={`${selectedDataset.name} Copy`} />
          </Field>
          <div className="flex flex-wrap gap-2">
            <button
              className="primary-button"
              onClick={() => cloneMutation.mutate({ datasetId: selectedDataset.id, name: cloneName || undefined })}
              disabled={cloneMutation.isPending}
            >
              <Copy size={16} /> Duplicate
            </button>
            <button className="secondary-button" onClick={() => setShowDuplicate(false)}>
              Cancel
            </button>
          </div>
        </section>
      )}
      <MutationError mutations={[updateDatasetMutation, cloneMutation, deleteDatasetMutation]} />
      <div className="detail-tab-row">
        <button
          className={`detail-tab ${detailTab === "images" ? "detail-tab-active" : ""}`}
          onClick={() => setDetailTab("images")}
          type="button"
        >
          <ImageIcon size={16} /> Images
        </button>
        <button
          className={`detail-tab ${detailTab === "annotate" ? "detail-tab-active" : ""}`}
          onClick={() => setDetailTab("annotate")}
          type="button"
        >
          <BarChart3 size={16} /> Annotate
        </button>
        <button
          className={`detail-tab ${detailTab === "eda" ? "detail-tab-active" : ""}`}
          onClick={() => setDetailTab("eda")}
          type="button"
        >
          <BarChart3 size={16} /> EDA
        </button>
        <button
          className={`detail-tab ${detailTab === "config" ? "detail-tab-active" : ""}`}
          onClick={() => setDetailTab("config")}
          type="button"
        >
          <Save size={16} /> Config
        </button>
      </div>
      {detailTab === "images" ? (
        <section className="panel dataset-work-panel">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <PanelTitle icon={<ImageIcon size={18} />} title="Images" />
            <button className="icon-button" onClick={() => itemsQuery.refetch()} title="Refresh">
              <RefreshCw size={16} />
            </button>
          </div>
          {selectedDataset && <DatasetSummaryBar dataset={selectedDataset} split={split} setSplit={setSplit} />}
          <LabelFilterChips dataset={selectedDataset} value={classFilter} onChange={setClassFilter} />
          {selectedDataset.editable && (
            <DatasetUploadDropCard
              files={uploadFiles}
              setFiles={setUploadFiles}
              dataset={selectedDataset}
              uploadClassId={uploadClassId}
              setUploadClassId={setUploadClassId}
              onUpload={uploadImages}
              pending={uploadMutation.isPending}
              errors={uploadErrors}
            />
          )}
          <div className="bulk-bar">
            <label className="check-row">
              <input
                type="checkbox"
                checked={items.length > 0 && selectedItemIds.length === items.length}
                onChange={(event) => setSelectedItemIds(event.target.checked ? items.map((item) => item.id) : [])}
              />
              <span>{selectedItemIds.length ? `${selectedItemIds.length} selected` : "Select images"}</span>
            </label>
            {selectedDataset?.editable && split !== "all" && (
              <>
                <select value={moveTarget} onChange={(event) => setMoveTarget(event.target.value as SplitKey)} disabled={selectedItemIds.length === 0}>
                  {SPLITS.filter((splitName) => splitName !== split).map((splitName) => (
                    <option value={splitName} key={splitName}>{splitName}</option>
                  ))}
                </select>
                <button
                  className="secondary-button"
                  onClick={confirmMoveSelectedImages}
                  disabled={selectedItemIds.length === 0 || moveItemsMutation.isPending || moveTarget === split}
                >
                  <Copy size={16} /> Move
                </button>
                <button
                  className="danger-button"
                  onClick={confirmDeleteSelectedImages}
                  disabled={selectedItemIds.length === 0 || deleteItemsMutation.isPending}
                >
                  <Trash2 size={16} /> Remove
                </button>
              </>
            )}
          </div>
          {itemsQuery.isLoading && <CardGridSkeleton count={6} />}
          <div className="image-grid">
            {items.map((item) => (
              <DatasetThumb
                item={item}
                key={`${item.split}-${item.id}`}
                active={item.id === selectedItemId && item.split === selectedItemSplit}
                onClick={() => selectDatasetItem(item, true)}
                selected={selectedItemIds.includes(item.id)}
                onSelected={(checked) =>
                  setSelectedItemIds((ids) => checked ? [...new Set([...ids, item.id])] : ids.filter((id) => id !== item.id))
                }
              />
            ))}
            {items.length === 0 && <EmptyState label="No images" />}
          </div>
          <DatasetPagination
            total={itemPage.total}
            offset={itemPage.offset}
            limit={imagesPerPage}
            onLimitChange={setImagesPerPage}
            onPageChange={setImagePage}
          />
          <MutationError mutations={[uploadMutation, deleteItemsMutation, moveItemsMutation]} />
        </section>
      ) : detailTab === "annotate" ? (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_440px]">
          <section className="panel dataset-work-panel">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <PanelTitle icon={<ImageIcon size={18} />} title="Annotate Images" />
              <button className="icon-button" onClick={() => itemsQuery.refetch()} title="Refresh">
                <RefreshCw size={16} />
              </button>
            </div>
            {selectedDataset && <DatasetSummaryBar dataset={selectedDataset} split={split} setSplit={setSplit} />}
            <LabelFilterChips dataset={selectedDataset} value={classFilter} onChange={setClassFilter} />
            <div className="bulk-bar">
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={items.length > 0 && selectedItemIds.length === items.length}
                  onChange={(event) => setSelectedItemIds(event.target.checked ? items.map((item) => item.id) : [])}
                />
                <span>{selectedItemIds.length ? `${selectedItemIds.length} selected` : "Select images"}</span>
              </label>
              {selectedDataset.editable && selectedDataset.task_type === "classification" && split !== "all" && (
                <>
                  <select value={bulkClassId} onChange={(event) => setBulkClassId(Number(event.target.value))} disabled={selectedItemIds.length === 0}>
                    {selectedDataset.labels.map((label, index) => (
                      <option value={index} key={label}>{label}</option>
                    ))}
                  </select>
                  <button
                    className="secondary-button"
                    onClick={confirmBulkLabelImages}
                    disabled={selectedItemIds.length === 0 || bulkLabelMutation.isPending}
                  >
                    <CheckCircle2 size={16} /> Edit label
                  </button>
                </>
              )}
            </div>
            {itemsQuery.isLoading && <CardGridSkeleton count={6} />}
            <div className="image-grid">
              {items.map((item) => (
                <DatasetThumb
                  item={item}
                  key={`${item.split}-${item.id}`}
                  active={item.id === selectedItemId && item.split === selectedItemSplit}
                  onClick={() => selectDatasetItem(item)}
                  selected={selectedItemIds.includes(item.id)}
                  onSelected={(checked) =>
                    setSelectedItemIds((ids) => checked ? [...new Set([...ids, item.id])] : ids.filter((id) => id !== item.id))
                  }
                />
              ))}
              {items.length === 0 && <EmptyState label="No images" />}
            </div>
            <DatasetPagination
              total={itemPage.total}
              offset={itemPage.offset}
              limit={imagesPerPage}
              onLimitChange={setImagesPerPage}
              onPageChange={setImagePage}
            />
            <MutationError mutations={[bulkLabelMutation]} />
          </section>

          <section className="panel">
            <PanelTitle icon={<BarChart3 size={18} />} title="Labels" />
            {selectedDataset && (
              <LabelManager
                dataset={selectedDataset}
                onChanged={async () => {
                  await catalogQuery.refetch();
                }}
              />
            )}
            <div className="divider" />
            <PanelTitle icon={<ImageIcon size={18} />} title="Annotation" />
            {detailQuery.isLoading ? (
              <CardGridSkeleton count={1} />
            ) : detailQuery.data && selectedDataset ? (
              <DatasetAnnotationEditor
                dataset={selectedDataset}
                item={detailQuery.data}
                onSaved={async () => {
                  await Promise.all([
                    detailQuery.refetch(),
                    itemsQuery.refetch(),
                    catalogQuery.refetch(),
                    edaQuery.refetch()
                  ]);
                }}
              />
            ) : (
              <EmptyState label="No image selected" />
            )}
          </section>
        </div>
      ) : detailTab === "eda" ? (
        <section className="panel">
          <EdaPanel
            eda={edaQuery.data}
            loading={edaQuery.isLoading}
            dataset={selectedDataset}
            split={split}
            setSplit={setSplit}
          />
        </section>
      ) : (
        <section className="panel">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <PanelTitle icon={<Save size={18} />} title="Config" />
            {catalogQuery.isFetching && <InlineSpinner label="Refreshing" />}
          </div>
          <div className="config-panel-stack">
            <PreprocessPanel
              config={preprocessConfig}
              setConfig={setPreprocessConfig}
              editable={selectedDataset.editable}
              onPreview={() => detailQuery.data && previewMutation.mutate({ datasetId: selectedDataset.id, item: detailQuery.data })}
              previewUrl={previewUrl}
              previewPending={previewMutation.isPending}
            />
            <SplitConfigPanel
              config={splitConfig}
              setConfig={setSplitConfig}
              editable={selectedDataset.editable}
              dataset={selectedDataset}
              onProceed={confirmProcessDataset}
              pending={processMutation.isPending}
            />
            <VersionPanel
              versions={versionsQuery.data ?? []}
              loading={versionsQuery.isLoading}
              versionName={versionName}
              setVersionName={setVersionName}
              config={preprocessConfig}
              onCreate={createVersion}
              pending={createVersionMutation.isPending}
              editable={selectedDataset.editable}
            />
          </div>
          <MutationError mutations={[previewMutation, createVersionMutation, processMutation]} />
        </section>
      )}
      {confirmationDialog}
    </div>
  );
}

function DatasetCreatePanel({
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
  onCreate: () => void;
  pending: boolean;
}) {
  const labels = labelDraft
    .split(",")
    .map((label) => label.trim())
    .filter(Boolean);
  const visibleLabels = labels.length ? labels : ["object"];
  const [labelInput, setLabelInput] = useState("");
  const format = taskType === "classification" ? "image_folder" : "yolo";

  function syncLabels(nextLabels: string[]) {
    setLabelDraft((nextLabels.length ? nextLabels : ["object"]).join(", "));
  }

  function addLabels() {
    const next = labelInput
      .split(",")
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
        <input value={newDatasetName} onChange={(event) => setNewDatasetName(event.target.value)} placeholder="Example: trash classification" />
      </Field>
      <div className="field">
        <label>Task</label>
        <div className="task-choice-grid">
          {allowedTaskTypes.map((task) => (
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
      </div>
      <div className="format-preview">
        <span>Format</span>
        <strong>{formatDatasetFormat(format)}</strong>
      </div>
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
            placeholder="Add label, e.g. plastic"
          />
          <button className="secondary-button" type="button" onClick={addLabels} disabled={!labelInput.trim()}>
            Add label
          </button>
        </div>
      </div>
      <div className="label-chip-row">
        {visibleLabels.map((label, index) => (
          <span className="label-chip" key={`${label}-${index}`}>
            <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
            {label}
            <button
              type="button"
              onClick={() => removeLabel(index)}
              disabled={visibleLabels.length <= 1}
              title="Remove label"
            >
              <X size={12} />
            </button>
          </span>
        ))}
      </div>
      <button className="primary-button" onClick={onCreate} disabled={pending || !newDatasetName.trim()}>
        <FilePlus2 size={16} /> Create dataset
      </button>
    </div>
  );
}

function PreprocessPanel({
  config,
  setConfig,
  editable,
  onPreview,
  previewUrl,
  previewPending
}: {
  config: DatasetPreprocessConfig;
  setConfig: (value: DatasetPreprocessConfig) => void;
  editable: boolean;
  onPreview: () => void;
  previewUrl: string | null;
  previewPending: boolean;
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
              <select
                value={config.preset}
                disabled={!editable}
                onChange={(event) => setConfig({ ...config, preset: event.target.value as DatasetPreprocessConfig["preset"] })}
              >
                <option value="none">None</option>
                <option value="light">Light</option>
                <option value="inspection">Inspection</option>
              </select>
            </Field>
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
            <div className="choice-list compact">
              {[
                ["horizontal_flip", "Horizontal flip"],
                ["vertical_flip", "Vertical flip"],
                ["brightness_contrast", "Brightness/contrast"],
                ["gaussian_blur", "Gaussian blur"]
              ].map(([value, label]) => (
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
              <select
                value={config.augmentation_mode}
                disabled={!editable}
                onChange={(event) => setConfig({ ...config, augmentation_mode: event.target.value as DatasetPreprocessConfig["augmentation_mode"] })}
              >
                <option value="random">Random during training</option>
                <option value="materialize">Generate version copies</option>
              </select>
            </Field>
            {config.augmentation_mode === "materialize" && (
              <Field label="Generated copies per train image">
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
          </div>
        ) : (
          <div className="accordion-empty">Preprocess is disabled.</div>
        )
      )}
    </div>
  );
}

function SplitConfigPanel({
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
  const unassigned = dataset.splits.unassigned?.image_count ?? 0;
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
              <label>
                <input type="checkbox" checked={config.stratify} disabled={!editable} onChange={(event) => setConfig({ ...config, stratify: event.target.checked })} />
                <span>Stratify labels</span>
              </label>
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
            Proceed saves preprocessing and distributes inbox images into train, valid, and test.
          </p>
        </div>
      )}
    </div>
  );
}

function VersionPanel({
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
          ? `Creates ${config.copies_per_image} generated train copies per image in storage.`
          : "Random mode stores config for training without generated augmentation copies."}
      </p>
      {loading ? (
        <CardGridSkeleton count={1} />
      ) : versions.length > 0 ? (
        <div className="version-list">
          {versions.slice(0, 4).map((version) => (
            <div className="version-row" key={version.id}>
              <strong title={version.name}>{version.name}</strong>
              <span>{version.image_count} images / {version.generated_count} generated</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="hint-text">No version artifacts yet.</p>
      )}
    </div>
  );
}

function LabelFilterChips({
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

function EdaPanel({
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
  const maxClassCount = eda ? Math.max(1, ...Object.values(eda.class_counts)) : 1;
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
          <div className="eda-metric-grid">
            <Metric label="Images" value={eda.image_count} />
            <Metric label="Annotations" value={eda.annotation_count} />
            <Metric label="Unlabeled" value={eda.unlabeled_count} />
            <Metric label="Coverage" value={`${coverage.toFixed(1)}%`} />
            <Metric label="Avg ann/img" value={avgAnnotations.toFixed(2)} />
            <Metric label="All splits" value={totalImages} />
          </div>
          <div className="eda-chart-grid">
            <div className="eda-chart-card">
              <h3>Class Balance</h3>
              <div className="eda-bars">
                {Object.entries(eda.class_counts).map(([label, count], index) => (
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
              <h3>Image Geometry</h3>
              <div className="eda-stats-list">
                <span>Mean size <strong>{eda.image_size.mean_width ?? "-"} x {eda.image_size.mean_height ?? "-"}</strong></span>
                <span>Width range <strong>{eda.image_size.min_width ?? "-"} - {eda.image_size.max_width ?? "-"}</strong></span>
                <span>Height range <strong>{eda.image_size.min_height ?? "-"} - {eda.image_size.max_height ?? "-"}</strong></span>
                <span>Aspect ratio <strong>{eda.aspect_ratio.mean ?? "-"}</strong></span>
              </div>
            </div>
            <div className="eda-chart-card">
              <h3>Dataset Health</h3>
              <div className="eda-stats-list">
                <span>Missing annotations <strong>{eda.missing_annotation_count}</strong></span>
                <span>Labeled images <strong>{eda.image_count - eda.unlabeled_count}</strong></span>
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

function DatasetSummaryBar({
  dataset,
  split,
  setSplit
}: {
  dataset: DatasetSummary;
  split: DatasetSplitFilter;
  setSplit: (split: DatasetSplitFilter) => void;
}) {
  const allCount = SPLITS.reduce((total, splitName) => total + (dataset.splits[splitName]?.image_count ?? 0), 0);
  return (
    <div className="summary-strip">
      {SPLIT_FILTERS.map((splitName) => {
        const count = splitName === "all" ? allCount : dataset.splits[splitName]?.image_count ?? 0;
        return (
          <button
            className={`split-pill ${split === splitName ? "split-pill-active" : ""}`}
            key={splitName}
            onClick={() => setSplit(splitName)}
          >
            <strong>{splitName}</strong>
            <span>{count} img</span>
          </button>
        );
      })}
    </div>
  );
}

function DatasetPagination({
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
        <select value={limit} onChange={(event) => onLimitChange(Number(event.target.value))}>
          {[25, 50, 100, 200].map((size) => (
            <option value={size} key={size}>{size}</option>
          ))}
        </select>
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

function LabelManager({ dataset, onChanged }: { dataset: DatasetSummary; onChanged: () => void | Promise<void> }) {
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

function InferencePageInner({ models, modelsLoading }: { models: ModelInfo[]; modelsLoading?: boolean }) {
  const { projectId } = useProject();
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [selectedModel, setSelectedModel] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.65);
  const [iou, setIou] = useState(0.7);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [activeJobId, setActiveJobId] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const taskModels = useMemo(
    () => models.filter((model) => model.task_type === taskType),
    [models, taskType]
  );
  const selectedModelInfo = useMemo(
    () => taskModels.find((model) => model.id === selectedModel) ?? null,
    [taskModels, selectedModel]
  );
  const modelNameById = useMemo(
    () => Object.fromEntries(models.map((model) => [model.id, model.name])) as Record<string, string>,
    [models]
  );
  const showDetectionParams = selectedModelInfo ? selectedModelInfo.task_type !== "classification" : false;
  const historyQuery = useQuery({
    queryKey: ["inference-history", projectId],
    queryFn: () => api.inferenceHistory(projectId)
  });
  const jobQuery = useQuery({
    queryKey: ["inference-job", activeJobId],
    queryFn: () => api.inferenceJob(activeJobId),
    enabled: Boolean(activeJobId),
    refetchInterval: (query) => activePollInterval(query.state.data as InferenceJob | undefined)
  });
  const inferenceMutation = useMutation({
    mutationFn: api.createInferenceJob,
    onSuccess: (job) => setActiveJobId(job.id)
  });
  const deleteMutation = useMutation({
    mutationFn: ({ ids, clearAll }: { ids: string[]; clearAll?: boolean }) =>
      api.deleteInference(ids, projectId, clearAll),
    onSuccess: async () => {
      setSelectedIds([]);
      await historyQuery.refetch();
    }
  });

  useEffect(() => {
    if (models.length > 0 && !models.some((model) => model.task_type === taskType)) {
      setTaskType(models[0].task_type);
    }
  }, [models, taskType]);

  useEffect(() => {
    if (taskModels.length === 0) {
      if (selectedModel) setSelectedModel("");
      return;
    }
    if (!taskModels.some((model) => model.id === selectedModel)) setSelectedModel(taskModels[0].id);
  }, [taskModels, selectedModel]);

  useEffect(() => {
    if (jobQuery.data?.result) {
      setResult(jobQuery.data.result);
      historyQuery.refetch();
    }
  }, [jobQuery.data?.result, historyQuery]);

  async function runInference() {
    if (!file || !selectedModel) return;
    const form = new FormData();
    form.append("file", file);
    form.append("project_id", projectId);
    form.append("model_id", selectedModel);
    if (showDetectionParams) {
      form.append("confidence_threshold", String(confidence));
      form.append("iou_threshold", String(iou));
    }
    await inferenceMutation.mutateAsync(form);
  }

  function confirmDeleteInferenceRows() {
    if (selectedIds.length === 0) return;
    confirm({
      title: "Delete inference history?",
      message: `This will delete ${selectedIds.length} selected inference result${selectedIds.length === 1 ? "" : "s"} and owned output artifacts.`,
      confirmLabel: "Delete results",
      onConfirm: () => deleteMutation.mutate({ ids: selectedIds })
    });
  }

  function confirmClearInferenceHistory() {
    confirm({
      title: "Clear all inference history?",
      message: "This will delete all inference history for the active project and owned output artifacts.",
      confirmLabel: "Clear all",
      onConfirm: () => deleteMutation.mutate({ ids: [], clearAll: true })
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Inference" subtitle="Run models on image data" icon={<ImageIcon size={20} />} />
      <div className="workspace-grid workspace-grid-inference">
        <section className="panel">
          <PanelTitle icon={<Upload size={18} />} title="Run" />
          {modelsLoading && <CardGridSkeleton count={1} />}
          <Field label="Task">
            <select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}>
              <option value="classification">Classification</option>
              <option value="object_detection">Object detection</option>
              <option value="segmentation">Segmentation</option>
            </select>
          </Field>
          <Field label="Model">
            <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              {taskModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} - {formatDatasetTask(model.task_type)}
                </option>
              ))}
            </select>
            {taskModels.length === 0 && <span className="field-hint">No available models for this task.</span>}
          </Field>
          {selectedModelInfo && (
            <div className="inference-model-meta">
              <span><strong>Task</strong>{formatDatasetTask(selectedModelInfo.task_type)}</span>
              <span><strong>Source</strong>{selectedModelInfo.source}</span>
              <span><strong>Labels</strong>{selectedModelInfo.labels.length}</span>
            </div>
          )}
          <Field label="Image">
            <input type="file" accept="image/png,image/jpeg,image/jpg,image/webp" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          </Field>
          {showDetectionParams && (
            <>
              <SliderField label="Confidence" value={confidence} min={0} max={1} step={0.01} onChange={setConfidence} />
              <SliderField label="IoU" value={iou} min={0} max={1} step={0.01} onChange={setIou} />
            </>
          )}
          <button className="primary-button mt-2 w-full" disabled={!file || !selectedModel || inferenceMutation.isPending} onClick={runInference}>
            <Play size={17} /> Run inference
          </button>
          {inferenceMutation.error && <p className="error-text">{inferenceMutation.error.message}</p>}
          {jobQuery.data && <ProgressPanel progress={jobQuery.data.progress} status={jobQuery.data.status} error={jobQuery.data.error} />}
        </section>
        <section className="panel min-h-[520px]">
          <PanelTitle icon={<ImageIcon size={18} />} title="Result" />
          {result ? <InferenceResultView result={result} modelNameById={modelNameById} /> : <EmptyState label="No result selected" />}
        </section>
        <section className="panel workspace-grid-full">
          <HistoryHeader
            title="Inference History"
            selectedCount={selectedIds.length}
            onRefresh={() => historyQuery.refetch()}
            onDelete={confirmDeleteInferenceRows}
            onClear={confirmClearInferenceHistory}
          />
          <InferenceHistory
            rows={historyQuery.data ?? []}
            selectedIds={selectedIds}
            setSelectedIds={setSelectedIds}
            onSelect={setResult}
            modelNameById={modelNameById}
          />
          {historyQuery.isLoading && <TableSkeleton rows={4} />}
          <MutationError mutations={[deleteMutation]} />
        </section>
      </div>
      {confirmationDialog}
    </div>
  );
}

export function InferencePage() {
  const { projectId } = useProject();
  const modelsQuery = useQuery({
    queryKey: ["models", "available", projectId],
    queryFn: () => api.models(true, projectId)
  });
  return <InferencePageInner models={modelsQuery.data ?? []} modelsLoading={modelsQuery.isLoading} />;
}

export function TestingPage() {
  const { projectId } = useProject();
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [modelIds, setModelIds] = useState<string[]>([]);
  const [datasetKey, setDatasetKey] = useState("");
  const [limit, setLimit] = useState<number | "">("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const modelsQuery = useQuery({
    queryKey: ["models", "available", projectId],
    queryFn: () => api.models(true, projectId)
  });
  const datasetsQuery = useQuery({ queryKey: ["testing-datasets", projectId], queryFn: () => api.datasets(projectId) });
  const jobsQuery = useQuery({
    queryKey: ["testing-jobs", projectId],
    queryFn: () => api.testingJobs(projectId),
    refetchInterval: (query) => listPollInterval(query.state.data as EvaluationJob[] | undefined)
  });
  const mutation = useMutation({
    mutationFn: api.createTestingJobsBatch,
    onSuccess: async () => {
      await jobsQuery.refetch();
    }
  });
  const deleteMutation = useMutation({
    mutationFn: ({ ids, clearAll }: { ids: string[]; clearAll?: boolean }) =>
      api.deleteTestingJobs(ids, projectId, clearAll),
    onSuccess: async () => {
      setSelectedIds([]);
      await jobsQuery.refetch();
    }
  });
  const rawModels = useMemo(() => modelsQuery.data ?? [], [modelsQuery.data]);
  const rawDatasets = useMemo(() => datasetsQuery.data ?? [], [datasetsQuery.data]);
  const datasets = useMemo(
    () => rawDatasets.filter((dataset) => areTasksCompatible(taskType, dataset.task_type)),
    [rawDatasets, taskType]
  );
  const models = useMemo(
    () => rawModels.filter((model) => areTasksCompatible(model.task_type, taskType)),
    [rawModels, taskType]
  );
  const modelTaskById = useMemo(
    () => Object.fromEntries(rawModels.map((model) => [model.id, model.task_type])) as Record<string, TaskType>,
    [rawModels]
  );
  const modelNameById = useMemo(
    () => Object.fromEntries(rawModels.map((model) => [model.id, model.name])) as Record<string, string>,
    [rawModels]
  );
  const comparisonJobs = useMemo(
    () =>
      (jobsQuery.data ?? [])
        .filter((job) => job.dataset_key === datasetKey && job.status === "completed")
        .slice(0, 8),
    [datasetKey, jobsQuery.data]
  );

  useEffect(() => {
    setModelIds((current) => {
      const availableIds = new Set(models.map((model) => model.id));
      const kept = current.filter((id) => availableIds.has(id));
      if (kept.length > 0 || models.length === 0) return kept;
      return [models[0].id];
    });
  }, [models]);
  useEffect(() => {
    const availableDataset = datasets.find((dataset) => dataset.available) ?? datasets[0];
    if (!availableDataset) {
      if (datasetKey) setDatasetKey("");
      return;
    }
    if (!datasets.some((dataset) => dataset.key === datasetKey)) {
      setDatasetKey(availableDataset.key);
    }
  }, [datasetKey, datasets]);

  async function runTesting() {
    if (modelIds.length === 0 || !datasetKey) return;
    await mutation.mutateAsync({
      project_id: projectId,
      model_ids: modelIds,
      dataset_key: datasetKey,
      limit: limit === "" ? null : limit
    });
  }

  function confirmDeleteTestingRows() {
    if (selectedIds.length === 0) return;
    confirm({
      title: "Delete testing jobs?",
      message: `This will delete ${selectedIds.length} selected terminal testing job${selectedIds.length === 1 ? "" : "s"} and owned artifacts.`,
      confirmLabel: "Delete jobs",
      onConfirm: () => deleteMutation.mutate({ ids: selectedIds })
    });
  }

  function confirmClearTestingJobs() {
    confirm({
      title: "Clear all testing jobs?",
      message: "This will delete all terminal testing jobs for the active project and owned artifacts.",
      confirmLabel: "Clear all",
      onConfirm: () => deleteMutation.mutate({ ids: [], clearAll: true })
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Testing" subtitle="Evaluate model runs" icon={<FlaskConical size={20} />} />
      <div className="workspace-grid workspace-grid-testing">
        <section className="panel">
          <PanelTitle icon={<FlaskConical size={18} />} title="New Test" />
          {(modelsQuery.isLoading || datasetsQuery.isLoading) && <CardGridSkeleton count={1} />}
          <Field label="Task">
            <select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}>
              <option value="classification">Classification</option>
              <option value="object_detection">Object detection</option>
              <option value="segmentation">Segmentation</option>
            </select>
          </Field>
          <Field label="Models">
            <div className="choice-list">
              {models.map((model) => (
                <label key={model.id}>
                  <input
                    type="checkbox"
                    checked={modelIds.includes(model.id)}
                    onChange={(event) => toggleId(model.id, event.target.checked, modelIds, setModelIds)}
                  />
                  <span>{model.name}</span>
                </label>
              ))}
              {models.length === 0 && <EmptyState label="No trained models available" />}
            </div>
          </Field>
          <Field label="Dataset">
            <select value={datasetKey} onChange={(event) => setDatasetKey(event.target.value)}>
              {datasets.map((dataset) => (
                <option key={dataset.key} value={dataset.key} disabled={!dataset.available}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Limit">
            <input type="number" min={1} max={500} value={limit} onChange={(event) => setLimit(event.target.value === "" ? "" : Number(event.target.value))} />
          </Field>
          <button className="primary-button w-full" onClick={runTesting} disabled={mutation.isPending || modelIds.length === 0 || !datasetKey}>
            <Play size={17} /> Run test
          </button>
          {mutation.error && <p className="error-text">{mutation.error.message}</p>}
        </section>
        <section className="panel">
          <HistoryHeader
            title="Testing Jobs"
            selectedCount={selectedIds.length}
            onRefresh={() => jobsQuery.refetch()}
            onDelete={confirmDeleteTestingRows}
            onClear={confirmClearTestingJobs}
          />
          <TestingJobTable
            jobs={jobsQuery.data ?? []}
            selectedIds={selectedIds}
            setSelectedIds={setSelectedIds}
            modelTaskById={modelTaskById}
            modelNameById={modelNameById}
          />
          {jobsQuery.isLoading && <TableSkeleton rows={5} />}
          <MutationError mutations={[deleteMutation]} />
        </section>
        <section className="panel workspace-grid-full">
          <PanelTitle icon={<BarChart3 size={18} />} title="Comparison" />
          <TestingComparison jobs={comparisonJobs} modelTaskById={modelTaskById} modelNameById={modelNameById} />
        </section>
      </div>
      {confirmationDialog}
    </div>
  );
}

export function TestingDetailPage({ jobId }: { jobId: string }) {
  const { projectId } = useProject();
  const jobQuery = useQuery({
    queryKey: ["testing-job", jobId],
    queryFn: () => api.testingJob(jobId),
    refetchInterval: (query) => activePollInterval(query.state.data as EvaluationJob | undefined)
  });
  const modelsQuery = useQuery({
    queryKey: ["models", "available", projectId],
    queryFn: () => api.models(true, projectId)
  });
  const comparisonQuery = useQuery({
    queryKey: ["testing-comparison", jobId],
    queryFn: () => api.testingComparison(jobId),
    enabled: Boolean(jobQuery.data)
  });
  const perImageQuery = useQuery({
    queryKey: ["testing-per-image", jobId],
    queryFn: () => api.testingPerImage(jobId),
    enabled: Boolean(jobQuery.data && TERMINAL_STATUSES.has(jobQuery.data.status))
  });
  const job = jobQuery.data;
  const detailModels = modelsQuery.data ?? [];
  const modelTaskById = useMemo(
    () => Object.fromEntries(detailModels.map((model) => [model.id, model.task_type])) as Record<string, TaskType>,
    [detailModels]
  );
  const modelNameById = useMemo(
    () => Object.fromEntries(detailModels.map((model) => [model.id, model.name])) as Record<string, string>,
    [detailModels]
  );
  return (
    <div className="space-y-5">
      <PageHeader title="Testing Detail" subtitle={job?.id.slice(0, 8) ?? jobId.slice(0, 8)} icon={<FlaskConical size={20} />} />
      {job ? (
        <>
          <section className="panel">
            <PanelTitle icon={<BarChart3 size={18} />} title="Comparison" />
            {comparisonQuery.isLoading ? (
              <TableSkeleton rows={4} />
            ) : (
              <TestingComparison
                jobs={comparisonQuery.data?.jobs ?? [job]}
                modelTaskById={modelTaskById}
                modelNameById={modelNameById}
              />
            )}
          </section>
          <section className="panel space-y-5">
            <ProgressPanel progress={job.progress} status={job.status} error={job.error} />
            {perImageQuery.isLoading ? <TableSkeleton rows={6} /> : <MetricsDetails job={job} rows={perImageQuery.data ?? []} />}
          </section>
        </>
      ) : (
        <PageSkeleton title="Loading testing job" />
      )}
    </div>
  );
}

export function TrainingPage() {
  const { projectId } = useProject();
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [modelOptionId, setModelOptionId] = useState("");
  const [modelName, setModelName] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [epochs, setEpochs] = useState(50);
  const [imageSize, setImageSize] = useState(512);
  const [batchSize, setBatchSize] = useState(16);
  const [optimizer, setOptimizer] = useState("AdamW");
  const [learningRate, setLearningRate] = useState(0.002);
  const [device, setDevice] = useState("");
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
  const createMutation = useMutation({
    mutationFn: api.createTrainingJob,
    onSuccess: async () => {
      setModelName("");
      await jobsQuery.refetch();
    }
  });
  const prepareMutation = useMutation({ mutationFn: () => api.prepareModelAsset(modelOptionId, true) });
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
  const datasets = useMemo(
    () => (datasetsQuery.data ?? []).filter((dataset) => dataset.task_type === taskType),
    [datasetsQuery.data, taskType]
  );

  useEffect(() => {
    if (!options.some((item) => item.id === modelOptionId)) {
      const firstRunnable = options.find((item) => item.runnable) ?? options[0];
      if (firstRunnable) setModelOptionId(firstRunnable.id);
    }
  }, [modelOptionId, options]);

  useEffect(() => {
    if (!option) return;
    setImageSize((value) => Number(option.defaults.image_size ?? value));
    setOptimizer((value) => String(option.defaults.optimizer ?? value));
    setLearningRate((value) => Number(option.defaults.learning_rate ?? value));
  }, [option]);

  useEffect(() => {
    if (datasets.length === 0) {
      setDatasetId("");
      return;
    }
    if (!datasets.some((dataset) => dataset.id === datasetId)) setDatasetId(datasets[0].id);
  }, [datasetId, datasets]);

  async function runTraining() {
    const selectedOption = option ?? options[0];
    if (!selectedOption || !datasetId) return;
    await createMutation.mutateAsync({
      project_id: projectId,
      task_type: taskType,
      model_family: selectedOption.family,
      model_option_id: selectedOption.id,
      model_name: modelName.trim() || null,
      epochs,
      image_size: imageSize,
      batch_size: batchSize,
      dataset_id: datasetId,
      optimizer,
      learning_rate: learningRate,
      device,
      cache: "disk",
      workers: 0,
      patience: 50
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
      <div className="workspace-grid workspace-grid-training">
        <section className="panel">
          <PanelTitle icon={<Activity size={18} />} title="New Run" />
          {(optionsQuery.isLoading || datasetsQuery.isLoading) && <CardGridSkeleton count={1} />}
          <Field label="Model name">
            <input value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="Optional display name" />
          </Field>
          <Field label="Task">
            <select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}>
              <option value="classification">Classification</option>
              <option value="object_detection">Object detection</option>
              <option value="segmentation">Segmentation</option>
            </select>
          </Field>
          <Field label="Base model">
            <select value={modelOptionId} onChange={(event) => setModelOptionId(event.target.value)}>
              {options.map((item) => (
                <option key={item.id} value={item.id} disabled={!item.runnable}>
                  {item.name}
                  {item.runnable ? "" : " (gated)"}
                </option>
              ))}
            </select>
          </Field>
          {option && <p className="text-sm text-slate-500">{option.description}</p>}
          <Field label="Dataset">
            <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
              {datasets.map((dataset) => (
                <option value={dataset.id} key={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </Field>
          <div className="form-grid form-grid-three">
            <Field label="Epochs">
              <input type="number" min={1} max={1000} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} />
            </Field>
            <Field label="Size">
              <input type="number" min={128} max={2048} value={imageSize} onChange={(event) => setImageSize(Number(event.target.value))} />
            </Field>
            <Field label="Batch">
              <input type="number" min={-1} max={256} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
            </Field>
          </div>
          <div className="form-grid form-grid-two">
            <Field label="Optimizer">
              <select value={optimizer} onChange={(event) => setOptimizer(event.target.value)}>
                <option value="AdamW">AdamW</option>
                <option value="adam">Adam</option>
                <option value="sgd">SGD</option>
              </select>
            </Field>
            <Field label="LR">
              <input type="number" step={0.0001} value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} />
            </Field>
          </div>
          <Field label="Device">
            <input value={device} onChange={(event) => setDevice(event.target.value)} />
          </Field>
          <div className="action-row action-row-split">
            <button className="secondary-button" onClick={() => prepareMutation.mutate()} disabled={!option?.needs_download || prepareMutation.isPending}>
              <Upload size={16} /> Prepare
            </button>
            <button className="primary-button flex-1" onClick={runTraining} disabled={createMutation.isPending || !option?.runnable || !datasetId}>
              <Play size={17} /> Start training
            </button>
          </div>
          {prepareMutation.data && <p className="text-sm text-slate-500">{prepareMutation.data.status}: {prepareMutation.data.message ?? prepareMutation.data.path}</p>}
          <MutationError mutations={[createMutation, prepareMutation]} />
        </section>
        <section className="panel">
          <HistoryHeader
            title="Training Jobs"
            selectedCount={selectedIds.length}
            onRefresh={() => jobsQuery.refetch()}
            onDelete={confirmDeleteTrainingRows}
            onClear={confirmClearTrainingJobs}
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

export function TrainingDetailPage({ jobId }: { jobId: string }) {
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const jobQuery = useQuery({
    queryKey: ["training-job", jobId],
    queryFn: () => api.trainingJob(jobId),
    refetchInterval: (query) => activePollInterval(query.state.data as TrainingJob | undefined)
  });
  const cancelMutation = useMutation({
    mutationFn: api.cancelTrainingJob,
    onSuccess: async () => {
      await jobQuery.refetch();
    }
  });
  const promoteMutation = useMutation({
    mutationFn: api.promoteTrainingJob,
    onSuccess: async () => {
      await jobQuery.refetch();
    }
  });
  const job = jobQuery.data;

  function confirmCancelTrainingJob(job: TrainingJob) {
    confirm({
      title: "Cancel training job?",
      message: `This will request cancellation for training job ${job.id.slice(0, 8)}. Partial artifacts may remain unavailable.`,
      confirmLabel: "Cancel job",
      tone: "warning",
      onConfirm: () => cancelMutation.mutate(job.id)
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Training Detail" subtitle={job?.id.slice(0, 8) ?? jobId.slice(0, 8)} icon={<Activity size={20} />} />
      {job ? (
        <section className="panel space-y-5">
          <div className="flex flex-wrap gap-2">
            {isActiveStatus(job.status) && (
              <button className="secondary-button" onClick={() => confirmCancelTrainingJob(job)}>
                <StopCircle size={16} /> Cancel
              </button>
            )}
            {job.status === "completed" && job.artifacts?.best_model && !job.promoted_model_id && (
              <button className="secondary-button" onClick={() => promoteMutation.mutate(job.id)}>
                Promote
              </button>
            )}
          </div>
          <ProgressPanel progress={job.progress} status={job.status} error={job.error} />
          <TrainingDetails job={job} />
          <MutationError mutations={[cancelMutation, promoteMutation]} />
        </section>
      ) : (
        <PageSkeleton title="Loading training job" />
      )}
      {confirmationDialog}
    </div>
  );
}

function DatasetThumb({
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
                    style={{ stroke: labelColor(annotation.class_id), fill: `${labelColor(annotation.class_id)}2b` }}
                  />
                ) : annotation.kind === "polygon" ? (
                  <polygon
                    key={`${annotation.class_name}-${index}`}
                    points={pointsAttr(annotation.polygon)}
                    style={{ stroke: labelColor(annotation.class_id), fill: `${labelColor(annotation.class_id)}2b` }}
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

function InferenceResultView({ result, modelNameById = {} }: { result: InferenceResult; modelNameById?: Record<string, string> }) {
  const overlay = mediaUrl(result.overlay_url);
  const modelName = displayModelName(result.model_id, modelNameById);
  return (
    <div className="result-grid">
      <div className="image-frame">{overlay ? <img src={overlay} alt="Prediction overlay" /> : null}</div>
      <div className="inference-result-side">
        <div className="inference-result-summary">
          <span><strong>Model</strong>{modelName}</span>
          <span><strong>Image label</strong>{result.image_level_label}</span>
          <span><strong>Detections</strong>{result.detections.length}</span>
          <span><strong>Time</strong>{result.duration_ms ? `${result.duration_ms} ms` : "-"}</span>
        </div>
        {Object.keys(result.class_scores ?? {}).length > 0 ? (
          <ClassScoreTable scores={result.class_scores} />
        ) : (
          <DetectionTable result={result} />
        )}
      </div>
    </div>
  );
}

function ClassScoreTable({ scores }: { scores: Record<string, number> }) {
  const maxScore = Math.max(0.0001, ...Object.values(scores));
  return (
    <div className="table-wrap mt-4">
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Score</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {Object.entries(scores).map(([label, score], index) => (
            <tr key={label}>
              <td><span className="class-dot" style={{ backgroundColor: labelColor(index) }} />{label}</td>
              <td>{formatMetric(score)}</td>
              <td>
                <div className="eda-bar-track">
                  <div className="eda-bar-fill" style={{ width: `${(score / maxScore) * 100}%`, backgroundColor: labelColor(index) }} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DetectionTable({ result }: { result: InferenceResult }) {
  return (
    <div className="table-wrap mt-4">
      <table>
        <thead>
          <tr>
            <th>Label</th>
            <th>Conf.</th>
            <th>Area</th>
            <th>Box</th>
          </tr>
        </thead>
        <tbody>
          {result.detections.map((detection, index) => (
            <tr key={`${detection.class_name}-${index}`}>
              <td>{detection.class_name}</td>
              <td>{detection.confidence.toFixed(2)}</td>
              <td>{Math.round(detection.mask_area)}</td>
              <td>
                {Math.round(detection.bbox.width)} x {Math.round(detection.bbox.height)}
              </td>
            </tr>
          ))}
          {result.detections.length === 0 && (
            <tr>
              <td colSpan={4}>No detections</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function InferenceHistory({
  rows,
  selectedIds,
  setSelectedIds,
  onSelect,
  modelNameById = {}
}: {
  rows: InferenceResult[];
  selectedIds: string[];
  setSelectedIds: (ids: string[]) => void;
  onSelect: (result: InferenceResult) => void;
  modelNameById?: Record<string, string>;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th />
            <th>Time</th>
            <th>Model</th>
            <th>Label</th>
            <th>Detections</th>
            <th>ms</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="click-row">
              <td data-label="Select">
                <input
                  type="checkbox"
                  checked={selectedIds.includes(row.id)}
                  onChange={(event) => toggleId(row.id, event.target.checked, selectedIds, setSelectedIds)}
                />
              </td>
              <td onClick={() => onSelect(row)}>{new Date(row.created_at).toLocaleString()}</td>
              <td onClick={() => onSelect(row)}>{displayModelName(row.model_id, modelNameById)}</td>
              <td onClick={() => onSelect(row)}>{row.image_level_label}</td>
              <td onClick={() => onSelect(row)}>{row.detections.length}</td>
              <td onClick={() => onSelect(row)}>{row.duration_ms ?? "-"}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={6}>No inference history</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function TestingJobTable({
  jobs,
  selectedIds,
  setSelectedIds,
  modelTaskById = {},
  modelNameById = {}
}: {
  jobs: EvaluationJob[];
  selectedIds: string[];
  setSelectedIds: (ids: string[]) => void;
  modelTaskById?: Record<string, TaskType>;
  modelNameById?: Record<string, string>;
}) {
  const metricColumns = testingTableColumns(jobs, modelTaskById);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th />
            <th>Status</th>
            <th>Model</th>
            <th>Dataset</th>
            <th>Samples</th>
            {metricColumns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td>
                <input
                  type="checkbox"
                  checked={selectedIds.includes(job.id)}
                  onChange={(event) => toggleId(job.id, event.target.checked, selectedIds, setSelectedIds)}
                />
              </td>
              <td data-label="Status"><StatusBadge status={job.status} /></td>
              <td data-label="Model">{displayModelName(job.model_id, modelNameById)}</td>
              <td data-label="Dataset">{job.dataset_key}</td>
              <td data-label="Samples">{job.metrics?.samples ?? "-"}</td>
              {metricColumns.map((column) => (
                <td data-label={column.label} key={column.key}>{formatMetric(column.get(job))}</td>
              ))}
              <td data-label="Open"><Link className="secondary-button" href={`/testing/${job.id}`}>Details</Link></td>
            </tr>
          ))}
          {jobs.length === 0 && (
            <tr>
              <td className="table-empty-cell" colSpan={6 + metricColumns.length}>No testing jobs</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function TestingComparison({
  jobs,
  modelTaskById = {},
  modelNameById = {}
}: {
  jobs: EvaluationJob[];
  modelTaskById?: Record<string, TaskType>;
  modelNameById?: Record<string, string>;
}) {
  if (jobs.length === 0) return <EmptyState label="No completed results for this dataset" />;
  const metricColumns = comparisonMetricColumns(jobs, modelTaskById);
  return (
    <div className="table-wrap responsive-card-table">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Type</th>
            <th>Samples</th>
            {metricColumns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td data-label="Model">{displayModelName(job.model_id, modelNameById)}</td>
              <td data-label="Type">{evaluationKindLabel(inferEvaluationKind(job, modelTaskById[job.model_id]))}</td>
              <td data-label="Samples">{job.metrics?.samples ?? "-"}</td>
              {metricColumns.map((column) => (
                <td data-label={column.label} key={column.key}>{formatMetric(column.get(job))}</td>
              ))}
              <td data-label="Open"><Link className="secondary-button" href={`/testing/${job.id}`}>Details</Link></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function displayModelName(modelId: string, modelNameById: Record<string, string>): string {
  return modelNameById[modelId] ?? modelId;
}

function testingTableColumns(jobs: EvaluationJob[], modelTaskById: Record<string, TaskType>) {
  const kinds = new Set(jobs.map((job) => inferEvaluationKind(job, modelTaskById[job.model_id])));
  if (kinds.size === 1 && kinds.has("classification")) {
    return classificationMetricColumns().slice(0, 2);
  }
  if (kinds.size === 1 && kinds.has("vision")) {
    return visionMetricColumns().slice(0, 2);
  }
  return [
    classificationMetricColumns()[0],
    visionMetricColumns()[0]
  ];
}

function comparisonMetricColumns(jobs: EvaluationJob[], modelTaskById: Record<string, TaskType>) {
  const kinds = new Set(jobs.map((job) => inferEvaluationKind(job, modelTaskById[job.model_id])));
  if (kinds.size === 1 && kinds.has("classification")) return classificationMetricColumns();
  if (kinds.size === 1 && kinds.has("vision")) return visionMetricColumns();
  return [
    ...classificationMetricColumns().slice(0, 2),
    ...visionMetricColumns().slice(0, 2)
  ];
}

function classificationMetricColumns() {
  return [
    { key: "accuracy", label: "Accuracy", get: (job: EvaluationJob) => numberMetric(job.metrics?.image?.overall?.accuracy) },
    { key: "macro_f1", label: "Macro F1", get: (job: EvaluationJob) => numberMetric(job.metrics?.image?.overall?.macro_f1) },
    { key: "weighted_f1", label: "Weighted F1", get: (job: EvaluationJob) => numberMetric(job.metrics?.image?.overall?.weighted_f1) },
    { key: "mcc", label: "MCC", get: (job: EvaluationJob) => numberMetric(job.metrics?.image?.overall?.mcc) }
  ];
}

function visionMetricColumns() {
  return [
    { key: "pixel_dice", label: "Pixel Dice", get: (job: EvaluationJob) => numberMetric(job.metrics?.pixel?.dice) },
    { key: "pixel_iou", label: "Pixel IoU", get: (job: EvaluationJob) => numberMetric(job.metrics?.pixel?.iou) },
    { key: "object_precision", label: "Object Precision", get: (job: EvaluationJob) => numberMetric(job.metrics?.object?.precision) },
    { key: "object_recall", label: "Object Recall", get: (job: EvaluationJob) => numberMetric(job.metrics?.object?.recall) }
  ];
}

function detailMetricEntries(job: EvaluationJob, kind: EvaluationDisplayKind): MetricEntry[] {
  const columns = kind === "classification" ? classificationMetricColumns() : visionMetricColumns();
  const entries = columns
    .map((column) => ({ key: column.key, label: column.label, value: column.get(job) }))
    .filter((entry): entry is MetricEntry => typeof entry.value === "number");
  if (kind === "classification") {
    const macroAuc = numberMetric(job.metrics?.classification?.macro_auc);
    const balancedAccuracy = numberMetric(job.metrics?.image?.overall?.balanced_accuracy);
    return [
      ...(typeof balancedAccuracy === "number" ? [{ key: "balanced_accuracy", label: "Balanced Acc.", value: balancedAccuracy }] : []),
      ...entries,
      ...(typeof macroAuc === "number" ? [{ key: "macro_auc", label: "Macro AUC", value: macroAuc }] : [])
    ];
  }
  return entries;
}

function inferEvaluationKind(job: EvaluationJob, modelTask?: TaskType): EvaluationDisplayKind {
  if (modelTask === "classification") return "classification";
  if (modelTask === "object_detection" || modelTask === "segmentation") return "vision";
  if (job.metrics?.pixel || job.metrics?.object) return "vision";
  return "classification";
}

function evaluationKindLabel(kind: EvaluationDisplayKind): string {
  return kind === "classification" ? "Classification" : "Detection / segmentation";
}

function numberMetric(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function TrainingJobTable({
  jobs,
  selectedIds,
  setSelectedIds
}: {
  jobs: TrainingJob[];
  selectedIds: string[];
  setSelectedIds: (ids: string[]) => void;
}) {
  return (
    <div className="table-wrap responsive-card-table">
      <table>
        <thead>
          <tr>
            <th />
            <th>Status</th>
            <th>Family</th>
            <th>Dataset</th>
            <th>Epoch</th>
            <th>Metric</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td data-label="Select">
                <input
                  type="checkbox"
                  checked={selectedIds.includes(job.id)}
                  onChange={(event) => toggleId(job.id, event.target.checked, selectedIds, setSelectedIds)}
                />
              </td>
              <td data-label="Status"><StatusBadge status={job.status} /></td>
              <td data-label="Family">{job.model_family}</td>
              <td data-label="Dataset">{job.parameters?.dataset_id ?? "-"}</td>
              <td data-label="Epoch">{job.metrics?.epoch ?? job.progress.processed ?? "-"}</td>
              <td data-label="Metric">{formatMetric(job.metrics?.["metrics/mAP50(B)"] ?? job.metrics?.val_accuracy)}</td>
              <td data-label="Open"><Link className="secondary-button" href={`/training/${job.id}`}>Details</Link></td>
            </tr>
          ))}
          {jobs.length === 0 && (
            <tr>
              <td className="table-empty-cell" colSpan={7}>No training jobs</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function MetricsDetails({ job, rows }: { job: EvaluationJob; rows: EvaluationPerImageRow[] }) {
  const metrics = job.metrics ?? {};
  if (!metrics.samples) return <EmptyState label="No metrics yet" />;
  const kind = inferEvaluationKind(job);
  const labels = metrics.labels ?? metrics.image?.labels ?? [];
  const scoreEntries = detailMetricEntries(job, kind);
  return (
    <div className="space-y-5">
      <div className="metric-grid">
        <Metric label="Samples" value={metrics.samples} />
        {scoreEntries.slice(0, 5).map((entry) => (
          <Metric key={entry.key} label={entry.label} value={formatMetric(entry.value)} />
        ))}
      </div>
      <ObjectSummaryPanel objectMetrics={metrics.object ?? {}} />
      <div className="grid gap-4 xl:grid-cols-2">
        <ConfusionMatrix title="Image Confusion" labels={metrics.image?.labels ?? labels} matrix={metrics.image?.confusion_matrix ?? []} />
        {kind === "vision" && (
          <ConfusionMatrix title="Object Confusion" labels={[...labels, "background"]} matrix={metrics.object?.confusion_matrix ?? []} />
        )}
      </div>
      <PerClassReport report={metrics.image?.report ?? {}} />
      <PerImageTable rows={rows} labels={[...labels, "Normal"]} kind={kind} />
    </div>
  );
}

function ObjectSummaryPanel({ objectMetrics }: { objectMetrics: Record<string, any> }) {
  const entries = [
    ["Matched", objectMetrics.matched],
    ["Ground truth", objectMetrics.ground_truth_objects],
    ["Predicted", objectMetrics.predicted_objects],
    ["IoU threshold", objectMetrics.iou_threshold],
  ].filter((entry): entry is [string, number] => typeof entry[1] === "number");
  if (entries.length === 0) return null;
  return (
    <div className="metric-grid">
      {entries.map(([label, value]) => (
        <Metric key={label} label={label} value={formatMetric(value)} />
      ))}
    </div>
  );
}

function TrainingDetails({ job }: { job: TrainingJob }) {
  const metricRows = Object.entries(job.metrics ?? {}).filter(([, value]) => typeof value !== "object");
  const chartKeys = trainingChartKeys(job);
  return (
    <div className="space-y-4">
      <KeyValueTable title="Parameters" value={job.parameters} />
      {metricRows.length > 0 && (
        <div className="metric-grid">
          {metricRows.slice(0, 10).map(([key, value]) => (
            <Metric key={key} label={key.replaceAll("_", " ")} value={formatMetric(value)} />
          ))}
        </div>
      )}
      {chartKeys.length > 0 && (
        <div className="chart-grid">
          {chartKeys.map((key, index) => (
            <MiniLineChart key={key} rows={job.history} valueKey={key} color={labelColor(index)} />
          ))}
        </div>
      )}
      <TrainingRocPanel curves={job.curves} />
      <TrainingArtifactImages urls={job.artifact_urls ?? {}} />
      {job.progress.logs.length > 0 && (
        <div className="log-box">
          {job.progress.logs.slice(-60).map((line, index) => (
            <pre key={`${line}-${index}`}>{line}</pre>
          ))}
        </div>
      )}
    </div>
  );
}

function trainingChartKeys(job: TrainingJob): string[] {
  if (!job.history?.length) return [];
  const preferred = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "train/box_loss",
    "train/cls_loss",
    "val_accuracy",
    "accuracy",
    "val_loss",
    "loss"
  ];
  const available = new Set(Object.keys(job.history[0] ?? {}));
  return preferred.filter((key) => available.has(key)).slice(0, 6);
}

function MiniLineChart({ rows, valueKey, color }: { rows: Array<Record<string, any>>; valueKey: string; color: string }) {
  const values = rows.map((row) => Number(row[valueKey])).filter((value) => Number.isFinite(value));
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 0.000001);
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100;
      const y = 44 - ((value - min) / span) * 38;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <div className="chart-card">
      <div className="flex items-center justify-between gap-3">
        <strong title={valueKey}>{valueKey}</strong>
        <span>{formatMetric(values[values.length - 1])}</span>
      </div>
      <svg viewBox="0 0 100 48" preserveAspectRatio="none">
        <polyline points={points} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  );
}

function TrainingRocPanel({ curves }: { curves: Record<string, any> }) {
  const roc = curves?.roc as Record<string, { fpr: number[]; tpr: number[] }> | undefined;
  if (!roc || Object.keys(roc).length === 0) return null;
  return (
    <div className="chart-card">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <strong>ROC / AUC</strong>
        <span>macro {formatMetric(curves.macro_auc)} / micro {formatMetric(curves.micro_auc)}</span>
      </div>
      <svg className="roc-chart" viewBox="0 0 100 64" preserveAspectRatio="none">
        <line x1="0" y1="64" x2="100" y2="0" stroke="#d7dde5" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        {Object.entries(roc).map(([label, curve], index) => {
          const points = (curve.fpr ?? [])
            .map((fpr, pointIndex) => `${(fpr * 100).toFixed(2)},${(64 - ((curve.tpr?.[pointIndex] ?? 0) * 64)).toFixed(2)}`)
            .join(" ");
          return <polyline key={label} points={points} fill="none" stroke={labelColor(index)} strokeWidth="2" vectorEffect="non-scaling-stroke" />;
        })}
      </svg>
      <div className="label-chip-row mt-3">
        {Object.keys(roc).map((label, index) => (
          <span className="label-chip" key={label}>
            <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

function TrainingArtifactImages({ urls }: { urls: Record<string, string> }) {
  const entries = Object.entries(urls).slice(0, 8);
  if (entries.length === 0) return null;
  return (
    <div>
      <h3 className="section-title">Curve Artifacts</h3>
      <div className="artifact-grid">
        {entries.map(([name, url]) => (
          <div className="artifact-card" key={name}>
            <strong>{name.replaceAll("_", " ")}</strong>
            <img src={mediaUrl(url) ?? url} alt={name} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ConfusionMatrix({ title, labels, matrix }: { title: string; labels: string[]; matrix: number[][] }) {
  if (!matrix.length) return <EmptyState label={title} />;
  return (
    <div>
      <h3 className="section-title">{title}</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th />
              {labels.map((label) => <th key={label}>{label}</th>)}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, index) => (
              <tr key={index}>
                <th>{labels[index] ?? index}</th>
                {row.map((value, valueIndex) => <td key={valueIndex}>{value}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PerClassReport({ report }: { report: Record<string, any> }) {
  const rows = Object.entries(report).filter(([, value]) => typeof value === "object");
  if (rows.length === 0) return null;
  return (
    <div>
      <h3 className="section-title">Class Report</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, value]) => (
              <tr key={label}>
                <td>{label}</td>
                <td>{formatMetric(value.precision)}</td>
                <td>{formatMetric(value.recall)}</td>
                <td>{formatMetric(value["f1-score"])}</td>
                <td>{value.support ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PerImageTable({
  rows,
  labels,
  kind
}: {
  rows: EvaluationPerImageRow[];
  labels: string[];
  kind: EvaluationDisplayKind;
}) {
  if (rows.length === 0) return null;
  const classification = kind === "classification";
  return (
    <div>
      <h3 className="section-title">Per Image</h3>
      <div className="table-wrap max-h-[420px] overflow-y-auto">
        <table>
          <thead>
            {classification ? (
              <tr>
                <th>Image</th>
                <th>Ground truth class</th>
                <th>Predicted class</th>
                <th>Accuracy score</th>
              </tr>
            ) : (
              <tr>
                <th>Image</th>
                <th>GT</th>
                <th>Pred</th>
                <th>Dice</th>
                <th>Matched</th>
                <th>FP</th>
                <th>FN</th>
              </tr>
            )}
          </thead>
          <tbody>
            {rows.map((row) => {
              const accuracyScore = row.ground_truth === row.prediction ? 1 : 0;
              return classification ? (
                <tr key={row.image}>
                  <td>{row.image}</td>
                  <td>{labels[row.ground_truth] ?? row.ground_truth}</td>
                  <td>{labels[row.prediction] ?? row.prediction}</td>
                  <td>{formatMetric(accuracyScore)}</td>
                </tr>
              ) : (
                <tr key={row.image}>
                  <td>{row.image}</td>
                  <td>{labels[row.ground_truth] ?? row.ground_truth}</td>
                  <td>{labels[row.prediction] ?? row.prediction}</td>
                  <td>{formatMetric(row.pixel?.dice)}</td>
                  <td>{row.object?.matched ?? "-"}</td>
                  <td>{row.object?.false_positives ?? "-"}</td>
                  <td>{row.object?.false_negatives ?? "-"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function KeyValueTable({ title, value }: { title: string; value: Record<string, any> }) {
  return (
    <div>
      <h3 className="section-title">{title}</h3>
      <div className="table-wrap">
        <table>
          <tbody>
            {Object.entries(value).map(([key, item]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{typeof item === "object" ? JSON.stringify(item) : String(item)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ProgressPanel({ progress, status, error }: { progress: JobProgress; status: string; error?: string | null }) {
  const progressLabel = progress.total ? `${progress.processed}/${progress.total}` : `${progress.percent.toFixed(0)}%`;
  const percent = Math.max(0, Math.min(100, progress.percent));
  return (
    <div className="progress-panel">
      <div className="flex items-center justify-between gap-3">
        <StatusBadge status={status} />
        <span className="text-sm text-slate-500">{progressLabel}</span>
      </div>
      <div className="progress-track"><span style={{ width: `${percent}%` }} /></div>
      <div className="progress-meta">
        <span>{progress.current_step}</span>
        <span>{formatSeconds(progress.elapsed_seconds)}</span>
      </div>
      {progress.current_item && <p className="text-sm text-slate-500">{progress.current_item}</p>}
      {error && <p className="error-text">{error}</p>}
      {progress.logs.length > 0 && (
        <div className="mini-log">
          {progress.logs.slice(-5).map((line, index) => <span key={`${line}-${index}`}>{line}</span>)}
        </div>
      )}
    </div>
  );
}

function HistoryHeader({
  title,
  selectedCount,
  onRefresh,
  onDelete,
  onClear
}: {
  title: string;
  selectedCount: number;
  onRefresh: () => void;
  onDelete: () => void;
  onClear: () => void;
}) {
  return (
    <div className="history-header">
      <PanelTitle icon={<RefreshCw size={18} />} title={title} />
      <div className="history-actions">
        <button className="icon-button" onClick={onRefresh} title="Refresh"><RefreshCw size={16} /></button>
        <button className="secondary-button" onClick={onDelete} disabled={selectedCount === 0}><Trash2 size={16} /> Delete</button>
        <button className="danger-button" onClick={onClear}><Trash2 size={16} /> Clear all</button>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
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
      <label>{label} <span>{value.toFixed(2)}</span></label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </div>
  );
}

function PageHeader({ title, subtitle, icon }: { title: string; subtitle: string; icon: React.ReactNode }) {
  return (
    <header className="page-header">
      <div>
        <span>{icon}</span>
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
    </header>
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

function PageSkeleton({ title = "Loading" }: { title?: string }) {
  return (
    <div className="space-y-5">
      <header className="page-header">
        <div>
          <span className="skeleton skeleton-icon" />
          <div className="skeleton-stack">
            <span className="skeleton skeleton-title" />
            <span className="skeleton skeleton-line short" />
          </div>
        </div>
      </header>
      <section className="panel">
        <span className="sr-only">{title}</span>
        <CardGridSkeleton count={4} />
      </section>
    </div>
  );
}

function CardGridSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="skeleton-card-grid">
      {Array.from({ length: count }).map((_, index) => (
        <div className="skeleton-card" key={index}>
          <span className="skeleton skeleton-title" />
          <span className="skeleton skeleton-line" />
          <span className="skeleton skeleton-line short" />
          <div className="skeleton-chip-row">
            <span className="skeleton skeleton-chip" />
            <span className="skeleton skeleton-chip" />
            <span className="skeleton skeleton-chip" />
          </div>
        </div>
      ))}
    </div>
  );
}

function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="table-wrap skeleton-table">
      {Array.from({ length: rows }).map((_, index) => (
        <div className="skeleton-table-row" key={index}>
          <span className="skeleton skeleton-line" />
          <span className="skeleton skeleton-line" />
          <span className="skeleton skeleton-line short" />
        </div>
      ))}
    </div>
  );
}

function InlineSpinner({ label }: { label: string }) {
  return (
    <span className="inline-spinner">
      <span className="spinner" />
      <span>{label}</span>
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  const ok = status === "completed" || status === "editable" || status === "available";
  const failed = status === "failed" || status === "canceled" || status === "missing";
  return <span className={`badge ${ok ? "badge-ok" : failed ? "badge-fail" : ""}`}>{status}</span>;
}

function EmptyState({ label }: { label: string }) {
  return (
    <div className="empty-state">
      <ImageIcon size={28} />
      <span>{label}</span>
    </div>
  );
}

function MutationError({ mutations }: { mutations: Array<{ error: Error | null }> }) {
  const error = mutations.find((mutation) => mutation.error)?.error;
  return error ? <p className="error-text">{error.message}</p> : null;
}

function toggleId(id: string, checked: boolean, selectedIds: string[], setSelectedIds: (ids: string[]) => void) {
  setSelectedIds(checked ? [...selectedIds, id] : selectedIds.filter((item) => item !== id));
}
