"use client";

/* eslint-disable @next/next/no-img-element */

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { BarChart3, CheckCircle2, Copy, Database, FilePlus2, FileText, FolderOpen, ImageIcon, MoreVertical, RefreshCw, Save, Trash2, X } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, apiAssetUrl } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { AnnotationEditor as DatasetAnnotationEditor } from "@/features/datasets/annotation-editor";
import { UploadDropCard as DatasetUploadDropCard } from "@/features/datasets/upload-drop-card";
import {
  DatasetCreatePanel,
  DatasetPagination,
  DatasetSummaryBar,
  DatasetThumb,
  EdaPanel,
  LabelFilterChips,
  LabelManager,
  PreprocessPanel,
  SplitConfigPanel,
  VersionPanel
} from "@/features/datasets/dataset-components";
import { SPLITS, TRAINING_SPLITS } from "@/features/platform/constants";
import {
  defaultPreprocessConfig,
  defaultSplitConfig,
  formatDatasetFormat,
  formatDatasetTask,
  isNlpTask,
  NLP_TASK_TYPES,
  preprocessFromDataset,
  splitConfigFromDataset,
  VISION_TASK_TYPES
} from "@/features/platform/utils";
import { CardGridSkeleton, EmptyState, Field, InlineSpinner, MutationError, PageHeader, PageSkeleton, PanelTitle, StatusBadge, useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetItemDetail, DatasetItemPage, DatasetItemSummary, DatasetPreprocessConfig, DatasetSplitConfig, DatasetSplitFilter, DatasetSummary, SplitKey, TaskType } from "@/types/api";

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
  const [previewText, setPreviewText] = useState<string | null>(null);
  const [versionName, setVersionName] = useState("");
  const [uploadErrors, setUploadErrors] = useState<Array<Record<string, string>>>([]);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const allowedDatasetTasks = useMemo(() => {
    const tasks = (project?.task_types ?? []).filter((task): task is TaskType => [...VISION_TASK_TYPES, ...NLP_TASK_TYPES].includes(task));
    return tasks.length ? tasks : VISION_TASK_TYPES;
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
  const selectedDatasetIsNlp = Boolean(selectedDataset && isNlpTask(selectedDataset.task_type));
  const ItemIcon = selectedDatasetIsNlp ? FileText : ImageIcon;
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
    onSuccess: (preview) => {
      setPreviewUrl(preview.media_type === "text" ? null : apiAssetUrl(preview.image_url));
      setPreviewText(preview.media_type === "text" ? preview.text_preview ?? "" : null);
    }
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
    setPreviewText(null);
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
      format: isNlpTask(taskType) ? "text_folder" : taskType === "classification" ? "image_folder" : "yolo",
      labels: labels.length ? labels : taskType === "summarization" ? ["summary"] : taskType === "question_answering" ? ["answer"] : ["object"]
    });
  }

  function uploadImages() {
    if (!selectedDataset?.editable || uploadFiles.length === 0) return;
    const form = new FormData();
    form.append("split", split === "all" ? "unassigned" : split);
    uploadFiles.forEach((file) => form.append("files", file));
    if (selectedDataset.task_type === "classification" || selectedDataset.task_type === "text_classification") {
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
      title: `Remove selected ${selectedDataset && isNlpTask(selectedDataset.task_type) ? "texts" : "images"}?`,
      message: `This will delete ${selectedItemIds.length} selected item${selectedItemIds.length === 1 ? "" : "s"} from "${selectedDataset.name}". This action cannot be undone.`,
      confirmLabel: "Remove items",
      onConfirm: () => deleteItemsMutation.mutate({ datasetId: selectedDataset.id, ids: selectedItemIds })
    });
  }

  function confirmMoveSelectedImages() {
    if (!selectedDataset || selectedItemIds.length === 0) return;
    confirm({
      title: `Move selected ${selectedDataset && isNlpTask(selectedDataset.task_type) ? "texts" : "images"}?`,
      message: `This will move ${selectedItemIds.length} selected item${selectedItemIds.length === 1 ? "" : "s"} from ${split} to ${moveTarget}.`,
      confirmLabel: "Move items",
      tone: "warning",
      onConfirm: () => moveItemsMutation.mutate({ datasetId: selectedDataset.id, target: moveTarget })
    });
  }

  function confirmBulkLabelImages() {
    if (!selectedDataset || selectedItemIds.length === 0) return;
    const label = selectedDataset.labels[bulkClassId] ?? "selected label";
    confirm({
      title: "Edit selected labels?",
      message: `This will set ${selectedItemIds.length} selected item${selectedItemIds.length === 1 ? "" : "s"} to "${label}".`,
      confirmLabel: "Edit labels",
      tone: "warning",
      onConfirm: () => bulkLabelMutation.mutate({ datasetId: selectedDataset.id, ids: selectedItemIds, classId: bulkClassId })
    });
  }

  function confirmProcessDataset() {
    if (!selectedDataset) return;
    confirm({
      title: "Proceed with dataset processing?",
      message: `This saves preprocessing settings and distributes inbox items in "${selectedDataset.name}" into train, valid, and test.`,
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
              const totalImages = SPLITS.reduce((total, splitName) => total + (dataset.splits[splitName]?.item_count ?? dataset.splits[splitName]?.image_count ?? 0), 0);
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
                      <span><strong>{totalImages}</strong> {isNlpTask(dataset.task_type) ? "texts" : "images"}</span>
                      <span><strong>{dataset.labels.length}</strong> labels</span>
                    </div>
                    <div className="dataset-card-stats">
                      {TRAINING_SPLITS.map((splitName) => (
                        <span key={splitName}>{splitName}: {dataset.splits[splitName]?.item_count ?? dataset.splits[splitName]?.image_count ?? 0}</span>
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
            {datasets.length === 0 && <EmptyState label="No datasets yet." icon={<Database size={30} />} centered description="Create or import a dataset to start annotating and training models." />}
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
          <ItemIcon size={16} /> {selectedDatasetIsNlp ? "Texts" : "Images"}
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
            <PanelTitle icon={<ItemIcon size={18} />} title={selectedDatasetIsNlp ? "Texts" : "Images"} />
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
              <span>{selectedItemIds.length ? `${selectedItemIds.length} selected` : `Select ${selectedDatasetIsNlp ? "texts" : "images"}`}</span>
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
          <div className={selectedDatasetIsNlp ? "text-item-list" : "image-grid"}>
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
            {items.length === 0 && <EmptyState label={selectedDatasetIsNlp ? "No texts" : "No images"} />}
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
              <PanelTitle icon={<ItemIcon size={18} />} title={selectedDatasetIsNlp ? "Annotate Texts" : "Annotate Images"} />
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
                <span>{selectedItemIds.length ? `${selectedItemIds.length} selected` : `Select ${selectedDatasetIsNlp ? "texts" : "images"}`}</span>
              </label>
              {selectedDataset.editable && (selectedDataset.task_type === "classification" || selectedDataset.task_type === "text_classification") && split !== "all" && (
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
            <div className={selectedDatasetIsNlp ? "text-item-list" : "image-grid"}>
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
              {items.length === 0 && <EmptyState label={selectedDatasetIsNlp ? "No texts" : "No images"} />}
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
              <EmptyState label={selectedDatasetIsNlp ? "No text selected" : "No image selected"} />
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
              previewText={previewText}
              previewPending={previewMutation.isPending}
              nlp={selectedDatasetIsNlp}
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
