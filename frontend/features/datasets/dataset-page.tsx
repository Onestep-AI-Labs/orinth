"use client";

/* eslint-disable @next/next/no-img-element */

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { FileText, ImageIcon } from "lucide-react";
import { useProject } from "@/components/app-shell";
import { DatasetAnnotateTab } from "@/features/datasets/annotate-tab";
import { DatasetCatalogView } from "@/features/datasets/catalog-view";
import { DatasetConfigTab } from "@/features/datasets/config-tab";
import { EdaPanel } from "@/features/datasets/dataset-components";
import { DatasetDetailHeader } from "@/features/datasets/detail-header";
import { DatasetDetailTabs } from "@/features/datasets/detail-tabs";
import {
  useBulkLabelDatasetItemsMutation,
  useCloneDatasetMutation,
  useCreateDatasetMutation,
  useCreateDatasetVersionMutation,
  useDatasetCatalogQuery,
  useDatasetEdaQuery,
  useDatasetItemDetailQuery,
  useDatasetItemsQuery,
  useDatasetVersionsQuery,
  useDeleteDatasetItemsMutation,
  useDeleteDatasetMutation,
  useMoveDatasetItemsMutation,
  usePreviewDatasetPreprocessMutation,
  useProcessDatasetMutation,
  useUpdateDatasetMutation,
  useUploadDatasetMutation
} from "@/features/datasets/hooks";
import { DatasetImagesTab } from "@/features/datasets/images-tab";
import { SPLITS } from "@/features/platform/constants";
import {
  defaultPreprocessConfig,
  defaultSplitConfig,
  isNlpTask,
  NLP_TASK_TYPES,
  preprocessFromDataset,
  splitConfigFromDataset,
  VISION_TASK_TYPES
} from "@/features/platform/utils";
import { MutationError, PageSkeleton, useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetItemPage, DatasetItemSummary, DatasetPreprocessConfig, DatasetSplitConfig, DatasetSplitFilter, DatasetSummary, SplitKey, TaskType } from "@/types/api";

export function DatasetPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { projectId, project } = useProject();
  const datasetParam = searchParams?.get("dataset") ?? "";
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
      const params = new URLSearchParams(searchParams?.toString() ?? "");
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

  const catalogQuery = useDatasetCatalogQuery(projectId);
  const datasets = useMemo(() => catalogQuery.data ?? [], [catalogQuery.data]);
  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId),
    [datasets, selectedDatasetId]
  );
  const selectedDatasetIsNlp = Boolean(selectedDataset && isNlpTask(selectedDataset.task_type));
  const ItemIcon = selectedDatasetIsNlp ? FileText : ImageIcon;
  const itemsQuery = useDatasetItemsQuery(selectedDataset?.id, split, classFilter, imagePage, imagesPerPage);
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
  const detailQuery = useDatasetItemDetailQuery(
    selectedDataset?.id,
    selectedItemDetailSplit,
    selectedItemId,
    Boolean(selectedDataset?.id && selectedItemId && selectedItemDetailSplit && selectedItemSummary)
  );
  const versionsQuery = useDatasetVersionsQuery(selectedDataset?.id);
  const edaQuery = useDatasetEdaQuery(selectedDataset?.id, split);

  const createMutation = useCreateDatasetMutation({ openDataset, setShowCreate, setNewDatasetName, catalogQuery });
  const updateDatasetMutation = useUpdateDatasetMutation({
    setEditName,
    setPreprocessConfig,
    setShowRename,
    setShowDatasetOptions,
    setCatalogRenameDatasetId,
    setCatalogRenameDraft,
    setCatalogMenuDatasetId,
    catalogQuery
  });
  const cloneMutation = useCloneDatasetMutation({
    projectId,
    openDataset,
    setCloneName,
    setShowDuplicate,
    setShowDatasetOptions,
    setCatalogMenuDatasetId,
    catalogQuery
  });
  const uploadMutation = useUploadDatasetMutation({
    setUploadFiles,
    setUploadErrors,
    setSelectedItemId,
    setSelectedItemSplit,
    itemsQuery,
    catalogQuery,
    edaQuery
  });
  const deleteItemsMutation = useDeleteDatasetItemsMutation({
    split,
    setSelectedItemIds,
    setSelectedItemId,
    setSelectedItemSplit,
    itemsQuery,
    catalogQuery,
    edaQuery
  });
  const bulkLabelMutation = useBulkLabelDatasetItemsMutation({ split, itemsQuery, detailQuery, catalogQuery, edaQuery });
  const previewMutation = usePreviewDatasetPreprocessMutation({ preprocessConfig, setPreviewUrl, setPreviewText });
  const createVersionMutation = useCreateDatasetVersionMutation({ setVersionName, versionsQuery });
  const deleteDatasetMutation = useDeleteDatasetMutation({
    showDatasetCatalog,
    setSelectedItemId,
    setSelectedItemSplit,
    setShowDatasetOptions,
    setCatalogMenuDatasetId,
    setCatalogRenameDatasetId,
    catalogQuery
  });
  const processMutation = useProcessDatasetMutation({
    setPreprocessConfig,
    setSplitConfig,
    setSplit,
    setSelectedItemIds,
    catalogQuery,
    itemsQuery,
    edaQuery
  });
  const moveItemsMutation = useMoveDatasetItemsMutation({
    split,
    selectedItemIds,
    setSelectedItemIds,
    itemsQuery,
    catalogQuery,
    edaQuery
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

  function selectDatasetItem(item: DatasetItemSummary, openAnnotate = false) {
    setSelectedItemId(item.id);
    setSelectedItemSplit(item.split);
    if (openAnnotate) setDetailTab("annotate");
  }

  function confirmDeleteDataset(dataset: DatasetSummary) {
    confirm({
      title: "Delete dataset?",
      message: `This will delete "${dataset.name}" and its editable dataset files. This action cannot be undone.`,
      confirmLabel: "Delete dataset",
      onConfirm: () => deleteDatasetMutation.mutate(dataset.id)
    });
  }

  if (catalogQuery.isLoading) {
    return <PageSkeleton title="Loading datasets" />;
  }

  if (!selectedDataset) {
    return (
      <DatasetCatalogView
        projectName={project?.name}
        projectId={projectId}
        datasets={datasets}
        catalogQuery={catalogQuery}
        showCreate={showCreate}
        setShowCreate={setShowCreate}
        newDatasetName={newDatasetName}
        setNewDatasetName={setNewDatasetName}
        taskType={taskType}
        setTaskType={setTaskType}
        allowedTaskTypes={allowedDatasetTasks}
        labelDraft={labelDraft}
        setLabelDraft={setLabelDraft}
        createMutation={createMutation}
        openDataset={openDataset}
        catalogMenuDatasetId={catalogMenuDatasetId}
        setCatalogMenuDatasetId={setCatalogMenuDatasetId}
        catalogRenameDatasetId={catalogRenameDatasetId}
        setCatalogRenameDatasetId={setCatalogRenameDatasetId}
        catalogRenameDraft={catalogRenameDraft}
        setCatalogRenameDraft={setCatalogRenameDraft}
        cloneMutation={cloneMutation}
        updateDatasetMutation={updateDatasetMutation}
        deleteDatasetMutation={deleteDatasetMutation}
        onDeleteDataset={confirmDeleteDataset}
        confirmationDialog={confirmationDialog}
      />
    );
  }

  const sharedTabProps = {
    dataset: selectedDataset,
    itemIcon: ItemIcon,
    isNlp: selectedDatasetIsNlp,
    itemsQuery,
    split,
    setSplit,
    classFilter,
    setClassFilter,
    items,
    itemPage,
    imagesPerPage,
    setImagesPerPage,
    imagePage,
    setImagePage,
    selectedItemId,
    selectedItemSplit,
    selectedItemIds,
    setSelectedItemIds,
    onSelectItem: selectDatasetItem,
    confirm
  };

  return (
    <div className="space-y-5">
      <DatasetDetailHeader
        dataset={selectedDataset}
        onBackToCatalog={showDatasetCatalog}
        showDatasetOptions={showDatasetOptions}
        setShowDatasetOptions={setShowDatasetOptions}
        showRename={showRename}
        setShowRename={setShowRename}
        showDuplicate={showDuplicate}
        setShowDuplicate={setShowDuplicate}
        editName={editName}
        setEditName={setEditName}
        cloneName={cloneName}
        setCloneName={setCloneName}
        preprocessConfig={preprocessConfig}
        updateDatasetMutation={updateDatasetMutation}
        cloneMutation={cloneMutation}
        deleteDatasetMutation={deleteDatasetMutation}
        onDeleteDataset={confirmDeleteDataset}
      />
      <MutationError mutations={[updateDatasetMutation, cloneMutation, deleteDatasetMutation]} />
      <DatasetDetailTabs
        itemIcon={ItemIcon}
        itemLabel={selectedDatasetIsNlp ? "Texts" : "Images"}
        activeTab={detailTab}
        setActiveTab={setDetailTab}
      />
      {detailTab === "images" ? (
        <DatasetImagesTab
          {...sharedTabProps}
          uploadFiles={uploadFiles}
          setUploadFiles={setUploadFiles}
          uploadClassId={uploadClassId}
          setUploadClassId={setUploadClassId}
          uploadErrors={uploadErrors}
          uploadMutation={uploadMutation}
          moveTarget={moveTarget}
          setMoveTarget={setMoveTarget}
          moveItemsMutation={moveItemsMutation}
          deleteItemsMutation={deleteItemsMutation}
        />
      ) : detailTab === "annotate" ? (
        <DatasetAnnotateTab
          {...sharedTabProps}
          bulkClassId={bulkClassId}
          setBulkClassId={setBulkClassId}
          bulkLabelMutation={bulkLabelMutation}
          catalogQuery={catalogQuery}
          detailQuery={detailQuery}
          edaQuery={edaQuery}
        />
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
        <DatasetConfigTab
          dataset={selectedDataset}
          nlp={selectedDatasetIsNlp}
          catalogQuery={catalogQuery}
          preprocessConfig={preprocessConfig}
          setPreprocessConfig={setPreprocessConfig}
          detailQuery={detailQuery}
          previewUrl={previewUrl}
          previewText={previewText}
          previewMutation={previewMutation}
          splitConfig={splitConfig}
          setSplitConfig={setSplitConfig}
          versionName={versionName}
          setVersionName={setVersionName}
          versionsQuery={versionsQuery}
          processMutation={processMutation}
          createVersionMutation={createVersionMutation}
          confirm={confirm}
        />
      )}
      {confirmationDialog}
    </div>
  );
}
