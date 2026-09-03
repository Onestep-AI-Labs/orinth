"use client";

/* eslint-disable @next/next/no-img-element */

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { FileJson, FileText, ImageIcon, Table2 } from "lucide-react";
import { useProject } from "@/components/app-shell";
import { DatasetAnnotateTab } from "@/features/datasets/annotate-tab";
import { DatasetCatalogView } from "@/features/datasets/catalog-view";
import { DatasetConfigTab } from "@/features/datasets/config-tab";
import { EdaPanel } from "@/features/datasets/dataset-components";
import { DatasetDetailHeader } from "@/features/datasets/detail-header";
import { DatasetDataGrid } from "@/features/datasets/data-grid";
import { DatasetDetailTabs, type DatasetDetailTab } from "@/features/datasets/detail-tabs";
import { DatasetOverviewTab } from "@/features/datasets/prep/overview-tab";
import { DatasetPrepareTab } from "@/features/datasets/prepare-tab";
import { HubImportPanel } from "@/features/datasets/hub-import-panel";
import { DatasetRecordsTab } from "@/features/datasets/records-tab";
import {
  useBulkLabelDatasetItemsMutation,
  useCloneDatasetMutation,
  useCreateDatasetMutation,
  useCreateDatasetRecordMutation,
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
  useSaveDatasetRecordMutation,
  useUpdateDatasetMutation,
  useUploadDatasetMutation,
  useUploadDatasetRecordsMutation
} from "@/features/datasets/hooks";
import { DatasetImagesTab } from "@/features/datasets/images-tab";
import { SPLITS } from "@/features/platform/constants";
import {
  allowedTaskTypesForProject,
  defaultPreprocessConfig,
  defaultSplitConfig,
  isLlmTask,
  isNlpTask,
  preprocessFromDataset,
  splitConfigFromDataset
} from "@/features/platform/utils";
import { MutationError, PageSkeleton, useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetItemPage, DatasetItemSummary, DatasetPreprocessConfig, DatasetSplitConfig, DatasetSplitFilter, DatasetSummary, SplitKey, TaskType } from "@/types/api";

export function DatasetPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { projectId, project } = useProject();
  const datasetParam = searchParams?.get("dataset") ?? "";
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [showDuplicate, setShowDuplicate] = useState(false);
  const [showDatasetOptions, setShowDatasetOptions] = useState(false);
  const [catalogMenuDatasetId, setCatalogMenuDatasetId] = useState("");
  const [catalogRenameDatasetId, setCatalogRenameDatasetId] = useState("");
  const [showRename, setShowRename] = useState(false);
  const [detailTab, setDetailTab] = useState<DatasetDetailTab>("overview");
  // The Data tab shows one dataset two ways. The table is the default because it
  // is the only view that works for every modality and the only one that answers
  // "what is in here" without scrolling; the gallery/records view is where the
  // annotation editor and the record drawer live, and those are still the right
  // tools for editing one item properly.
  const [dataView, setDataView] = useState<"table" | "detail">("table");
  const [split, setSplit] = useState<DatasetSplitFilter>("all");
  const [imagePage, setImagePage] = useState(0);
  const [imagesPerPage, setImagesPerPage] = useState(50);
  const [classFilter, setClassFilter] = useState("");
  const [selectedItemId, setSelectedItemId] = useState("");
  const [selectedItemSplit, setSelectedItemSplit] = useState<SplitKey | "">("");
  const [selectedItemIds, setSelectedItemIds] = useState<string[]>([]);
  const [selectedItemSplits, setSelectedItemSplits] = useState<Record<string, SplitKey>>({});
  const [newDatasetName, setNewDatasetName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [labelDraft, setLabelDraft] = useState("");
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
  const allowedDatasetTasks = useMemo(() => allowedTaskTypesForProject(project), [project]);
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
  const selectedDatasetIsLlm = Boolean(selectedDataset && isLlmTask(selectedDataset.task_type));
  const ItemIcon = selectedDatasetIsLlm ? FileJson : selectedDatasetIsNlp ? FileText : ImageIcon;
  const canImportHub = allowedDatasetTasks.includes("llm_finetune");
  const itemsQuery = useDatasetItemsQuery(selectedDataset?.id, split, classFilter, imagePage, imagesPerPage);
  const itemPage = useMemo<DatasetItemPage>(
    () => itemsQuery.data ?? { items: [], total: 0, limit: imagesPerPage, offset: imagePage * imagesPerPage },
    [imagePage, imagesPerPage, itemsQuery.data]
  );
  const items = useMemo(() => itemPage.items, [itemPage.items]);
  const selectedItemSummary = useMemo(
    // Guard on dataset_id too: the items query keeps previous data across a
    // dataset switch, and a leftover item from the old dataset would otherwise
    // fire a detail fetch against the new one (a spurious 404).
    () =>
      items.find(
        (item) =>
          item.id === selectedItemId &&
          item.split === selectedItemSplit &&
          item.dataset_id === selectedDatasetId
      ) ?? null,
    [items, selectedItemId, selectedItemSplit, selectedDatasetId]
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

  const createMutation = useCreateDatasetMutation({ openDataset, setNewDatasetName, catalogQuery });
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
    setSelectedItemIds,
    setSelectedItemSplits,
    setSelectedItemId,
    setSelectedItemSplit,
    itemsQuery,
    catalogQuery,
    edaQuery
  });
  const bulkLabelMutation = useBulkLabelDatasetItemsMutation({ itemsQuery, detailQuery, catalogQuery, edaQuery });
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
    setSelectedItemSplits,
    setSelectedItemId,
    setSelectedItemSplit,
    catalogQuery,
    itemsQuery,
    edaQuery
  });
  const moveItemsMutation = useMoveDatasetItemsMutation({
    setSelectedItemIds,
    setSelectedItemSplits,
    itemsQuery,
    catalogQuery,
    edaQuery
  });
  const createRecordMutation = useCreateDatasetRecordMutation({ itemsQuery, catalogQuery, edaQuery });
  const saveRecordMutation = useSaveDatasetRecordMutation({ itemsQuery, catalogQuery, edaQuery, detailQuery });
  const uploadRecordsMutation = useUploadDatasetRecordsMutation({ itemsQuery, catalogQuery, edaQuery });

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
    setSelectedItemSplits({});
    setSelectedItemId("");
    setSelectedItemSplit("");
    setPreviewUrl(null);
    setPreviewText(null);
    setShowDatasetOptions(false);
    setShowRename(false);
    setShowDuplicate(false);
    setCatalogRenameDatasetId("");
    // Overview leads: a dataset the agent just touched should say what it did
    // and whether it worked before showing rows.
    setDetailTab("overview");
    setSplit("all");
    setImagePage(0);
  }, [selectedDataset]);

  useEffect(() => {
    setSelectedItemIds([]);
    setSelectedItemSplits({});
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
    // Empty means the project has not resolved yet — leave the task alone
    // rather than snapping it to a default the project may not allow.
    if (allowedDatasetTasks.length === 0) return;
    if (!allowedDatasetTasks.includes(taskType)) {
      setTaskType(allowedDatasetTasks[0]);
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
    // Annotating is no longer a separate tab; the editor sits beside the grid in
    // Data, so selecting an item is already all the navigation there is.
    if (openAnnotate) setDetailTab("data");
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
        canImportHub={canImportHub}
        hubImportPanel={
          <HubImportPanel
            projectId={projectId}
            openDataset={openDataset}
            catalogQuery={catalogQuery}
            // Importing navigates into the new dataset, so there is nothing left
            // on this screen for the panel to close.
            onImported={() => undefined}
          />
        }
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
    selectedItemSplits,
    setSelectedItemSplits,
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
        isLlm={selectedDatasetIsLlm}
        readiness={selectedDataset.readiness}
      />
      {detailTab === "overview" ? (
        <DatasetOverviewTab dataset={selectedDataset} onGoToData={() => setDetailTab("data")} />
      ) : detailTab === "data" ? (
        <div className="space-y-5">
          <div className="segmented-control" role="tablist" aria-label="Data view">
            <button
              type="button"
              role="tab"
              aria-selected={dataView === "table"}
              className={dataView === "table" ? "segmented-active" : ""}
              onClick={() => setDataView("table")}
            >
              <Table2 size={15} /> Table
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={dataView === "detail"}
              className={dataView === "detail" ? "segmented-active" : ""}
              onClick={() => setDataView("detail")}
            >
              <ItemIcon size={15} /> {selectedDatasetIsLlm ? "Records" : "Gallery"}
            </button>
          </div>
          {dataView === "table" ? (
            <DatasetDataGrid dataset={selectedDataset} />
          ) : selectedDatasetIsLlm ? (
          <DatasetRecordsTab
            dataset={selectedDataset}
            items={items}
            itemsQuery={itemsQuery}
            itemPage={itemPage}
            split={split}
            setSplit={setSplit}
            imagesPerPage={imagesPerPage}
            imagePage={imagePage}
            setImagePage={setImagePage}
            detailQuery={detailQuery}
            onSelectItem={selectDatasetItem}
            selectedItemId={selectedItemId}
            createRecordMutation={createRecordMutation}
            saveRecordMutation={saveRecordMutation}
            uploadRecordsMutation={uploadRecordsMutation}
            deleteItemsMutation={deleteItemsMutation}
            confirm={confirm}
          />
        ) : (
          <div className="space-y-5">
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
              bulkClassId={bulkClassId}
              setBulkClassId={setBulkClassId}
              bulkLabelMutation={bulkLabelMutation}
            />
            {/* Labelling used to be its own tab showing a second copy of the same
                grid. It is the same work on the same items, so it belongs under
                the items: the editor appears once something is selected. */}
            <DatasetAnnotateTab
              {...sharedTabProps}
              bulkClassId={bulkClassId}
              setBulkClassId={setBulkClassId}
              bulkLabelMutation={bulkLabelMutation}
              catalogQuery={catalogQuery}
              detailQuery={detailQuery}
              edaQuery={edaQuery}
            />
          </div>
          )}
        </div>
      ) : (
        <DatasetPrepareTab
          dataset={selectedDataset}
          nlp={selectedDatasetIsNlp || selectedDatasetIsLlm}
          split={split}
          setSplit={setSplit}
          edaQuery={edaQuery}
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
    </div>
  );
}
