"use client";

import { useEffect, useRef, useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { Braces, ChevronLeft, ChevronRight, FileJson, LayoutList, Plus, Save, Trash2, Upload, X } from "lucide-react";
import { toast } from "@/features/platform/toast";
import { SPLIT_FILTERS } from "@/features/platform/constants";
import type {
  useCreateDatasetRecordMutation,
  useDeleteDatasetItemsMutation,
  useSaveDatasetRecordMutation,
  useUploadDatasetRecordsMutation
} from "@/features/datasets/hooks";
import { Badge, EmptyState, Field, InlineSpinner, Select } from "@/features/platform/ui";
import type {
  DatasetItemDetail,
  DatasetItemPage,
  DatasetItemSummary,
  DatasetSplitFilter,
  DatasetSummary,
  SplitKey
} from "@/types/api";

type ChatMessage = { role: "system" | "user" | "assistant"; content: string };
type InstructionDraft = { instruction: string; input: string; output: string };

const CHAT_ROLE_OPTIONS: ChatMessage["role"][] = ["system", "user", "assistant"];

/**
 * Records tab for `llm_finetune` datasets (phase 10): a paginated table with an
 * edit drawer that is role-aware for chat records and three labeled fields for
 * instruction records. Replaces the Images/Annotate tabs, which have no meaning
 * for record data.
 */
export function DatasetRecordsTab({
  dataset,
  items,
  itemsQuery,
  itemPage,
  split,
  setSplit,
  imagesPerPage,
  imagePage,
  setImagePage,
  detailQuery,
  onSelectItem,
  selectedItemId,
  createRecordMutation,
  saveRecordMutation,
  uploadRecordsMutation,
  deleteItemsMutation,
  confirm
}: {
  dataset: DatasetSummary;
  items: DatasetItemSummary[];
  itemsQuery: UseQueryResult<DatasetItemPage>;
  itemPage: DatasetItemPage;
  split: DatasetSplitFilter;
  setSplit: (value: DatasetSplitFilter) => void;
  imagesPerPage: number;
  imagePage: number;
  setImagePage: (value: number) => void;
  detailQuery: UseQueryResult<DatasetItemDetail>;
  onSelectItem: (item: DatasetItemSummary) => void;
  selectedItemId: string;
  createRecordMutation: ReturnType<typeof useCreateDatasetRecordMutation>;
  saveRecordMutation: ReturnType<typeof useSaveDatasetRecordMutation>;
  uploadRecordsMutation: ReturnType<typeof useUploadDatasetRecordsMutation>;
  deleteItemsMutation: ReturnType<typeof useDeleteDatasetItemsMutation>;
  confirm: (options: { title: string; message: string; confirmLabel: string; onConfirm: () => void }) => void;
}) {
  const isChat = dataset.format === "chat_jsonl";
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mode, setMode] = useState<"create" | "edit">("edit");
  const [draftSplit, setDraftSplit] = useState<SplitKey>("unassigned");
  const [instruction, setInstruction] = useState<InstructionDraft>({ instruction: "", input: "", output: "" });
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "user", content: "" },
    { role: "assistant", content: "" }
  ]);
  // Raw-JSON mode lets a record be edited as whatever structure the source
  // actually has, not just the known instruction/chat fields — imported rows can
  // carry extra columns, and this keeps the editor honest to the data.
  const [rawMode, setRawMode] = useState(false);
  const [rawText, setRawText] = useState("");

  const originalRecord: Record<string, unknown> =
    mode === "edit" && detailQuery.data?.record ? detailQuery.data.record : {};
  const knownKeys = isChat ? ["messages"] : ["instruction", "input", "output"];
  const extraKeys = Object.keys(originalRecord).filter((key) => !knownKeys.includes(key));

  // When an existing record's detail loads, hydrate the drawer from it.
  useEffect(() => {
    if (mode !== "edit" || !drawerOpen) return;
    const record = detailQuery.data?.record;
    if (!record) return;
    if (isChat) {
      const parsed = Array.isArray(record.messages) ? (record.messages as ChatMessage[]) : [];
      setMessages(parsed.length ? parsed : [{ role: "user", content: "" }]);
    } else {
      setInstruction({
        instruction: String(record.instruction ?? ""),
        input: String(record.input ?? ""),
        output: String(record.output ?? "")
      });
    }
  }, [detailQuery.data, drawerOpen, isChat, mode]);

  const lastPage = Math.max(0, Math.ceil(itemPage.total / imagesPerPage) - 1);

  function openEdit(item: DatasetItemSummary) {
    setMode("edit");
    setRawMode(false);
    setDrawerOpen(true);
    onSelectItem(item);
  }

  function openCreate() {
    setMode("create");
    setRawMode(false);
    setDrawerOpen(true);
    setDraftSplit(split === "all" ? "unassigned" : (split as SplitKey));
    setInstruction({ instruction: "", input: "", output: "" });
    setMessages([
      { role: "user", content: "" },
      { role: "assistant", content: "" }
    ]);
  }

  function structuredRecord(): Record<string, unknown> {
    // Carry forward any fields the structured editor doesn't surface (e.g. extra
    // columns from an imported dataset) so editing never silently drops data.
    const record: Record<string, unknown> = {};
    for (const key of extraKeys) record[key] = originalRecord[key];
    if (isChat) {
      record.messages = messages.map((message) => ({ role: message.role, content: message.content }));
    } else {
      record.instruction = instruction.instruction;
      record.output = instruction.output;
      if (instruction.input.trim()) record.input = instruction.input;
    }
    return record;
  }

  function toggleRaw() {
    if (!rawMode) setRawText(JSON.stringify(structuredRecord(), null, 2));
    setRawMode((value) => !value);
  }

  function buildRecord(): Record<string, unknown> | null {
    if (!rawMode) return structuredRecord();
    try {
      const parsed = JSON.parse(rawText);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        toast.error("Record JSON must be an object");
        return null;
      }
      return parsed as Record<string, unknown>;
    } catch {
      toast.error("Invalid JSON — check the record body");
      return null;
    }
  }

  function saveDraft() {
    const record = buildRecord();
    if (!record) return;
    if (mode === "create") {
      createRecordMutation.mutate(
        { datasetId: dataset.id, split: draftSplit, record },
        { onSuccess: () => setDrawerOpen(false) }
      );
    } else if (selectedItemId && detailQuery.data) {
      saveRecordMutation.mutate(
        { datasetId: dataset.id, split: detailQuery.data.split, itemId: selectedItemId, record },
        { onSuccess: () => setDrawerOpen(false) }
      );
    }
  }

  function deleteRecord(item: DatasetItemSummary) {
    confirm({
      title: "Delete record?",
      message: "This removes the record and rewrites the split's data.jsonl. This cannot be undone.",
      confirmLabel: "Delete record",
      onConfirm: () => deleteItemsMutation.mutate({ datasetId: dataset.id, items: [{ id: item.id, split: item.split }] })
    });
  }

  function uploadFile(file: File | undefined) {
    if (!file) return;
    const form = new FormData();
    form.append("split", split === "all" ? "unassigned" : split);
    form.append("file", file);
    uploadRecordsMutation.mutate({ datasetId: dataset.id, form });
  }

  const pending = createRecordMutation.isPending || saveRecordMutation.isPending;

  return (
    <section className="panel records-panel">
      <div className="records-toolbar">
        <div className="segmented-control">
          {SPLIT_FILTERS.map((filter) => (
            <button
              key={filter}
              type="button"
              className={split === filter ? "segmented-active" : ""}
              onClick={() => setSplit(filter)}
            >
              {filter}
            </button>
          ))}
        </div>
        <div className="records-toolbar-actions">
          {(itemsQuery.isFetching || uploadRecordsMutation.isPending) && (
            <InlineSpinner label={uploadRecordsMutation.isPending ? "Uploading" : "Refreshing"} />
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".jsonl,.json,.csv"
            hidden
            onChange={(event) => {
              uploadFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
          <button
            className="secondary-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadRecordsMutation.isPending}
            title="Upload a .jsonl, .json, or .csv file of records"
          >
            <Upload size={16} /> Upload file
          </button>
          <button className="primary-button" onClick={openCreate}>
            <Plus size={16} /> New record
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <EmptyState
          label="No records yet."
          icon={<FileJson size={28} />}
          centered
          description="Add a record by hand, upload a .jsonl/.csv file, or import from the HuggingFace Hub."
        />
      ) : (
        <div className="table-wrap">
          <table className="records-table">
            <thead>
              <tr>
                <th>{isChat ? "First user message" : "Instruction"}</th>
                <th>{isChat ? "Last assistant message" : "Output"}</th>
                <th className="records-col-num">Tokens</th>
                <th className="records-col-split">Split</th>
                <th className="records-col-actions" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={`${item.split}-${item.id}`} onClick={() => openEdit(item)} className="records-row">
                  <td className="records-cell-excerpt">{item.text_preview || "—"}</td>
                  <td className="records-cell-excerpt">{item.output_preview || "—"}</td>
                  <td className="records-col-num">{item.token_estimate}</td>
                  <td className="records-col-split">
                    <Badge tone="neutral">{item.split}</Badge>
                  </td>
                  <td className="records-col-actions">
                    <button
                      className="icon-button"
                      title="Delete record"
                      onClick={(event) => {
                        event.stopPropagation();
                        deleteRecord(item);
                      }}
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {itemPage.total > imagesPerPage && (
        <div className="records-pager">
          <button className="secondary-button" onClick={() => setImagePage(Math.max(0, imagePage - 1))} disabled={imagePage <= 0}>
            <ChevronLeft size={15} /> Prev
          </button>
          <span>
            Page {imagePage + 1} of {lastPage + 1} — {itemPage.total} records
          </span>
          <button className="secondary-button" onClick={() => setImagePage(Math.min(lastPage, imagePage + 1))} disabled={imagePage >= lastPage}>
            Next <ChevronRight size={15} />
          </button>
        </div>
      )}

      {drawerOpen && (
        <div className="record-drawer" role="dialog" aria-label="Edit record">
          <div className="record-drawer-header">
            <strong>{mode === "create" ? "New record" : "Edit record"}</strong>
            <div className="record-drawer-header-actions">
              <button
                className="secondary-button button-sm"
                onClick={toggleRaw}
                title={rawMode ? "Structured editor" : "Edit raw JSON"}
              >
                {rawMode ? <><LayoutList size={14} /> Structured</> : <><Braces size={14} /> Raw JSON</>}
              </button>
              <button className="icon-button" onClick={() => setDrawerOpen(false)} title="Close">
                <X size={16} />
              </button>
            </div>
          </div>
          <div className="record-drawer-body">
            {mode === "edit" && (extraKeys.length > 0 || Object.keys(originalRecord).length > 0) && (
              <div className="record-fields">
                <span className="record-fields-label">Fields</span>
                <div className="record-fields-chips">
                  {Object.keys(originalRecord).map((key) => (
                    <span key={key} className={`record-field-chip ${knownKeys.includes(key) ? "" : "record-field-chip-extra"}`}>
                      {key}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {rawMode ? (
              <Field label="Record JSON">
                <textarea
                  className="record-raw-editor"
                  value={rawText}
                  rows={16}
                  spellCheck={false}
                  onChange={(event) => setRawText(event.target.value)}
                  placeholder='{ "instruction": "...", "output": "..." }'
                />
              </Field>
            ) : (
              <>
            {mode === "create" && (
              <Field label="Split">
                <Select value={draftSplit} onChange={(event) => setDraftSplit(event.target.value as SplitKey)}>
                  {(["unassigned", "train", "valid", "test"] as SplitKey[]).map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </Select>
              </Field>
            )}
            {mode === "edit" && detailQuery.isLoading ? (
              <InlineSpinner label="Loading record" />
            ) : isChat ? (
              <div className="record-messages">
                {messages.map((message, index) => (
                  <div className="record-message" key={index}>
                    <div className="record-message-head">
                      <Select
                        value={message.role}
                        onChange={(event) =>
                          setMessages((current) =>
                            current.map((entry, entryIndex) =>
                              entryIndex === index ? { ...entry, role: event.target.value as ChatMessage["role"] } : entry
                            )
                          )
                        }
                      >
                        {CHAT_ROLE_OPTIONS.map((role) => (
                          <option key={role} value={role}>{role}</option>
                        ))}
                      </Select>
                      <button
                        className="icon-button"
                        title="Remove message"
                        onClick={() => setMessages((current) => current.filter((_entry, entryIndex) => entryIndex !== index))}
                        disabled={messages.length <= 1}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    <textarea
                      value={message.content}
                      rows={3}
                      onChange={(event) =>
                        setMessages((current) =>
                          current.map((entry, entryIndex) =>
                            entryIndex === index ? { ...entry, content: event.target.value } : entry
                          )
                        )
                      }
                      placeholder="Message content"
                    />
                  </div>
                ))}
                <button
                  className="secondary-button"
                  onClick={() => setMessages((current) => [...current, { role: "user", content: "" }])}
                >
                  <Plus size={15} /> Add message
                </button>
              </div>
            ) : (
              <>
                <Field label="Instruction">
                  <textarea
                    value={instruction.instruction}
                    rows={3}
                    onChange={(event) => setInstruction((current) => ({ ...current, instruction: event.target.value }))}
                    placeholder="What the model should do"
                  />
                </Field>
                <Field label="Input (optional)">
                  <textarea
                    value={instruction.input}
                    rows={2}
                    onChange={(event) => setInstruction((current) => ({ ...current, input: event.target.value }))}
                    placeholder="Optional context"
                  />
                </Field>
                <Field label="Output">
                  <textarea
                    value={instruction.output}
                    rows={4}
                    onChange={(event) => setInstruction((current) => ({ ...current, output: event.target.value }))}
                    placeholder="Target response"
                  />
                </Field>
              </>
            )}
              </>
            )}
          </div>
          <div className="record-drawer-footer">
            <button className="primary-button" onClick={saveDraft} disabled={pending}>
              <Save size={16} /> {mode === "create" ? "Add record" : "Save record"}
            </button>
            <button className="secondary-button" onClick={() => setDrawerOpen(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
