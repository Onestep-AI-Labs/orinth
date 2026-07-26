"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, ArrowRight, ChevronLeft, ChevronRight, Plus, Save, Trash2, X } from "lucide-react";
import { Badge, EmptyState, Field, InlineSpinner, MutationError, PanelTitle, TableSkeleton } from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import {
  useAddRecordMutation,
  useDeleteRecordsMutation,
  useRecipeRecordsQuery,
  useUpdateRecordMutation
} from "@/features/recipes/hooks";
import type { RecipeRecord, RecipeRead } from "@/types/api";

type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

function excerpt(text: string, limit = 140) {
  const compact = String(text ?? "").replace(/\s+/g, " ").trim();
  return compact.length > limit ? `${compact.slice(0, limit)}…` : compact;
}

function previews(record: Record<string, unknown>, isChat: boolean): [string, string] {
  if (isChat) {
    const messages = Array.isArray(record.messages) ? (record.messages as ChatMessage[]) : [];
    const firstUser = messages.find((message) => message.role === "user")?.content ?? "";
    const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant")?.content ?? "";
    return [excerpt(firstUser), excerpt(lastAssistant)];
  }
  return [excerpt(String(record.instruction ?? "")), excerpt(String(record.output ?? ""))];
}

export function RecipeReviewStep({ recipe, onCommit }: { recipe: RecipeRead; onCommit: () => void }) {
  const isChat = recipe.output_format === "chat_jsonl";
  const [page, setPage] = useState(1);
  const recordsQuery = useRecipeRecordsQuery(recipe.id, page, true);
  const updateMutation = useUpdateRecordMutation(recipe.id, page);
  const addMutation = useAddRecordMutation(recipe.id);
  const deleteMutation = useDeleteRecordsMutation(recipe.id);

  const [drawer, setDrawer] = useState<{ mode: "edit" | "create"; record: RecipeRecord | null } | null>(null);

  const data = recordsQuery.data;
  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;

  return (
    <section className="panel">
      <div className="records-toolbar">
        <PanelTitle icon={<Save size={18} />} title={`Review — ${data?.total ?? recipe.record_count} records`} />
        <div className="records-toolbar-actions">
          {recordsQuery.isFetching && <InlineSpinner label="Refreshing" />}
          <button
            className="secondary-button"
            onClick={() => setDrawer({ mode: "create", record: null })}
          >
            <Plus size={16} /> Add record
          </button>
        </div>
      </div>

      {recipe.warnings.length > 0 && (
        <div className="recipe-warning-panel mt-3">
          <span className="recipe-warning-panel-head">
            <AlertTriangle size={15} /> {recipe.warnings.length} warnings from the last run
          </span>
          <ul>
            {recipe.warnings.slice(0, 6).map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {recordsQuery.isLoading ? (
        <TableSkeleton rows={6} />
      ) : !data || data.records.length === 0 ? (
        <EmptyState
          centered
          label="No records to review."
          icon={<Save size={28} />}
          description="Generate records first, or add one by hand."
        />
      ) : (
        <div className="table-wrap mt-3">
          <table className="records-table">
            <thead>
              <tr>
                <th>{isChat ? "First user message" : "Instruction"}</th>
                <th>{isChat ? "Last assistant message" : "Output"}</th>
                <th className="records-col-split">Origin</th>
                <th className="records-col-actions" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {data.records.map((record) => {
                const [left, right] = previews(record.record, isChat);
                return (
                  <tr
                    key={record.index}
                    className="records-row"
                    onClick={() => setDrawer({ mode: "edit", record })}
                  >
                    <td className="records-cell-excerpt">{left || "—"}</td>
                    <td className="records-cell-excerpt">{right || "—"}</td>
                    <td className="records-col-split">
                      <Badge tone={record.generator === "llm" ? "info" : "neutral"}>{record.generator}</Badge>
                    </td>
                    <td className="records-col-actions">
                      <button
                        className="icon-button"
                        title="Delete record"
                        onClick={(event) => {
                          event.stopPropagation();
                          deleteMutation.mutate([record.index]);
                        }}
                      >
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {data && data.total > data.page_size && (
        <div className="records-pager">
          <button className="secondary-button" onClick={() => setPage((value) => Math.max(1, value - 1))} disabled={page <= 1}>
            <ChevronLeft size={15} /> Prev
          </button>
          <span>Page {page} of {totalPages} — {data.total} records</span>
          <button className="secondary-button" onClick={() => setPage((value) => Math.min(totalPages, value + 1))} disabled={page >= totalPages}>
            Next <ChevronRight size={15} />
          </button>
        </div>
      )}

      <MutationError mutations={[updateMutation, addMutation, deleteMutation]} />

      <div className="recipe-step-footer">
        <span className="text-sm text-ink-subtle">Edit, add, or delete before committing to a dataset.</span>
        <button className="primary-button" onClick={onCommit} disabled={(data?.total ?? recipe.record_count) === 0}>
          Commit <ArrowRight size={16} />
        </button>
      </div>

      {drawer && (
        <RecordDrawer
          isChat={isChat}
          mode={drawer.mode}
          record={drawer.record}
          pending={updateMutation.isPending || addMutation.isPending}
          onClose={() => setDrawer(null)}
          onSave={(record) => {
            if (drawer.mode === "edit" && drawer.record) {
              updateMutation.mutate({ index: drawer.record.index, record }, { onSuccess: () => setDrawer(null) });
            } else {
              addMutation.mutate(record, { onSuccess: () => setDrawer(null) });
            }
          }}
        />
      )}
    </section>
  );
}

function RecordDrawer({
  isChat,
  mode,
  record,
  pending,
  onClose,
  onSave
}: {
  isChat: boolean;
  mode: "edit" | "create";
  record: RecipeRecord | null;
  pending: boolean;
  onClose: () => void;
  onSave: (record: Record<string, unknown>) => void;
}) {
  const [instruction, setInstruction] = useState({ instruction: "", input: "", output: "" });
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "user", content: "" },
    { role: "assistant", content: "" }
  ]);

  useEffect(() => {
    const data = record?.record ?? {};
    if (isChat) {
      const parsed = Array.isArray(data.messages) ? (data.messages as ChatMessage[]) : [];
      setMessages(parsed.length ? parsed : [{ role: "user", content: "" }, { role: "assistant", content: "" }]);
    } else {
      setInstruction({
        instruction: String(data.instruction ?? ""),
        input: String(data.input ?? ""),
        output: String(data.output ?? "")
      });
    }
  }, [record, isChat]);

  function save() {
    if (isChat) {
      const cleaned = messages.filter((message) => message.content.trim());
      if (!cleaned.some((m) => m.role === "user") || !cleaned.some((m) => m.role === "assistant")) {
        toast.error("Chat records need a user and an assistant message");
        return;
      }
      onSave({ messages: cleaned.map((m) => ({ role: m.role, content: m.content })) });
      return;
    }
    if (!instruction.instruction.trim() || !instruction.output.trim()) {
      toast.error("Instruction and output are required");
      return;
    }
    const built: Record<string, unknown> = { instruction: instruction.instruction, output: instruction.output };
    if (instruction.input.trim()) built.input = instruction.input;
    onSave(built);
  }

  return (
    <div className="record-drawer" role="dialog" aria-label="Edit record">
      <div className="record-drawer-header">
        <strong>{mode === "create" ? "New record" : "Edit record"}</strong>
        <button className="icon-button" onClick={onClose} title="Close"><X size={16} /></button>
      </div>
      <div className="record-drawer-body">
        {isChat ? (
          <div className="record-messages">
            {messages.map((message, index) => (
              <div className="record-message" key={index}>
                <div className="record-message-head">
                  <select
                    value={message.role}
                    onChange={(event) =>
                      setMessages((current) =>
                        current.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, role: event.target.value as ChatMessage["role"] } : entry
                        )
                      )
                    }
                  >
                    {(["system", "user", "assistant"] as ChatMessage["role"][]).map((role) => (
                      <option key={role} value={role}>{role}</option>
                    ))}
                  </select>
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
      </div>
      <div className="record-drawer-footer">
        <button className="primary-button" onClick={save} disabled={pending}>
          <Save size={16} /> {mode === "create" ? "Add record" : "Save record"}
        </button>
        <button className="secondary-button" onClick={onClose}>Cancel</button>
      </div>
    </div>
  );
}
