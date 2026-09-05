"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { Braces, LayoutList, Plus, Save, Trash2 } from "lucide-react";
import { toast } from "@/features/platform/toast";
import { Field, InlineSpinner, Select } from "@/features/platform/ui";
import type { DatasetItemSummary, DatasetSummary, SplitKey } from "@/types/api";

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };
type InstructionDraft = { instruction: string; input: string; output: string };

const CHAT_ROLE_OPTIONS: ChatMessage["role"][] = ["system", "user", "assistant"];

//: How a record field is titled in a column head. Anything not listed keeps its
//: own name, because that is what it is called in the file being trained on.
const FIELD_LABELS: Record<string, string> = {
  instruction: "Instruction",
  input: "Input",
  output: "Output",
  question: "Question",
  context: "Context",
  answer: "Answer",
  text: "Text",
  label: "Label",
  messages: "Messages"
};

/**
 * The two column headings, taken from the fields the excerpts were actually
 * read from.
 *
 * They used to be the constants "Instruction" and "Output" for every non-chat
 * dataset, whatever the records held. An import carrying `question`/`answer`,
 * or a prepared table carrying `text`/`label`, was therefore shown under
 * headings naming fields it did not have. The server reports the fields per
 * item (`preview_fields`), so the head names what the body shows.
 */
export function recordPreviewHeadings(items: DatasetItemSummary[], isChat: boolean): [string, string] {
  if (isChat) return ["First user message", "Last assistant message"];
  const fields = items.find((item) => (item.preview_fields ?? []).length > 0)?.preview_fields ?? [];
  const label = (field: string | undefined, fallback: string) =>
    field ? (FIELD_LABELS[field] ?? field) : fallback;
  return [label(fields[0], "Instruction"), label(fields[1], "Output")];
}

/**
 * The record form for `llm_finetune` datasets: role-aware for chat records,
 * three labelled fields for instruction records, and a raw-JSON escape hatch.
 *
 * It lives on its own because there are two places a record gets edited and
 * only one way a record should be edited. Records opens it in a drawer over the
 * table; Annotate mounts it inline beside the record list, the same shape the
 * vision and NLP editors take there. A second copy of this form would be a
 * second set of rules about which fields survive a save.
 */
export function DatasetRecordEditor({
  dataset,
  mode,
  record,
  loading = false,
  defaultSplit = "unassigned",
  pending = false,
  header,
  onSave,
  onCancel
}: {
  dataset: DatasetSummary;
  mode: "create" | "edit";
  record: Record<string, unknown> | null;
  loading?: boolean;
  defaultSplit?: SplitKey;
  pending?: boolean;
  // Rendered at the head of the toolbar row. Inline in Annotate the editor is
  // the panel, so it carries the panel's own title beside the raw-JSON toggle
  // rather than stacking a title, a gap, and a lone right-aligned button.
  header?: ReactNode;
  onSave: (record: Record<string, unknown>, split: SplitKey) => void;
  onCancel?: () => void;
}) {
  const isChat = dataset.format === "chat_jsonl";
  const [draftSplit, setDraftSplit] = useState<SplitKey>(defaultSplit);
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

  const originalRecord: Record<string, unknown> = mode === "edit" && record ? record : {};
  const knownKeys = isChat ? ["messages"] : ["instruction", "input", "output"];
  const extraKeys = Object.keys(originalRecord).filter((key) => !knownKeys.includes(key));

  useEffect(() => {
    setDraftSplit(defaultSplit);
  }, [defaultSplit]);

  // Hydrate from the record being edited; reset to blank for a new one.
  useEffect(() => {
    setRawMode(false);
    if (mode === "create") {
      setInstruction({ instruction: "", input: "", output: "" });
      setMessages([
        { role: "user", content: "" },
        { role: "assistant", content: "" }
      ]);
      return;
    }
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
  }, [isChat, mode, record]);

  function structuredRecord(): Record<string, unknown> {
    // Carry forward any fields the structured editor doesn't surface (e.g. extra
    // columns from an imported dataset) so editing never silently drops data.
    const next: Record<string, unknown> = {};
    for (const key of extraKeys) next[key] = originalRecord[key];
    if (isChat) {
      next.messages = messages.map((message) => ({ role: message.role, content: message.content }));
    } else {
      next.instruction = instruction.instruction;
      next.output = instruction.output;
      if (instruction.input.trim()) next.input = instruction.input;
    }
    return next;
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

  function save() {
    const built = buildRecord();
    if (!built) return;
    onSave(built, draftSplit);
  }

  return (
    <div className="record-editor">
      <div className="record-editor-toolbar">
        {/* Always rendered, empty or not: it holds the left half of the row so
            the toggle stays right-aligned in the drawer, where there is no
            title beside it. */}
        <div className="record-editor-title">{header}</div>
        <button
          className="secondary-button button-sm"
          onClick={toggleRaw}
          title={rawMode ? "Structured editor" : "Edit raw JSON"}
          type="button"
        >
          {rawMode ? <><LayoutList size={14} /> Structured</> : <><Braces size={14} /> Raw JSON</>}
        </button>
      </div>
      <div className="record-editor-body">
        {mode === "edit" && Object.keys(originalRecord).length > 0 && (
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
            {mode === "edit" && loading ? (
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
                        type="button"
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
                  type="button"
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
      <div className="record-editor-footer">
        <button className="primary-button" onClick={save} disabled={pending} type="button">
          <Save size={16} /> {mode === "create" ? "Add record" : "Save record"}
        </button>
        {onCancel && (
          <button className="secondary-button" onClick={onCancel} type="button">
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
