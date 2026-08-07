"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ShieldAlert, UploadCloud, X } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Badge, Button, Field, InlineSpinner, Select } from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import type { components } from "@/types/generated/api";

type UploadOption = components["schemas"]["ModelUploadOption"];
type UploadResult = components["schemas"]["ModelUploadResult"];

export function UploadModelDialog({
  projectId,
  onClose,
  onUploaded
}: {
  projectId: string;
  onClose: () => void;
  onUploaded: (result: UploadResult) => void;
}) {
  const optionsQuery = useQuery({
    queryKey: ["models", "upload-options"],
    queryFn: api.modelUploadOptions
  });
  const options = useMemo(() => optionsQuery.data ?? [], [optionsQuery.data]);

  const [family, setFamily] = useState("");
  const [name, setName] = useState("");
  const [fieldValues, setFieldValues] = useState<Record<string, string>>({});
  const [files, setFiles] = useState<Record<string, File>>({});

  const active = useMemo(() => options.find((option) => option.family === family), [options, family]);

  // Default to the first family once options land.
  useEffect(() => {
    if (!family && options.length > 0) setFamily(options[0].family);
  }, [family, options]);

  // Reset per-family inputs whenever the family changes.
  useEffect(() => {
    setFieldValues({});
    setFiles({});
  }, [family]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const upload = useMutation({
    mutationFn: (form: FormData) => api.uploadModel(form),
    onSuccess: (result) => {
      toast.success(`Uploaded "${result.model.name}"`);
      result.warnings.forEach((warning) => toast.error(warning));
      onUploaded(result);
    },
    onError: (error: Error) => toast.error(error.message)
  });

  const missingFile = active?.files.some((slot) => slot.required && !files[slot.key]) ?? true;
  const missingField =
    active?.fields.some((field) => field.required && !fieldValues[field.key]?.trim()) ?? false;
  const canSubmit = Boolean(active) && name.trim().length > 0 && !missingFile && !missingField;

  function submit() {
    if (!active || !canSubmit) return;
    const form = new FormData();
    form.set("family", active.family);
    form.set("name", name.trim());
    form.set("project_id", projectId);
    for (const field of active.fields) {
      const value = fieldValues[field.key]?.trim();
      if (value) form.set(field.key, value);
    }
    for (const slot of active.files) {
      const file = files[slot.key];
      if (file) form.set(slot.key, file);
    }
    upload.mutate(form);
  }

  return (
    <div className="confirmation-overlay" role="presentation" onMouseDown={onClose}>
      <section
        className="upload-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="upload-dialog-header">
          <div>
            <span className="page-eyebrow">Custom weights</span>
            <h2 id="upload-dialog-title">Upload model</h2>
          </div>
          <button className="icon-button" type="button" aria-label="Close" onClick={onClose}>
            <X size={16} />
          </button>
        </header>

        {optionsQuery.isLoading ? (
          <InlineSpinner label="Loading upload options" />
        ) : (
          <div className="upload-dialog-body">
            <Field label="Model family">
              <div className="upload-family-row" role="radiogroup" aria-label="Model family">
                {options.map((option) => (
                  <button
                    key={option.family}
                    type="button"
                    role="radio"
                    aria-checked={option.family === family}
                    className={`upload-family-chip${option.family === family ? " is-active" : ""}`}
                    onClick={() => setFamily(option.family)}
                  >
                    {option.label}
                    {!option.servable && <Badge tone="warn">Gated</Badge>}
                  </button>
                ))}
              </div>
            </Field>

            {active && <FamilyForm
              option={active}
              name={name}
              onName={setName}
              fieldValues={fieldValues}
              onField={(key, value) => setFieldValues((prev) => ({ ...prev, [key]: value }))}
              files={files}
              onFile={(key, file) => setFiles((prev) => ({ ...prev, [key]: file }))}
            />}
          </div>
        )}

        <footer className="upload-dialog-actions">
          <p className="upload-dialog-note">Structural validation only — deep checks run on first use.</p>
          <div className="upload-dialog-buttons">
            <Button variant="secondary" onClick={onClose}>Cancel</Button>
            <Button onClick={submit} disabled={!canSubmit || upload.isPending}>
              <UploadCloud size={16} /> {upload.isPending ? "Uploading…" : "Upload"}
            </Button>
          </div>
        </footer>
      </section>
    </div>
  );
}

function FamilyForm({
  option,
  name,
  onName,
  fieldValues,
  onField,
  files,
  onFile
}: {
  option: UploadOption;
  name: string;
  onName: (value: string) => void;
  fieldValues: Record<string, string>;
  onField: (key: string, value: string) => void;
  files: Record<string, File>;
  onFile: (key: string, file: File) => void;
}) {
  return (
    <div className="upload-form">
      <p className="upload-family-description">{option.description}</p>
      {option.gate_note && (
        <p className="upload-gate-note"><AlertTriangle size={14} /> {option.gate_note}</p>
      )}
      {option.security_note && (
        <p className="upload-security-note"><ShieldAlert size={14} /> {option.security_note}</p>
      )}

      <Field label="Display name">
        <input value={name} onChange={(event) => onName(event.target.value)} placeholder="My custom model" autoFocus />
      </Field>

      {option.files.map((slot) => (
        <Field key={slot.key} label={slot.label} hint={slot.accept.join(" · ")}>
          <input
            type="file"
            accept={slot.accept.join(",")}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onFile(slot.key, file);
            }}
          />
          {files[slot.key] && <span className="upload-file-name">{files[slot.key].name}</span>}
        </Field>
      ))}

      {option.fields.map((field) => (
        <Field key={field.key} label={field.label} hint={field.help ?? undefined}>
          {field.type === "select" ? (
            <Select value={fieldValues[field.key] ?? ""} onChange={(event) => onField(field.key, event.target.value)}>
              <option value="" disabled>Select…</option>
              {(field.options ?? []).map((choice) => (
                <option key={choice} value={choice}>{choice.replaceAll("_", " ")}</option>
              ))}
            </Select>
          ) : (
            <input
              type={field.type === "number" ? "number" : "text"}
              value={fieldValues[field.key] ?? ""}
              placeholder={field.placeholder ?? undefined}
              onChange={(event) => onField(field.key, event.target.value)}
            />
          )}
        </Field>
      ))}
    </div>
  );
}
