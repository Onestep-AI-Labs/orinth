"use client";

import { useRef, useState } from "react";
import { FileImage, FolderOpen, Upload, UploadCloud } from "lucide-react";
import { isNlpTask } from "@/features/platform/utils";
import type { DatasetSummary } from "@/types/api";
import { Select } from "@/features/platform/ui";

type FileSystemFileHandleLike = {
  kind: "file";
  name: string;
  getFile: () => Promise<File>;
};
type FileSystemDirectoryHandleLike = {
  kind: "directory";
  name: string;
  values: () => AsyncIterable<FileSystemFileHandleLike | FileSystemDirectoryHandleLike>;
};

declare global {
  interface Window {
    showDirectoryPicker?: () => Promise<FileSystemDirectoryHandleLike>;
  }
}

declare module "react" {
  // `webkitdirectory` is a real, widely supported input attribute that React's
  // types do not carry.
  interface InputHTMLAttributes<T> {
    webkitdirectory?: string;
  }
}

const IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"];
const TEXT_SUFFIXES = [".txt", ".csv", ".jsonl"];

/// Directory picks are matched on extension as well as MIME type: files read
/// out of a folder often arrive with an empty `type`, and filtering on MIME
/// alone silently drops the entire selection.
function isAcceptedFile(file: File, nlp: boolean): boolean {
  const suffixes = nlp ? TEXT_SUFFIXES : IMAGE_SUFFIXES;
  const name = file.name.toLowerCase();
  if (suffixes.some((suffix) => name.endsWith(suffix))) return true;
  return nlp ? false : file.type.startsWith("image/");
}


export function UploadDropCard({
  files,
  setFiles,
  dataset,
  uploadClassId,
  setUploadClassId,
  onUpload,
  pending,
  errors
}: {
  files: File[];
  setFiles: (files: File[]) => void;
  dataset: DatasetSummary;
  uploadClassId: number;
  setUploadClassId: (value: number) => void;
  onUpload: () => void;
  pending: boolean;
  errors: Array<Record<string, string>>;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [folderError, setFolderError] = useState("");
  const nlp = isNlpTask(dataset.task_type);

  function addFiles(fileList: FileList | null, fromFolder = false) {
    if (!fileList) return;
    const acceptedFiles = Array.from(fileList).filter((file) => isAcceptedFile(file, nlp));
    setFiles(acceptedFiles);
    if (acceptedFiles.length > 0) {
      setFolderError("");
      setUploadDialogOpen(true);
    } else if (fromFolder && fileList.length > 0) {
      setFolderError(`No supported ${nlp ? "text" : "image"} files were found in that folder.`);
    }
  }

  async function collectDirectoryImages(directory: FileSystemDirectoryHandleLike): Promise<File[]> {
    const acceptedFiles: File[] = [];
    for await (const entry of directory.values()) {
      if (entry.kind === "directory") {
        acceptedFiles.push(...await collectDirectoryImages(entry));
        continue;
      }
      const file = await entry.getFile();
      if (isAcceptedFile(file, nlp)) {
        acceptedFiles.push(file);
      }
    }
    return acceptedFiles;
  }

  /// Two ways to read a folder, tried in order of quality.
  ///
  /// `showDirectoryPicker` (File System Access API) is Chromium-only. WebKit
  /// and Firefox never implement it, and neither does the desktop app's
  /// WKWebView — which is why this used to dead-end in "not supported in this
  /// browser". The `webkitdirectory` input is the long-standing standard those
  /// engines do support, so it is the fallback.
  ///
  /// The fallback is not feature-detected. `webkitdirectory` is not uniformly
  /// visible on `HTMLInputElement.prototype`, so a strict check can report
  /// "unsupported" on an engine where the picker works. An engine that ignores
  /// the attribute simply opens its ordinary file picker, which still lets the
  /// user select the folder's contents — strictly better than refusing.
  async function selectFolder() {
    setFolderError("");

    if (window.showDirectoryPicker) {
      try {
        const directory = await window.showDirectoryPicker();
        const acceptedFiles = await collectDirectoryImages(directory);
        setFiles(acceptedFiles);
        if (acceptedFiles.length > 0) setUploadDialogOpen(true);
        else setFolderError(`No supported ${nlp ? "text" : "image"} files were found in that folder.`);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setFolderError(error instanceof Error ? error.message : "Could not read the selected folder.");
      }
      return;
    }

    folderInputRef.current?.click();
  }

  function uploadSelectedFiles() {
    setUploadDialogOpen(false);
    onUpload();
  }

  return (
    <div
      className="upload-drop-card"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        addFiles(event.dataTransfer.files);
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        hidden
        multiple
        accept={nlp ? ".txt,.csv,.jsonl,text/plain,text/csv,application/jsonl" : "image/png,image/jpeg,image/jpg,image/webp,image/bmp,image/avif"}
        onChange={(event) => addFiles(event.target.files)}
      />
      <input
        ref={folderInputRef}
        type="file"
        hidden
        multiple
        // No `accept` filter: WebKit applies it per-file when a directory is
        // chosen, which can leave the picker with nothing selectable. The
        // selection is filtered in `addFiles` instead.
        webkitdirectory=""
        onChange={(event) => addFiles(event.target.files, true)}
      />
      <div className="upload-drop-main">
        <div className="upload-icon">
          <UploadCloud size={28} />
        </div>
        <div>
          <strong>Drag and drop {nlp ? "text" : "image"} file(s) to upload</strong>
          <span>{files.length ? `${files.length} ${nlp ? "text" : "image"}${files.length === 1 ? "" : "s"} selected` : `Choose one or more ${nlp ? "text" : "image"} files`}</span>
        </div>
        <div className="upload-actions">
          <button className="secondary-button" type="button" onClick={() => fileInputRef.current?.click()}>
            <FileImage size={16} /> Select file(s)
          </button>
          <button className="secondary-button" type="button" onClick={selectFolder}>
            <FolderOpen size={16} /> Select folder
          </button>
        </div>
      </div>
      <div className="upload-support">
        <span>{nlp ? "Text: .txt, .csv, .jsonl" : "Images: .jpg, .png, .bmp, .webp, .avif"}</span>
        {(dataset.task_type === "classification" || dataset.task_type === "text_classification") && (
          <label className="select-label">
            Class label
            <Select value={uploadClassId} onChange={(event) => setUploadClassId(Number(event.target.value))}>
              {dataset.labels.map((label, index) => (
                <option value={index} key={label}>{label}</option>
              ))}
            </Select>
          </label>
        )}
        <button className="primary-button" onClick={() => setUploadDialogOpen(true)} disabled={files.length === 0 || pending}>
          <Upload size={16} /> Upload batch
        </button>
      </div>
      {errors.length > 0 && <p className="error-text">{errors.length} files could not be uploaded.</p>}
      {folderError && <p className="error-text">{folderError}</p>}
      {uploadDialogOpen && (
        <div className="confirmation-overlay" role="presentation" onMouseDown={() => setUploadDialogOpen(false)}>
          <section
            className="confirmation-dialog upload-review-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="upload-review-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="confirmation-dialog-header">
              <span className="confirmation-dialog-icon" aria-hidden="true">
                <UploadCloud size={19} />
              </span>
              <div>
                <h2 id="upload-review-title">Review {nlp ? "text" : "image"} batch</h2>
                <p>{files.length ? `${files.length} ${nlp ? "text" : "image"}${files.length === 1 ? "" : "s"} ready for upload.` : `Choose ${nlp ? "text" : "image"} files or a folder.`}</p>
              </div>
            </div>
            <div className="upload-review-list">
              {files.slice(0, 8).map((file) => (
                <span key={`${file.name}-${file.size}`}>{file.webkitRelativePath || file.name}</span>
              ))}
              {files.length > 8 && <span>{files.length - 8} more...</span>}
            </div>
            <div className="confirmation-dialog-actions">
              <button className="secondary-button" type="button" onClick={() => fileInputRef.current?.click()}>
                <FileImage size={16} /> Files
              </button>
              <button className="secondary-button" type="button" onClick={selectFolder}>
                <FolderOpen size={16} /> Folder
              </button>
              <button className="secondary-button" type="button" onClick={() => setUploadDialogOpen(false)}>
                Cancel
              </button>
              <button className="primary-button" type="button" onClick={uploadSelectedFiles} disabled={files.length === 0 || pending}>
                <Upload size={16} /> Upload
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
