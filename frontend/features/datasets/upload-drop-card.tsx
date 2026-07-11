"use client";

import { useRef, useState } from "react";
import { FileImage, FolderOpen, Upload, UploadCloud } from "lucide-react";
import type { DatasetSummary } from "@/types/api";

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
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [folderError, setFolderError] = useState("");

  function addFiles(fileList: FileList | null) {
    if (!fileList) return;
    const imageFiles = Array.from(fileList).filter((file) => file.type.startsWith("image/"));
    setFiles(imageFiles);
    if (imageFiles.length > 0) setUploadDialogOpen(true);
  }

  async function collectDirectoryImages(directory: FileSystemDirectoryHandleLike): Promise<File[]> {
    const imageFiles: File[] = [];
    for await (const entry of directory.values()) {
      if (entry.kind === "directory") {
        imageFiles.push(...await collectDirectoryImages(entry));
        continue;
      }
      const file = await entry.getFile();
      if (file.type.startsWith("image/")) imageFiles.push(file);
    }
    return imageFiles;
  }

  async function selectFolder() {
    setFolderError("");
    if (!window.showDirectoryPicker) {
      setFolderError("Folder selection is not supported in this browser. Select files or drag image files instead.");
      return;
    }
    try {
      const directory = await window.showDirectoryPicker();
      const imageFiles = await collectDirectoryImages(directory);
      setFiles(imageFiles);
      if (imageFiles.length > 0) setUploadDialogOpen(true);
      if (imageFiles.length === 0) setFolderError("No supported image files were found in that folder.");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setFolderError(error instanceof Error ? error.message : "Could not read the selected folder.");
    }
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
        accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp,image/avif"
        onChange={(event) => addFiles(event.target.files)}
      />
      <div className="upload-drop-main">
        <div className="upload-icon">
          <UploadCloud size={28} />
        </div>
        <div>
          <strong>Drag and drop image file(s) to upload</strong>
          <span>{files.length ? `${files.length} image${files.length === 1 ? "" : "s"} selected` : "Choose one or more image files"}</span>
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
        <span>Images: .jpg, .png, .bmp, .webp, .avif</span>
        {dataset.task_type === "classification" && (
          <label className="select-label">
            Class label
            <select value={uploadClassId} onChange={(event) => setUploadClassId(Number(event.target.value))}>
              {dataset.labels.map((label, index) => (
                <option value={index} key={label}>{label}</option>
              ))}
            </select>
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
                <h2 id="upload-review-title">Review image batch</h2>
                <p>{files.length ? `${files.length} image${files.length === 1 ? "" : "s"} ready for upload.` : "Choose image files or a folder."}</p>
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
