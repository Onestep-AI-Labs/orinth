"use client";

import { useRef, useState } from "react";
import { FileText, FolderOpen, Sparkles, Upload } from "lucide-react";
import {
  useDatasetPrepStatusQuery,
  useIngestDatasetMutation,
  useRunPrepMutation
} from "@/features/datasets/prep/prep-hooks";
import { prepJob } from "@/features/datasets/prep/prep-progress";
import {
  usePublishForegroundJob,
  type ForegroundJob
} from "@/features/platform/foreground-job";
import { Button, Field, MutationError } from "@/features/platform/ui";

/**
 * The way a dataset starts: drop files, declare nothing.
 *
 * The old flow asked for a domain, a task, a record schema and a list of class
 * labels before it would accept a single byte. It had to: once uploaded, files
 * were flattened into one directory and the only remaining evidence of what they
 * were had been destroyed.
 *
 * So the important thing this card does is not the drop target — it is sending
 * `relative_paths` alongside the files. `data.yaml` beside `train/labels/` means
 * object detection; sibling `normal/` and `kista/` folders mean classification
 * with those two classes. Structure is the signal, and it only survives if the
 * client bothers to send it.
 */

type Picked = { file: File; path: string };

type FileHandleLike = { kind: "file"; name: string; getFile: () => Promise<File> };
type DirectoryHandleLike = {
  kind: "directory";
  name: string;
  values: () => AsyncIterable<FileHandleLike | DirectoryHandleLike>;
};

declare global {
  interface Window {
    showDirectoryPicker?: () => Promise<DirectoryHandleLike>;
  }
}

//: Anything a dataset might arrive as. Deliberately broad — the point of ingest
//: is that the task type is not known yet, so it cannot be filtered by one.
const ACCEPTED = [
  ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif",
  ".txt", ".csv", ".tsv", ".jsonl", ".json", ".parquet", ".yaml", ".yml"
];

function isAccepted(file: File): boolean {
  const name = file.name.toLowerCase();
  return ACCEPTED.some((suffix) => name.endsWith(suffix));
}

/** Walk a directory handle, keeping each file's path relative to the root. */
async function walk(directory: DirectoryHandleLike, prefix = ""): Promise<Picked[]> {
  const found: Picked[] = [];
  for await (const entry of directory.values()) {
    const path = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.kind === "directory") {
      found.push(...(await walk(entry, path)));
      continue;
    }
    const file = await entry.getFile();
    if (isAccepted(file)) found.push({ file, path });
  }
  return found;
}

//: How many picked files are named before the list collapses to a count. Enough
//: to recognise a folder by its contents, short enough that the panel does not
//: become the file list.
const NAMED_FILES = 6;

function formatBytes(total: number): string {
  if (total >= 1_000_000_000) return `${(total / 1_000_000_000).toFixed(1)} GB`;
  if (total >= 1_000_000) return `${(total / 1_000_000).toFixed(1)} MB`;
  if (total >= 1_000) return `${Math.round(total / 1_000)} KB`;
  return `${total} B`;
}

/** The top-level folder a pick came from, when it came from one. */
function rootFolder(picked: Picked[]): string {
  const first = picked[0]?.path ?? "";
  return first.includes("/") ? first.split("/")[0] : "";
}

/**
 * The selection, named.
 *
 * A count alone ("312 files ready") answers "did the click work" and nothing
 * else. Picking the wrong folder produces exactly the same sentence as picking
 * the right one, and the mistake is only discovered after an upload and a prep
 * run. So the folder is named, the first few files are listed with their
 * subfolders intact — the subfolders are the labels, for a classification
 * upload — and the rest collapse to a count.
 */
function PickedFiles({ picked }: { picked: Picked[] }) {
  const folder = rootFolder(picked);
  const bytes = picked.reduce((total, entry) => total + entry.file.size, 0);
  const remaining = picked.length - NAMED_FILES;

  return (
    <div className="ingest-picked">
      <p className="ingest-picked-head">
        {folder ? (
          <>
            <FolderOpen size={14} aria-hidden="true" />
            <strong title={folder}>{folder}</strong>
          </>
        ) : (
          <>
            <FileText size={14} aria-hidden="true" />
            <strong>{picked.length === 1 ? picked[0].path : "Selected files"}</strong>
          </>
        )}
        <span className="form-caption">
          {picked.length.toLocaleString()} file{picked.length === 1 ? "" : "s"} ·{" "}
          {formatBytes(bytes)}
          {folder ? " · folder structure preserved" : ""}
        </span>
      </p>
      <ul className="ingest-picked-list">
        {/* Keyed by position as well as path: a flat pick can hold two files
            with the same name from different folders. */}
        {picked.slice(0, NAMED_FILES).map((entry, index) => (
          <li key={`${entry.path}-${index}`} title={entry.path}>
            {entry.path}
          </li>
        ))}
        {remaining > 0 && (
          <li className="ingest-picked-more">and {remaining.toLocaleString()} more</li>
        )}
      </ul>
    </div>
  );
}

/**
 * Bytes sent, as the job the shell's overlay draws.
 *
 * The readout was the sentence "Uploading 312 files…" for however long the
 * request took, which on a folder of images is minutes of a screen that cannot
 * be told apart from a hung one — and it was drawn *under* the blocking overlay,
 * so in practice nobody saw even that. It is published now, and the overlay
 * renders it. `fraction` is null only where the browser refuses to report a
 * total, and that case goes indeterminate rather than drawing a bar that does
 * not track anything.
 */
function uploadJob(fraction: number | null, files: number, bytes: number): ForegroundJob {
  const noun = `${files.toLocaleString()} file${files === 1 ? "" : "s"}`;
  return {
    label: `Uploading ${noun}`,
    percent: fraction === null ? null : Math.round(fraction * 100),
    count:
      fraction === null
        ? null
        : `${formatBytes(Math.round(bytes * fraction))} of ${formatBytes(bytes)}`,
    stages: [
      { key: "upload", label: "Send the files", state: "active" },
      { key: "prepare", label: "Prepare the dataset", state: "pending" }
    ]
  };
}

export function IngestDropCard({ projectId, onCreated }: { projectId: string; onCreated: (datasetId: string) => void }) {
  const [picked, setPicked] = useState<Picked[]>([]);
  const [name, setName] = useState("");
  const [dragging, setDragging] = useState(false);
  const [pickError, setPickError] = useState("");
  //: The dataset the current run belongs to. Held here rather than read off the
  //: mutation so the progress readout survives `onSettled` clearing it.
  const [runningId, setRunningId] = useState("");
  //: 0..1 of the request body sent, from `XMLHttpRequest`. `null` while the
  //: browser has not reported a computable length — an indeterminate upload is
  //: honest; a bar sitting at zero is not.
  const [uploaded, setUploaded] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const ingestMutation = useIngestDatasetMutation();
  const runMutation = useRunPrepMutation();
  const statusQuery = useDatasetPrepStatusQuery(runningId || undefined, Boolean(runningId));

  const pickedBytes = picked.reduce((total, entry) => total + entry.file.size, 0);

  // Upload first, then the agent — one continuous readout on the shell's
  // overlay, because to the user this is one action. Publishing rather than
  // rendering is the whole point: the overlay is modal, so a panel drawn here
  // would sit behind it.
  usePublishForegroundJob(
    ingestMutation.isPending
      ? uploadJob(uploaded, picked.length, pickedBytes)
      : runMutation.isPending
        ? (prepJob(statusQuery.data) ?? {
            label: "Orinth is reading your files",
            percent: null
          })
        : null
  );

  function takeFileList(list: FileList | null) {
    setPickError("");
    const accepted = Array.from(list ?? [])
      .filter(isAccepted)
      // `webkitRelativePath` is how a folder input reports structure; a plain
      // file pick has none, and the bare name is the right fallback there.
      .map((file) => ({ file, path: file.webkitRelativePath || file.name }));
    if (accepted.length === 0) {
      setPickError("None of those files are a format Orinth can read.");
      return;
    }
    setPicked(accepted);
  }

  /**
   * Two ways to read a folder, same reasoning as `upload-drop-card`:
   * `showDirectoryPicker` is Chromium-only and absent from the desktop app's
   * WKWebView, so `webkitdirectory` is the fallback — and it is not
   * feature-detected, because the attribute is not reliably visible on the
   * prototype and a strict check reports "unsupported" where it works.
   */
  async function selectFolder() {
    setPickError("");
    if (window.showDirectoryPicker) {
      try {
        const directory = await window.showDirectoryPicker();
        const found = await walk(directory);
        if (found.length === 0) {
          setPickError("No files Orinth can read were found in that folder.");
          return;
        }
        setPicked(found);
        if (!name) setName(directory.name);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setPickError(error instanceof Error ? error.message : "Could not read that folder.");
      }
      return;
    }
    folderInputRef.current?.click();
  }

  const busy = ingestMutation.isPending || runMutation.isPending;

  function submit() {
    if (picked.length === 0) return;
    const form = new FormData();
    form.append("project_id", projectId);
    if (name.trim()) form.append("name", name.trim());
    for (const entry of picked) {
      form.append("files", entry.file);
      // Parallel to `files`, and read positionally by the backend.
      form.append("relative_paths", entry.path);
    }
    // Null, not zero: the bar is indeterminate until the browser reports a
    // computable length, and a browser that never reports one must keep saying
    // "running" rather than "0%".
    setUploaded(null);
    ingestMutation.mutate({ form, onProgress: setUploaded }, {
      onSuccess: (dataset) => {
        setPicked([]);
        setName("");
        setUploaded(null);
        setRunningId(dataset.id);
        // Uploading *is* the request to prepare. Landing the user on a dataset
        // with a "now press Prepare" button would just be the old declare-first
        // flow with the questions moved to the end.
        runMutation.mutate(
          { datasetId: dataset.id },
          { onSettled: () => onCreated(dataset.id) }
        );
      }
    });
  }

  return (
    // No panel wrapper of its own: this is now one source inside
    // `NewDatasetPanel`, and a panel nested in a panel is two borders saying
    // one thing (DESIGN.md §5).
    <div className="ingest-card" data-tour="dataset-ingest">
      <div
        className={`ingest-drop ${dragging ? "ingest-drop-active" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          takeFileList(event.dataTransfer.files);
        }}
      >
        <Sparkles size={22} aria-hidden="true" />
        <p className="ingest-drop-title">Drop your data here</p>
        <p className="form-caption">
          Images, text, CSV, JSONL or Parquet — a folder or a pile of files. Orinth works out what
          it is, labels and all.
        </p>
        <div className="action-row">
          <Button variant="secondary" onClick={() => fileInputRef.current?.click()}>
            <Upload size={16} /> Choose files
          </Button>
          <Button variant="secondary" onClick={selectFolder}>
            <FolderOpen size={16} /> Choose folder
          </Button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          onChange={(event) => takeFileList(event.target.files)}
        />
        <input
          ref={folderInputRef}
          type="file"
          multiple
          hidden
          webkitdirectory=""
          onChange={(event) => takeFileList(event.target.files)}
        />
      </div>

      {pickError && <p className="form-caption ingest-error">{pickError}</p>}

      {/* The run keeps going after the user is navigated into the dataset, so
          this is the same readout the Overview tab shows — the transition just
          moves it, rather than restarting the explanation. Upload gets its own
          line because no run exists yet to report on. */}
      {picked.length > 0 && (
        <div className="ingest-review">
          {/* What was actually chosen, by name. The panel used to report only a
              count, which cannot distinguish "the folder I meant" from "the
              folder next to it" — and a folder pick is exactly the gesture where
              that mistake is easy and invisible. */}
          <PickedFiles picked={picked} />
          <Field label="Dataset name" hint="optional">
            <input
              className="text-input"
              value={name}
              placeholder="Named after the folder if left blank"
              onChange={(event) => setName(event.target.value)}
            />
          </Field>
          <div className="action-row">
            {/* One label, because the readout above now says what is happening.
                A button that narrates and a progress panel that narrates are
                two sources for one fact. */}
            <Button variant="primary" onClick={submit} disabled={busy}>
              <Sparkles size={16} /> Upload and prepare
            </Button>
            <Button variant="secondary" onClick={() => setPicked([])} disabled={busy}>
              Clear
            </Button>
          </div>
        </div>
      )}

      <MutationError mutations={[ingestMutation, runMutation]} />
    </div>
  );
}
