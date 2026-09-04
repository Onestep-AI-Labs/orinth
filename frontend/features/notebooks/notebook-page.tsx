"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useProject } from "@/components/app-shell";
import {
  ArrowLeft,
  Play,
  ChevronDown,
  ChevronUp,
  Code2,
  Eraser,
  FastForward,
  Pencil,
  Plus,
  RotateCcw,
  Save,
  Text,
  Square,
  Trash2
} from "lucide-react";
import { CellEditor } from "@/features/notebooks/cell-editor";
import { CellOutput } from "@/features/notebooks/cell-output";
import { DatasetRail } from "@/features/notebooks/dataset-rail";
import { MarkdownCell, MarkdownEditor } from "@/features/notebooks/markdown-cell";
import { NotebookKernel, type ExecutionOutput, type KernelState } from "@/features/notebooks/kernel-client";
import { useNotebookQuery, useNotebookRunsQuery } from "@/features/notebooks/hooks";
import { RuntimeMenu } from "@/features/notebooks/runtime-menu";
import { RuntimeStartDialog } from "@/features/notebooks/runtime-dialog";
import {
  useNotebookRuntimeQuery,
  useRenameNotebookMutation
} from "@/features/notebooks/hooks";
import { api } from "@/lib/api";
import { toast } from "@/features/platform/toast";
import {
  Badge,
  EmptyState,
  IconButton,
  PageSkeleton,
  PanelTitle
} from "@/features/platform/ui";

/**
 * The notebook editor.
 *
 * Cells live in React state and are executed one at a time against a kernel
 * reached through the proxy. Deliberately *not* a re-implementation of
 * JupyterLab: there is no command palette, no drag-reorder, no widget manager.
 * What is here is the loop that matters — write a cell, run it, read the
 * output, keep the variables — and everything else is scope this phase declined.
 *
 * State is client-side and the `.ipynb` on disk is the source of truth only at
 * load. Saving writes the whole document back through the contents API, which
 * is what `jupyter-server` is for; two tabs on one notebook is an edge case the
 * spec explicitly does not solve.
 */

type CellKind = "code" | "markdown";

type Cell = {
  id: string;
  kind: CellKind;
  source: string;
  outputs: ExecutionOutput[];
  executionCount: number | null;
  running: boolean;
  //: Markdown renders unless it is being edited. A cell that opened in edit
  //: mode would show every template as raw `#` headings, which is the opposite
  //: of what a template is for.
  editing: boolean;
};

const KERNEL_TONE = {
  idle: "ok",
  busy: "info",
  starting: "info",
  dead: "fail",
  unknown: "neutral"
} as const;

/**
 * `editing` differs by *where the cell came from*, which is why it is a
 * parameter rather than a rule about markdown.
 *
 * A markdown cell **loaded from a file** renders — a template that opened as raw
 * `#` headings teaches nothing. A markdown cell you just **inserted** opens in
 * the editor, because you pressed Text in order to write some: adding one and
 * getting an empty rendered block that says "double-click to edit" is a click
 * that did not do what it said.
 */
function newCell(source = "", kind: CellKind = "code", editing = kind === "code"): Cell {
  return {
    id: Math.random().toString(36).slice(2, 10),
    kind,
    source,
    outputs: [],
    executionCount: null,
    running: false,
    editing
  };
}

export function NotebookPage({ notebookId }: { notebookId: string }) {
  const { projectId } = useProject();
  const notebookQuery = useNotebookQuery(notebookId);
  const runtimeQuery = useNotebookRuntimeQuery();
  const runsQuery = useNotebookRunsQuery(notebookId);

  const renameMutation = useRenameNotebookMutation();

  const [cells, setCells] = useState<Cell[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [kernelState, setKernelState] = useState<KernelState>("unknown");
  const [connecting, setConnecting] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  //: Asked once per visit. A modal that returns after you dismissed it is not
  //: asking, it is insisting — so this latches rather than tracking the runtime.
  const [runtimePrompted, setRuntimePrompted] = useState(false);
  const [runtimeDialogOpen, setRuntimeDialogOpen] = useState(false);
  const kernel = useRef<NotebookKernel | null>(null);
  //: Cell id -> its article node, for scrolling the running one into view.
  const cellNodes = useRef(new Map<string, HTMLElement>());
  //: `runAll` awaits between cells, so a closed-over `cells` would be the array
  //: as it stood when the loop started — stale by the second iteration.
  const cellsRef = useRef<Cell[]>([]);
  cellsRef.current = cells;

  const runtimeRunning = runtimeQuery.data?.state === "running";

  // --- ask to start the runtime, once ------------------------------------
  useEffect(() => {
    if (runtimePrompted || runtimeQuery.isLoading || !runtimeQuery.data) return;
    setRuntimePrompted(true);
    if (runtimeQuery.data.state !== "running") setRuntimeDialogOpen(true);
  }, [runtimePrompted, runtimeQuery.data, runtimeQuery.isLoading]);

  /**
   * Bring a cell into view, and keep it there while it runs.
   *
   * `block: "center"` rather than `"nearest"`: a running cell parked at the very
   * bottom edge shows its first output line and nothing else, which is the one
   * thing you opened it to watch. `behavior: "smooth"` is dropped under
   * `prefers-reduced-motion`, per DESIGN.md §4.
   */
  function scrollToCell(id: string) {
    const node = cellNodes.current.get(id);
    if (!node) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    node.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "center" });
  }

  // --- load the document once the notebook exists -------------------------
  useEffect(() => {
    if (loaded || !notebookQuery.data || !runtimeRunning) return;
    let cancelled = false;
    (async () => {
      try {
        const path = notebookQuery.data.path;
        const response = await fetch(`/api/notebooks/proxy/api/contents/${path}`);
        if (!response.ok) throw new Error(`Could not read the notebook (${response.status})`);
        const payload = await response.json();
        const source = (payload?.content?.cells ?? []) as Array<{
          cell_type: string;
          source: string | string[];
        }>;
        if (cancelled) return;
        // Markdown cells are kept, not filtered. Dropping them lost every
        // template's prose on open — and then wrote it out of the file on the
        // next save, which is data loss, not a missing feature.
        const restored = source.map((cell) =>
          newCell(
            Array.isArray(cell.source) ? cell.source.join("") : cell.source,
            cell.cell_type === "markdown" ? "markdown" : "code"
          )
        );
        setCells(restored.length > 0 ? restored : [newCell()]);
        setLoaded(true);
      } catch (error) {
        if (!cancelled) toast.error((error as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loaded, notebookQuery.data, runtimeRunning]);

  // --- connect the kernel -------------------------------------------------
  /**
   * One kernel client, owned by this effect.
   *
   * The connect used to live outside an effect, and leaked: a client is
   * constructed *before* `connect()` resolves and starts polling `api/kernels`
   * the moment it exists, so a failed connect — or a navigation away mid-flight
   * — left a poller running against a page nobody was on. Against a stopped
   * runtime that is `GET /api/notebooks/proxy/api/kernels → 503` forever, on a
   * backoff that never gives up. Everything the effect creates, the effect
   * disposes.
   */
  useEffect(() => {
    if (!runtimeRunning || !loaded || !notebookQuery.data) return;
    let client: NotebookKernel | null = null;
    let cancelled = false;
    setConnecting(true);
    (async () => {
      try {
        const session = await api.notebookSession(notebookId);
        client = new NotebookKernel(
          session.base_url,
          session.ws_url,
          session.notebook_path,
          session.kernel_name
        );
        await client.connect();
        if (cancelled) {
          void client.dispose();
          return;
        }
        kernel.current = client;
        setKernelState(client.state);
        client.onStatusChange(setKernelState);
        // A runtime that stops under an open page is the other half of the same
        // bug: without this the client keeps retrying on its own backoff and the
        // banner keeps saying `running`. Re-asking flips `runtimeRunning`, and
        // this effect's own cleanup then disposes the client.
        client.onConnectionFailure(() => void runtimeQuery.refetch());
      } catch (error) {
        void client?.dispose();
        if (cancelled) return;
        // The usual cause is a runtime that stopped under us — a backend
        // restart leaves this page holding a status that says `running`. Re-ask
        // rather than leaving the banner claiming everything is fine.
        void runtimeQuery.refetch();
        toast.error(`Could not start a kernel: ${(error as Error).message}`);
      } finally {
        if (!cancelled) setConnecting(false);
      }
    })();
    return () => {
      cancelled = true;
      // Drops the client connection only; the kernel keeps running so a page
      // reload reattaches to live state rather than losing it.
      void (kernel.current ?? client)?.dispose();
      kernel.current = null;
      setKernelState("unknown");
    };
    // `runtimeQuery` is a react-query result object and changes identity on
    // every render; depending on it would tear the kernel down on each one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, notebookId, notebookQuery.data, runtimeRunning]);

  // --- actions ------------------------------------------------------------
  function patch(id: string, update: Partial<Cell>) {
    setCells((current) =>
      current.map((cell) => (cell.id === id ? { ...cell, ...update } : cell))
    );
  }

  /** Returns true when the cell errored, so `runAll` can stop where it should. */
  async function runCell(id: string): Promise<boolean> {
    const cell = cellsRef.current.find((entry) => entry.id === id);
    if (!cell) return false;
    if (cell.kind === "markdown") {
      // "Running" a prose cell means rendering it — the same gesture, the same
      // key, a different meaning, which is the convention every notebook uses.
      patch(id, { editing: false });
      return false;
    }
    const client = kernel.current;
    if (!client) return false;
    // Before the request, not after: a cell that takes ten seconds to start
    // should already be on screen while it does.
    scrollToCell(id);
    patch(id, { outputs: [], running: true });
    try {
      const result = await client.execute(cell.source, (output) => {
        // Appending as they arrive is what makes a long-running cell show its
        // first line immediately instead of looking frozen.
        setCells((current) =>
          current.map((entry) =>
            entry.id === id ? { ...entry, outputs: [...entry.outputs, output] } : entry
          )
        );
      });
      patch(id, { running: false, executionCount: result.executionCount });
      return result.status === "error";
    } catch (error) {
      patch(id, { running: false });
      toast.error((error as Error).message);
      return true;
    }
  }

  function insertAfter(index: number, kind: CellKind) {
    const cell = newCell("", kind, true);
    setCells((current) => [
      ...current.slice(0, index + 1),
      cell,
      ...current.slice(index + 1)
    ]);
    // A new cell you cannot see is a click that appeared to do nothing —
    // inserting near the top of a long notebook is exactly when that happens.
    requestAnimationFrame(() => scrollToCell(cell.id));
  }

  function append(kind: CellKind) {
    const cell = newCell("", kind, true);
    setCells((current) => [...current, cell]);
    requestAnimationFrame(() => scrollToCell(cell.id));
  }

  function remove(id: string) {
    setCells((current) => (current.length === 1 ? current : current.filter((cell) => cell.id !== id)));
  }

  function move(index: number, delta: number) {
    setCells((current) => {
      const target = index + delta;
      if (target < 0 || target >= current.length) return current;
      const next = [...current];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }

  /**
   * Run every cell in order, stopping at the first error.
   *
   * Sequential rather than parallel because a notebook is a script: cell three
   * depends on cell two having defined something. Stopping on error is what
   * makes the result readable — continuing past a failure produces a cascade of
   * NameErrors that bury the one that mattered.
   */
  async function runAll() {
    for (const cell of cells) {
      if (cell.kind === "markdown") {
        patch(cell.id, { editing: false });
        continue;
      }
      // `runCell` scrolls to each one as it starts, so a long run reads as a
      // walk down the notebook rather than as a page that stopped responding.
      const failed = await runCell(cell.id);
      if (failed) {
        // Stop *on* the failure, in view. Leaving the viewport wherever it
        // happened to be is how a traceback goes unread.
        scrollToCell(cell.id);
        break;
      }
    }
  }

  function clearOutputs() {
    setCells((current) =>
      current.map((cell) => ({ ...cell, outputs: [], executionCount: null }))
    );
  }

  async function save() {
    const notebook = notebookQuery.data;
    if (!notebook) return;
    const document = {
      cells: cells.map((cell) => ({
        cell_type: cell.kind,
        // nbformat forbids `execution_count` and `outputs` on a markdown cell;
        // including them makes the file fail validation in any other reader.
        ...(cell.kind === "code" ? { execution_count: cell.executionCount, outputs: [] } : {}),
        id: cell.id,
        metadata: {},
        source: cell.source.split("\n").map((line, index, all) =>
          index === all.length - 1 ? line : `${line}\n`
        )
      })),
      metadata: {
        kernelspec: { display_name: "Orinth", language: "python", name: "orinth" },
        language_info: { name: "python" }
      },
      nbformat: 4,
      nbformat_minor: 5
    };
    const response = await fetch(`/api/notebooks/proxy/api/contents/${notebook.path}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "notebook", format: "json", content: document })
    });
    if (!response.ok) {
      toast.error(`Could not save (${response.status})`);
      return;
    }
    toast.success("Notebook saved.");
  }

  if (notebookQuery.isLoading) return <PageSkeleton title="Loading notebook" />;
  if (!notebookQuery.data) {
    return (
      <EmptyState
        icon={<Square size={26} />}
        label="No such notebook."
        description="It may have been deleted. Go back to the list to pick another."
      />
    );
  }

  const notebook = notebookQuery.data;
  const busy = kernelState === "busy" || cells.some((cell) => cell.running);

  return (
    <div className="space-y-5">
      <div className="nb-header">
        <Link className="nb-back" href="/notebooks">
          <ArrowLeft size={15} /> Notebooks
        </Link>
        {renaming === null ? (
          <h2 className="nb-title">
            <button
              type="button"
              className="nb-title-button"
              onClick={() => setRenaming(notebook.name)}
              title="Rename this notebook"
            >
              {notebook.name}
              <Pencil size={13} aria-hidden="true" />
            </button>
          </h2>
        ) : (
          <form
            className="nb-title-form"
            onSubmit={(event) => {
              event.preventDefault();
              const name = renaming.trim();
              setRenaming(null);
              // An unchanged or emptied name is a cancel, not a request. The
              // API would take "" and leave the list with a blank row.
              if (!name || name === notebook.name) return;
              renameMutation.mutate({ notebookId: notebook.id, name });
            }}
          >
            <input
              className="text-input nb-title-input"
              value={renaming}
              autoFocus
              aria-label="Notebook name"
              onChange={(event) => setRenaming(event.target.value)}
              onBlur={(event) => event.currentTarget.form?.requestSubmit()}
              onKeyDown={(event) => {
                if (event.key === "Escape") setRenaming(null);
              }}
            />
          </form>
        )}
        <div className="nb-header-actions">
          <Badge tone={KERNEL_TONE[kernelState] ?? "neutral"} title="Kernel status">
            {kernelState}
          </Badge>
          <RuntimeMenu />
          <IconButton aria-label="Save the notebook" title="Save (the .ipynb on disk)" onClick={save}>
            <Save size={17} />
          </IconButton>
        </div>
      </div>

      {/* One row of verbs, in the order they are reached for. Colab's shape,
          and the reason it works: insert is what you do most, run is what you do
          next, and everything that resets state is on the far side of a divider
          from everything that does not. */}
      <div className="nb-toolbar" role="toolbar" aria-label="Notebook actions">
        <button type="button" className="nb-tool" onClick={() => append("code")}>
          <Plus size={14} aria-hidden /> <Code2 size={14} aria-hidden /> Code
        </button>
        <button type="button" className="nb-tool" onClick={() => append("markdown")}>
          <Plus size={14} aria-hidden /> <Text size={14} aria-hidden /> Text
        </button>
        <span className="nb-toolbar-divider" aria-hidden />
        <button
          type="button"
          className="nb-tool"
          onClick={runAll}
          disabled={!kernel.current || busy}
          title="Run every cell in order, stopping at the first error"
        >
          <FastForward size={14} aria-hidden /> Run all
        </button>
        <button
          type="button"
          className="nb-tool"
          onClick={() => kernel.current?.interrupt()}
          disabled={!busy}
          title="Send KeyboardInterrupt to the running cell"
        >
          <Square size={14} aria-hidden /> Interrupt
        </button>
        <span className="nb-toolbar-divider" aria-hidden />
        <button
          type="button"
          className="nb-tool"
          onClick={() => kernel.current?.restart()}
          disabled={!kernel.current}
          title="Restart the kernel. Every variable is lost; the runtime stays up."
        >
          <RotateCcw size={14} aria-hidden /> Restart kernel
        </button>
        <button
          type="button"
          className="nb-tool"
          onClick={clearOutputs}
          title="Clear every output. Variables in the kernel are untouched."
        >
          <Eraser size={14} aria-hidden /> Clear outputs
        </button>
      </div>

      <RuntimeStartDialog open={runtimeDialogOpen} />

      <div className="nb-layout">
        <section className="nb-cells">
          {cells.map((cell, index) => (
            <article
              className={`nb-cell ${cell.running ? "nb-cell-running" : ""}`}
              key={cell.id}
              ref={(node) => {
                // A map rather than an array of refs: cells are inserted,
                // deleted and moved, so an index would point at the wrong one
                // the moment anything reorders.
                if (node) cellNodes.current.set(cell.id, node);
                else cellNodes.current.delete(cell.id);
              }}
            >
              <div className="nb-cell-gutter">
                <button
                  type="button"
                  className="nb-run-button"
                  onClick={() => runCell(cell.id)}
                  disabled={cell.kind === "code" && (!kernel.current || cell.running)}
                  title={
                    cell.kind === "markdown"
                      ? "Render this cell (Shift-Enter)"
                      : "Run this cell (Shift-Enter)"
                  }
                  aria-label={cell.kind === "markdown" ? "Render this cell" : "Run this cell"}
                >
                  <Play size={14} />
                </button>
                <span className="nb-cell-count">
                  {cell.kind === "markdown" ? "" : cell.running ? "[*]" : cell.executionCount ? `[${cell.executionCount}]` : "[ ]"}
                </span>
              </div>

              {/* The cell's own actions, top-right, appearing on hover or focus
                  — the shape Colab uses. In the gutter they competed with the
                  run button for the one control anyone presses. */}
              <div className="nb-cell-tools">
                <IconButton
                  aria-label="Move cell up"
                  title="Move up"
                  onClick={() => move(index, -1)}
                  disabled={index === 0}
                >
                  <ChevronUp size={16} />
                </IconButton>
                <IconButton
                  aria-label="Move cell down"
                  title="Move down"
                  onClick={() => move(index, 1)}
                  disabled={index === cells.length - 1}
                >
                  <ChevronDown size={16} />
                </IconButton>
                {cell.kind === "markdown" && (
                  <IconButton
                    aria-label={cell.editing ? "Render this cell" : "Edit this cell"}
                    title={cell.editing ? "Render (Shift-Enter)" : "Edit (double-click)"}
                    onClick={() => patch(cell.id, { editing: !cell.editing })}
                  >
                    <Pencil size={16} />
                  </IconButton>
                )}
                <IconButton
                  aria-label="Delete this cell"
                  title="Delete this cell"
                  danger
                  onClick={() => remove(cell.id)}
                  disabled={cells.length === 1}
                >
                  <Trash2 size={16} />
                </IconButton>
              </div>

              <div className="nb-cell-body">
                {cell.kind === "markdown" ? (
                  cell.editing ? (
                    <MarkdownEditor
                      value={cell.source}
                      onChange={(value) => patch(cell.id, { source: value })}
                      onDone={() => patch(cell.id, { editing: false })}
                    />
                  ) : (
                    <MarkdownCell
                      source={cell.source}
                      editing={false}
                      onEdit={() => patch(cell.id, { editing: true })}
                    />
                  )
                ) : (
                  <CellEditor
                    value={cell.source}
                    onChange={(value) => patch(cell.id, { source: value })}
                    onRun={() => runCell(cell.id)}
                    // A getter, not the kernel itself: the editor mounts before
                    // the kernel connects, and passing the value would freeze
                    // `null` into the completion source for the cell's life.
                    getKernel={() => kernel.current}
                  />
                )}
                {cell.outputs.length > 0 && (
                  <div className="nb-cell-outputs">
                    {cell.outputs.map((output, outputIndex) => (
                      <CellOutput key={outputIndex} output={output} />
                    ))}
                  </div>
                )}
              </div>

              {/* Insert between two cells, where you are actually looking when
                  you want one. Hidden until the gap is hovered, so a notebook at
                  rest is cells and nothing else. */}
              <div className="nb-insert-row">
                <button type="button" className="nb-insert" onClick={() => insertAfter(index, "code")}>
                  <Plus size={12} aria-hidden /> <Code2 size={12} aria-hidden /> Code
                </button>
                <button
                  type="button"
                  className="nb-insert"
                  onClick={() => insertAfter(index, "markdown")}
                >
                  <Plus size={12} aria-hidden /> <Text size={12} aria-hidden /> Text
                </button>
              </div>

            </article>
          ))}
          {connecting && <p className="form-caption">Starting a kernel…</p>}
        </section>

        <aside className="nb-rail">
          <DatasetRail projectId={projectId} />
          <section className="panel">
            <PanelTitle icon={<Play size={16} />} title="Runs" />
            {(runsQuery.data ?? []).length === 0 ? (
              <p className="form-caption">
                Nothing logged yet. <code>orinth.runs.log(loss=…)</code> from a cell and it
                appears here.
              </p>
            ) : (
              <ul className="nb-run-list">
                {(runsQuery.data ?? []).map((run) => (
                  <li key={run.id}>
                    <div className="nb-run-row">
                      <strong>{run.name}</strong>
                      <Badge tone={run.status === "completed" ? "ok" : run.status === "failed" ? "fail" : "info"}>
                        {run.status}
                      </Badge>
                    </div>
                    {run.metric_names.length > 0 && (
                      <p className="form-caption">{run.metric_names.join(", ")}</p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}
