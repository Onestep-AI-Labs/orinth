"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useProject } from "@/components/app-shell";
import {
  ArrowLeft,
  Circle,
  Play,
  Code2,
  Plus,
  RotateCcw,
  Text,
  Square,
  Trash2
} from "lucide-react";
import { CellEditor } from "@/features/notebooks/cell-editor";
import { CellOutput } from "@/features/notebooks/cell-output";
import { DatasetRail } from "@/features/notebooks/dataset-rail";
import { MarkdownCell } from "@/features/notebooks/markdown-cell";
import { NotebookKernel, type ExecutionOutput, type KernelState } from "@/features/notebooks/kernel-client";
import { useNotebookQuery, useNotebookRunsQuery } from "@/features/notebooks/hooks";
import { RuntimeBanner } from "@/features/notebooks/runtime-banner";
import { useNotebookRuntimeQuery } from "@/features/notebooks/hooks";
import { api } from "@/lib/api";
import { toast } from "@/features/platform/toast";
import {
  Badge,
  Button,
  EmptyState,
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

function newCell(source = "", kind: CellKind = "code"): Cell {
  return {
    id: Math.random().toString(36).slice(2, 10),
    kind,
    source,
    outputs: [],
    executionCount: null,
    running: false,
    editing: kind === "code"
  };
}

export function NotebookPage({ notebookId }: { notebookId: string }) {
  const { projectId } = useProject();
  const notebookQuery = useNotebookQuery(notebookId);
  const runtimeQuery = useNotebookRuntimeQuery();
  const runsQuery = useNotebookRunsQuery(notebookId);

  const [cells, setCells] = useState<Cell[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [kernelState, setKernelState] = useState<KernelState>("unknown");
  const [connecting, setConnecting] = useState(false);
  const kernel = useRef<NotebookKernel | null>(null);

  const runtimeRunning = runtimeQuery.data?.state === "running";

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
  const connect = useCallback(async () => {
    if (kernel.current || !notebookQuery.data) return;
    setConnecting(true);
    try {
      const session = await api.notebookSession(notebookId);
      const client = new NotebookKernel(
        session.base_url,
        session.ws_url,
        session.notebook_path,
        session.kernel_name
      );
      await client.connect();
      kernel.current = client;
      setKernelState(client.state);
      client.onStatusChange(setKernelState);
    } catch (error) {
      toast.error(`Could not start a kernel: ${(error as Error).message}`);
    } finally {
      setConnecting(false);
    }
  }, [notebookId, notebookQuery.data]);

  useEffect(() => {
    if (runtimeRunning && loaded) void connect();
  }, [connect, loaded, runtimeRunning]);

  useEffect(
    () => () => {
      // Drops the client connection only; the kernel keeps running so a page
      // reload reattaches to live state rather than losing it.
      void kernel.current?.dispose();
      kernel.current = null;
    },
    []
  );

  // --- actions ------------------------------------------------------------
  function patch(id: string, update: Partial<Cell>) {
    setCells((current) =>
      current.map((cell) => (cell.id === id ? { ...cell, ...update } : cell))
    );
  }

  async function runCell(id: string) {
    const cell = cells.find((entry) => entry.id === id);
    if (!cell) return;
    if (cell.kind === "markdown") {
      // "Running" a prose cell means rendering it — the same gesture, the same
      // key, a different meaning, which is the convention every notebook uses.
      patch(id, { editing: false });
      return;
    }
    const client = kernel.current;
    if (!client) return;
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
    } catch (error) {
      patch(id, { running: false });
      toast.error((error as Error).message);
    }
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
        <h2 className="nb-title">{notebook.name}</h2>
        <div className="nb-header-actions">
          <Badge tone={KERNEL_TONE[kernelState] ?? "neutral"} title="Kernel status">
            <Circle size={8} /> {kernelState}
          </Badge>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => kernel.current?.interrupt()}
            disabled={!busy}
            title="Send KeyboardInterrupt to the running cell."
          >
            <Square size={14} /> Interrupt
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => kernel.current?.restart()}
            disabled={!kernel.current}
            title="Restart the kernel. Every variable is lost."
          >
            <RotateCcw size={14} /> Restart
          </Button>
          <Button variant="secondary" size="sm" onClick={save}>
            Save
          </Button>
        </div>
      </div>

      {!runtimeRunning && <RuntimeBanner compact />}

      <div className="nb-layout">
        <section className="nb-cells">
          {cells.map((cell, index) => (
            <article className="nb-cell" key={cell.id}>
              <div className="nb-cell-gutter">
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => runCell(cell.id)}
                  disabled={cell.kind === "code" && (!kernel.current || cell.running)}
                  title={cell.kind === "markdown" ? "Render this cell (Shift-Enter)" : "Run this cell (Shift-Enter)"}
                >
                  <Play size={14} />
                </button>
                <span className="nb-cell-count">
                  {cell.kind === "markdown" ? "md" : cell.running ? "*" : (cell.executionCount ?? " ")}
                </span>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => setCells((current) => current.filter((entry) => entry.id !== cell.id))}
                  disabled={cells.length === 1}
                  title="Delete this cell"
                >
                  <Trash2 size={13} />
                </button>
              </div>
              <div className="nb-cell-body">
                {cell.kind === "markdown" && !cell.editing ? (
                  <MarkdownCell
                    source={cell.source}
                    editing={false}
                    onEdit={() => patch(cell.id, { editing: true })}
                  />
                ) : (
                  <CellEditor
                    value={cell.source}
                    onChange={(value) => patch(cell.id, { source: value })}
                    onRun={() => runCell(cell.id)}
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
              {index === cells.length - 1 && (
                <div className="nb-add-cell-row">
                  <button
                    type="button"
                    className="nb-add-cell"
                    onClick={() => setCells((current) => [...current, newCell("", "code")])}
                  >
                    <Plus size={13} /> <Code2 size={12} /> Code
                  </button>
                  <button
                    type="button"
                    className="nb-add-cell"
                    onClick={() => setCells((current) => [...current, newCell("", "markdown")])}
                  >
                    <Plus size={13} /> <Text size={12} /> Text
                  </button>
                </div>
              )}
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
