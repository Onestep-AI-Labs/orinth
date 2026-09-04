"use client";

import { KernelManager, ServerConnection, SessionManager } from "@jupyterlab/services";
import type { ISessionConnection } from "@jupyterlab/services/lib/session/session";

/**
 * The connection to the kernel, through our own proxy.
 *
 * `@jupyterlab/services` is the Jupyter protocol client — the half of
 * JupyterLab worth having. The editor is ours; the messaging is not something
 * to reimplement, because the kernel protocol is a real specification with
 * ordering, parent-header correlation, and a WebSocket subprotocol.
 *
 * Every URL is same-origin (`/api/notebooks/proxy/…`) because the backend runs
 * `jupyter-server` with that as its `base_url`. The browser never learns the
 * loopback port, never holds the server's token, and needs no CORS.
 */

export type ExecutionOutput = {
  output_type: string;
  [key: string]: unknown;
};

export type KernelState = "starting" | "idle" | "busy" | "dead" | "unknown";

function settingsFor(baseUrl: string, wsUrl: string): ServerConnection.ISettings {
  const origin = typeof window === "undefined" ? "" : window.location.origin;
  return ServerConnection.makeSettings({
    baseUrl: `${origin}${baseUrl}/`,
    wsUrl: `${origin.replace(/^http/, "ws")}${wsUrl}/`,
    // The proxy injects the credential outbound; there is nothing here to send.
    token: "",
    appendToken: false
  });
}

export class NotebookKernel {
  private manager: SessionManager;
  private kernels: KernelManager;
  private session: ISessionConnection | null = null;
  private disposed = false;

  constructor(
    private readonly baseUrl: string,
    private readonly wsUrl: string,
    private readonly notebookPath: string,
    private readonly kernelName: string
  ) {
    const serverSettings = settingsFor(baseUrl, wsUrl);
    // Both managers poll from the moment they are constructed, which is why
    // they are held: disposing the session alone left a `KernelManager` polling
    // `api/kernels` forever, and against a stopped runtime that is an endless
    // stream of 503s from a page the user has already navigated away from.
    this.kernels = new KernelManager({ serverSettings, standby: "when-hidden" });
    this.manager = new SessionManager({
      kernelManager: this.kernels,
      serverSettings,
      // A background tab does not need to know a kernel list changed. This is
      // `@jupyterlab/services`' own switch for it.
      standby: "when-hidden"
    });
  }

  /**
   * Attach to this notebook's session, starting one if it has none.
   *
   * Keyed by path rather than by a fresh id so reopening the page reattaches to
   * the kernel that is already running — reconnecting must not silently discard
   * the variables the user spent ten minutes computing.
   */
  async connect(): Promise<void> {
    await this.manager.ready;
    this.session = await this.manager.startNew(
      {
        path: this.notebookPath,
        type: "notebook",
        name: this.notebookPath,
        kernel: { name: this.kernelName }
      },
      { kernelConnectionOptions: { handleComms: false } }
    );
  }

  get state(): KernelState {
    const status = this.session?.kernel?.status;
    if (!status) return "unknown";
    if (status === "idle" || status === "busy" || status === "starting" || status === "dead") {
      return status;
    }
    // `restarting`, `autorestarting`, `terminating`, `connected`, `unknown` all
    // mean "not ready for input yet", which `busy` already communicates.
    return status === "unknown" ? "unknown" : "busy";
  }

  /**
   * The managers gave up talking to the server.
   *
   * Which, here, almost always means the runtime stopped — a backend restart,
   * or someone pressing Stop. Without this the page keeps a dead session and
   * `@jupyterlab/services` keeps retrying on its own backoff, so the banner goes
   * on saying `running` while every request answers 503. The page uses it to
   * re-ask the runtime and tear the client down.
   */
  onConnectionFailure(listener: () => void): () => void {
    const handler = () => listener();
    this.manager.connectionFailure.connect(handler);
    this.kernels.connectionFailure.connect(handler);
    return () => {
      this.manager.connectionFailure.disconnect(handler);
      this.kernels.connectionFailure.disconnect(handler);
    };
  }

  onStatusChange(listener: (state: KernelState) => void): () => void {
    const kernel = this.session?.kernel;
    if (!kernel) return () => undefined;
    const handler = () => listener(this.state);
    kernel.statusChanged.connect(handler);
    return () => kernel.statusChanged.disconnect(handler);
  }

  /**
   * Run one cell, streaming outputs as they arrive.
   *
   * Outputs are delivered through the callback rather than returned at the end
   * because a cell that prints for thirty seconds should show its first line
   * immediately — collecting them and resolving once is how a notebook ends up
   * looking frozen.
   */
  async execute(
    code: string,
    onOutput: (output: ExecutionOutput) => void
  ): Promise<{ status: string; executionCount: number | null }> {
    const kernel = this.session?.kernel;
    if (!kernel) throw new Error("No kernel is connected.");

    const future = kernel.requestExecute({ code, stop_on_error: false });
    future.onIOPub = (message) => {
      const type = message.header.msg_type;
      if (type === "status" || type === "execute_input" || type === "clear_output") return;
      onOutput({ output_type: type, ...(message.content as Record<string, unknown>) });
    };

    const reply = await future.done;
    const content = reply.content as { status: string; execution_count?: number };
    return { status: content.status, executionCount: content.execution_count ?? null };
  }

  /**
   * Completions from the kernel, not from a word list.
   *
   * This is what makes autocomplete useful rather than decorative: the kernel
   * introspects the *live* namespace, so after `df = orinth.datasets.load(…)`
   * it knows `df.` offers polars methods. A static Python keyword list cannot
   * know that, and a list scraped from the buffer would offer strings that are
   * not attributes of anything.
   *
   * Returns null rather than throwing when there is no kernel or the request
   * fails — a completion popup is a convenience, and one that raises into the
   * editor on every keystroke is worse than none.
   */
  async complete(
    code: string,
    cursor: number
  ): Promise<{ matches: string[]; start: number; end: number } | null> {
    const kernel = this.session?.kernel;
    if (!kernel) return null;
    try {
      const reply = await kernel.requestComplete({ code, cursor_pos: cursor });
      const content = reply.content as {
        status: string;
        matches?: string[];
        cursor_start?: number;
        cursor_end?: number;
      };
      if (content.status !== "ok" || !content.matches?.length) return null;
      return {
        matches: content.matches,
        start: content.cursor_start ?? cursor,
        end: content.cursor_end ?? cursor
      };
    } catch {
      return null;
    }
  }

  /**
   * The docstring behind a symbol, for Shift-Tab.
   *
   * `detail_level: 0` is the short form — a signature and the opening lines —
   * which is what fits in a popup. Level 1 returns the full source, which is a
   * different feature (go-to-definition) and not this one.
   */
  async inspect(code: string, cursor: number): Promise<string | null> {
    const kernel = this.session?.kernel;
    if (!kernel) return null;
    try {
      const reply = await kernel.requestInspect({ code, cursor_pos: cursor, detail_level: 0 });
      const content = reply.content as { status: string; found?: boolean; data?: Record<string, unknown> };
      if (content.status !== "ok" || !content.found) return null;
      const text = content.data?.["text/plain"];
      return typeof text === "string" ? text : null;
    } catch {
      return null;
    }
  }

  async interrupt(): Promise<void> {
    await this.session?.kernel?.interrupt();
  }

  async restart(): Promise<void> {
    await this.session?.kernel?.restart();
  }

  /**
   * Drop every client-side handle, including the pollers.
   *
   * Only the *connection* is dropped: the kernel keeps running so a page reload
   * reattaches to live state, and `cull_idle_timeout` reaps it when it
   * genuinely goes idle. Idempotent, because the page disposes on unmount and
   * again when a connect that was already in flight resolves.
   */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.session?.dispose();
    this.manager.dispose();
    this.kernels.dispose();
  }
}
