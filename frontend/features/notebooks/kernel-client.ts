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
  private session: ISessionConnection | null = null;

  constructor(
    private readonly baseUrl: string,
    private readonly wsUrl: string,
    private readonly notebookPath: string,
    private readonly kernelName: string
  ) {
    const serverSettings = settingsFor(baseUrl, wsUrl);
    this.manager = new SessionManager({
      kernelManager: new KernelManager({ serverSettings }),
      serverSettings
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

  async interrupt(): Promise<void> {
    await this.session?.kernel?.interrupt();
  }

  async restart(): Promise<void> {
    await this.session?.kernel?.restart();
  }

  async dispose(): Promise<void> {
    // Only the client connection is dropped. The kernel keeps running so a
    // page reload reattaches to live state, and `cull_idle_timeout` reaps it
    // when it genuinely goes idle.
    this.session?.dispose();
    this.manager.dispose();
  }
}
