"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, Circle, Cpu, Play, Power, RotateCcw } from "lucide-react";
import {
  useComputeTargetsQuery,
  useNotebookRuntimeQuery,
  useRestartRuntimeMutation,
  useStartRuntimeMutation,
  useStopRuntimeMutation
} from "@/features/notebooks/hooks";
import { Button, InlineSpinner, Select } from "@/features/platform/ui";

/**
 * The runtime, from inside the notebook.
 *
 * It used to live on the Notebooks *list* page, which is the one place you are
 * not using it: you start a runtime because you are about to run something, and
 * that is here. The list page now shows nothing about it at all.
 *
 * A menu rather than a row of buttons because these are the rare controls —
 * pressed once at the start of a session and then not again — and a permanent
 * strip of them would sit at the same weight as Run all, which is pressed
 * constantly. The trigger carries the state, so the *information* stays visible
 * while the actions fold away.
 */

const TONE = {
  running: "var(--color-ok)",
  starting: "var(--color-info)",
  stopped: "var(--color-ink-subtle)",
  failed: "var(--color-danger)"
} as const;

export function RuntimeMenu() {
  const runtimeQuery = useNotebookRuntimeQuery();
  const targetsQuery = useComputeTargetsQuery();
  const startMutation = useStartRuntimeMutation();
  const stopMutation = useStopRuntimeMutation();
  const restartMutation = useRestartRuntimeMutation();
  const [open, setOpen] = useState(false);
  const [device, setDevice] = useState("auto");
  const host = useRef<HTMLDivElement>(null);

  const runtime = runtimeQuery.data;
  const targets = targetsQuery.data ?? [];
  const running = runtime?.state === "running";
  const busy =
    startMutation.isPending ||
    restartMutation.isPending ||
    stopMutation.isPending ||
    runtime?.state === "starting";

  // Follows the running server: the dropdown should agree with what is actually
  // up, so "restart on this" starts from where you are rather than from `auto`.
  useEffect(() => {
    if (running && runtime?.device) setDevice(runtime.device);
  }, [running, runtime?.device]);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (!host.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  if (!runtime) return null;

  const selected = targets.find((target) => target.id === device);
  const moved = running && device !== runtime.device;

  return (
    <div className="nb-runtime-menu" ref={host}>
      <button
        type="button"
        className="nb-runtime-trigger"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((value) => !value)}
        title="Runtime: which machine, and whether it is up"
      >
        <Circle size={8} fill="currentColor" style={{ color: TONE[runtime.state] }} aria-hidden />
        <span className="nb-runtime-trigger-label">
          {running ? runtime.device : runtime.state}
        </span>
        <ChevronDown size={14} aria-hidden />
      </button>

      {open && (
        <div className="nb-runtime-panel" role="dialog" aria-label="Runtime">
          <div className="nb-runtime-panel-head">
            <span className="nb-runtime-title">Runtime</span>
            <span className="form-caption">
              {running
                ? `Python ${runtime.python_version} · ${runtime.kernel_count} kernel${
                    runtime.kernel_count === 1 ? "" : "s"
                  }`
                : runtime.state}
            </span>
          </div>

          {runtime.available ? (
            <>
              <label className="nb-runtime-field">
                <span className="advanced-group-label">Machine</span>
                <span className="nb-runtime-select">
                  <Cpu size={15} aria-hidden />
                  <Select
                    value={device}
                    disabled={busy || targetsQuery.isLoading}
                    aria-label="Machine to run kernels on"
                    onChange={(event) => setDevice(event.target.value)}
                  >
                    {targets.map((target) => (
                      <option key={target.id} value={target.id} disabled={!target.available}>
                        {target.label}
                        {target.available ? "" : " — not available yet"}
                      </option>
                    ))}
                  </Select>
                </span>
                {selected?.detail && <span className="form-caption">{selected.detail}</span>}
              </label>

              {moved && (
                <p className="form-caption nb-runtime-moved">
                  Running on <strong>{runtime.device}</strong>. Restart to move to{" "}
                  <strong>{device}</strong> — every kernel is lost.
                </p>
              )}

              <div className="nb-runtime-panel-actions">
                {running ? (
                  <>
                    <Button
                      variant={moved ? "primary" : "secondary"}
                      size="sm"
                      disabled={busy}
                      onClick={() => restartMutation.mutate(device)}
                      title="Stop and start again. Every kernel dies."
                    >
                      {busy ? <InlineSpinner label="Restarting" /> : (<><RotateCcw size={14} /> Restart</>)}
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={busy}
                      onClick={() => stopMutation.mutate()}
                      title="Stops every kernel. Anything held in memory is lost."
                    >
                      <Power size={14} /> Stop
                    </Button>
                  </>
                ) : (
                  <Button
                    variant="primary"
                    size="sm"
                    disabled={busy}
                    onClick={() => startMutation.mutate(device)}
                  >
                    {busy ? <InlineSpinner label="Starting" /> : (<><Play size={14} /> Start</>)}
                  </Button>
                )}
              </div>
            </>
          ) : (
            <p className="form-caption">{runtime.install_hint}</p>
          )}

          {runtime.error && <p className="nb-runtime-error">{runtime.error}</p>}
        </div>
      )}
    </div>
  );
}
