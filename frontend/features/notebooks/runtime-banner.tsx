"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Cpu, Play, Power, Terminal } from "lucide-react";
import {
  useComputeTargetsQuery,
  useNotebookRuntimeQuery,
  useStartRuntimeMutation,
  useStopRuntimeMutation
} from "@/features/notebooks/hooks";
import { Badge, Button, InlineSpinner, Select } from "@/features/platform/ui";

/**
 * Whether the kernel gateway is up, which machine it is on, and the controls
 * that change either.
 *
 * A stopped runtime is a normal state, not an error: it is a subprocess that
 * costs memory, and starting it on demand is the same posture the llama.cpp
 * server has. So this reads as a status line with an action, not as a failure
 * banner — the difference matters because the user sees it every time they open
 * the page.
 *
 * A missing optional extra is likewise not an error. It is an install command,
 * shown as one.
 */

const TONE = {
  running: "ok",
  starting: "info",
  stopped: "neutral",
  failed: "fail"
} as const;

/**
 * The machine dropdown.
 *
 * Only shown while the runtime is **stopped**, because a kernel inherits its
 * environment when it spawns: offering a live switch would be a control that
 * silently does nothing until the next restart. A running runtime shows what it
 * is on instead, as a fact rather than a field.
 *
 * Unavailable providers are rendered and disabled, not hidden — phase 24's rule.
 * A roadmap the UI cannot show is a roadmap nobody can plan against.
 */
export function ComputeTargetPicker({
  value,
  onChange,
  disabled = false
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const targetsQuery = useComputeTargetsQuery();
  const targets = targetsQuery.data ?? [];
  const selected = targets.find((target) => target.id === value);

  return (
    <label className="nb-device-picker">
      <span className="sr-only">Machine</span>
      <Cpu size={15} aria-hidden="true" />
      <Select
        value={value}
        disabled={disabled || targetsQuery.isLoading}
        onChange={(event) => onChange(event.target.value)}
        aria-label="Machine to run kernels on"
      >
        {targets.map((target) => (
          <option key={target.id} value={target.id} disabled={!target.available}>
            {target.label}
            {target.available ? "" : " — not available yet"}
          </option>
        ))}
      </Select>
      {selected?.detail && <span className="form-caption">{selected.detail}</span>}
    </label>
  );
}

export function RuntimeBanner({ compact = false }: { compact?: boolean }) {
  const runtimeQuery = useNotebookRuntimeQuery();
  const startMutation = useStartRuntimeMutation();
  const stopMutation = useStopRuntimeMutation();
  const [device, setDevice] = useState("auto");
  const runtime = runtimeQuery.data;

  // Follows the running server rather than resetting the user's pick: after a
  // start, the dropdown should agree with what is actually running.
  useEffect(() => {
    if (runtime?.state === "running" && runtime.device) setDevice(runtime.device);
  }, [runtime?.device, runtime?.state]);

  if (runtimeQuery.isLoading) return <InlineSpinner label="Checking the runtime" />;
  if (!runtime) return null;

  if (!runtime.available) {
    return (
      <div className="nb-runtime nb-runtime-missing">
        <AlertTriangle size={15} aria-hidden="true" />
        <div>
          <p className="nb-runtime-title">Notebooks are not installed</p>
          <p className="form-caption">{runtime.install_hint}</p>
        </div>
      </div>
    );
  }

  const busy = startMutation.isPending || runtime.state === "starting";

  return (
    <div className="nb-runtime">
      <Terminal size={15} aria-hidden="true" />
      <div className="nb-runtime-body">
        <span className="nb-runtime-title">Kernel runtime</span>
        <Badge tone={TONE[runtime.state] ?? "neutral"}>{runtime.state}</Badge>
        {runtime.state === "running" && (
          <span className="form-caption">
            Python {runtime.python_version} · {runtime.kernel_count} kernel
            {runtime.kernel_count === 1 ? "" : "s"} · {runtime.device}
          </span>
        )}
        {runtime.error && <span className="nb-runtime-error">{runtime.error}</span>}
      </div>
      <div className="action-row">
        {runtime.state === "running" ? (
          !compact && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
              title="Stops every kernel. Anything a notebook was holding in memory is lost."
            >
              <Power size={14} /> Stop
            </Button>
          )
        ) : (
          <>
            {!compact && (
              <ComputeTargetPicker value={device} onChange={setDevice} disabled={busy} />
            )}
            <Button
              variant="primary"
              size="sm"
              onClick={() => startMutation.mutate(device)}
              disabled={busy}
            >
              {busy ? <InlineSpinner label="Starting" /> : (<><Play size={14} /> Start runtime</>)}
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
