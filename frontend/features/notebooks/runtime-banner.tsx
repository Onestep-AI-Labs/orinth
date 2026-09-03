"use client";

import { AlertTriangle, Play, Power, Terminal } from "lucide-react";
import {
  useNotebookRuntimeQuery,
  useStartRuntimeMutation,
  useStopRuntimeMutation
} from "@/features/notebooks/hooks";
import { Badge, Button, InlineSpinner } from "@/features/platform/ui";

/**
 * Whether the kernel gateway is up, and the one control that changes it.
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

export function RuntimeBanner({ compact = false }: { compact?: boolean }) {
  const runtimeQuery = useNotebookRuntimeQuery();
  const startMutation = useStartRuntimeMutation();
  const stopMutation = useStopRuntimeMutation();
  const runtime = runtimeQuery.data;

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
            {runtime.kernel_count === 1 ? "" : "s"}
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
          <Button
            variant="primary"
            size="sm"
            onClick={() => startMutation.mutate()}
            disabled={busy}
          >
            {busy ? <InlineSpinner label="Starting" /> : (<><Play size={14} /> Start runtime</>)}
          </Button>
        )}
      </div>
    </div>
  );
}
