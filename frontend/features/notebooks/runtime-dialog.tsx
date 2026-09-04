"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, Cpu, Play, Terminal } from "lucide-react";
import {
  useComputeTargetsQuery,
  useNotebookRuntimeQuery,
  useStartRuntimeMutation
} from "@/features/notebooks/hooks";
import { Button, ButtonLink, InlineSpinner, Select } from "@/features/platform/ui";

/**
 * Start the runtime, on which machine — asked on the way in, and not dismissed.
 *
 * There is no close button, no Escape, and no backdrop click. Dismissing it
 * left the user on a notebook where every cell was inert and every Run button
 * disabled: a page that looks like an editor and is not one. The honest choice
 * is not "keep the dialog or not" — it is *start it, or leave* — so those are
 * the two things it offers.
 *
 * The machine picker is here rather than after the fact because a kernel
 * inherits its environment when it spawns. This is the moment the choice is
 * still free; afterwards it costs a restart, which the runtime menu says.
 */
export function RuntimeStartDialog({ open }: { open: boolean }) {
  const runtimeQuery = useNotebookRuntimeQuery();
  const targetsQuery = useComputeTargetsQuery();
  const startMutation = useStartRuntimeMutation();
  const [device, setDevice] = useState("auto");
  const runtime = runtimeQuery.data;

  // Nothing behind the dialog is reachable, so nothing behind it should scroll.
  useEffect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [open]);

  if (!open || !runtime) return null;

  const targets = targetsQuery.data ?? [];
  const selected = targets.find((target) => target.id === device);
  const busy = startMutation.isPending || runtime.state === "starting";
  const installed = runtime.available;

  return (
    <div className="modal-overlay" role="presentation">
      <section
        className="modal-panel nb-start-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="nb-start-title"
      >
        <header className="modal-head">
          <div className="modal-heading">
            <h2 id="nb-start-title">
              {installed ? "Start the kernel runtime" : "Notebooks are not installed"}
            </h2>
            <p>
              {installed
                ? "Nothing in this notebook can run until it is up."
                : "The optional extra is missing, so no kernel can start."}
            </p>
          </div>
        </header>

        <div className="modal-body">
          <div className="modal-body-padded">
            {installed ? (
              <>
                <p className="nb-runtime-explainer">
                  <Terminal size={15} aria-hidden="true" />
                  <span>
                    Cells run in a local Python process on this machine — the same environment
                    the platform trains in. It costs memory while it is up, so it starts on
                    demand and can be stopped from the Runtime menu.
                  </span>
                </p>

                <label className="nb-runtime-field">
                  <span className="advanced-group-label">Machine</span>
                  <span className="nb-runtime-select">
                    <Cpu size={15} aria-hidden="true" />
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

                <p className="form-caption">
                  Every kernel inherits this. Changing it later means restarting the runtime,
                  so it is asked now.
                </p>
              </>
            ) : (
              <p className="nb-runtime-explainer">
                <Terminal size={15} aria-hidden="true" />
                <span>{runtime.install_hint}</span>
              </p>
            )}
            {runtime.error && <p className="nb-runtime-error">{runtime.error}</p>}
          </div>
        </div>

        <div className="modal-actions">
          {/* Not a dismiss. The way out of a notebook you cannot run is the
              notebook list, and saying so beats a Cancel that leaves you on a
              dead page. */}
          <ButtonLink variant="secondary" href="/notebooks">
            <ArrowLeft size={15} /> Back to notebooks
          </ButtonLink>
          {installed && (
            <Button
              variant="primary"
              disabled={busy}
              onClick={() => startMutation.mutate(device)}
              autoFocus
            >
              {busy ? <InlineSpinner label="Starting" /> : (<><Play size={15} /> Start runtime</>)}
            </Button>
          )}
        </div>
      </section>
    </div>
  );
}
