"use client";

import { useState } from "react";
import { Play, Terminal } from "lucide-react";
import { ComputeTargetPicker } from "@/features/notebooks/runtime-banner";
import {
  useNotebookRuntimeQuery,
  useStartRuntimeMutation
} from "@/features/notebooks/hooks";
import { Button, InlineSpinner, Modal } from "@/features/platform/ui";

/**
 * Asked on the way in, once: start the kernel runtime, on which machine.
 *
 * Opening a notebook with the runtime stopped used to leave a banner above the
 * cells and nothing else — every cell was inert, the Run buttons were disabled,
 * and the reason was a status line you had to notice. The one thing the user
 * came here to do was blocked by a state they had not been asked about.
 *
 * So it is a dialog, and it is the *only* dialog: it appears when the page opens
 * to a stopped runtime, and dismissing it leaves the banner behind for a second
 * try. It does not reappear while the page stays open, because a modal that
 * comes back after you dismissed it is not asking, it is insisting.
 *
 * The machine picker lives here rather than after the fact because a kernel
 * inherits its environment when it spawns — this is the moment the choice is
 * still free.
 */
export function RuntimeStartDialog({
  open,
  onClose
}: {
  open: boolean;
  onClose: () => void;
}) {
  const runtimeQuery = useNotebookRuntimeQuery();
  const startMutation = useStartRuntimeMutation();
  const [device, setDevice] = useState("auto");
  const runtime = runtimeQuery.data;

  if (!open || !runtime) return null;

  // A missing extra is an install command, not a dialog with a Start button
  // that cannot work. The banner already says it well; this stays out of the way.
  if (!runtime.available) return null;

  const busy = startMutation.isPending || runtime.state === "starting";

  return (
    <Modal
      title="Start the kernel runtime"
      subtitle="Nothing in this notebook can run until it is up."
      onClose={onClose}
      footer={
        <>
          <Button variant="secondary" onClick={onClose} disabled={busy}>
            Not now
          </Button>
          <Button
            variant="primary"
            onClick={() =>
              startMutation.mutate(device, {
                // Closes only on success. A failed start with the dialog gone
                // would leave the error in a toast and the page unexplained.
                onSuccess: (status) => {
                  if (status.state === "running") onClose();
                }
              })
            }
            disabled={busy}
          >
            {busy ? <InlineSpinner label="Starting" /> : (<><Play size={15} /> Start runtime</>)}
          </Button>
        </>
      }
    >
      <div className="modal-body-padded">
        <p className="nb-runtime-explainer">
          <Terminal size={15} aria-hidden="true" />
          <span>
            Orinth runs your cells in a local Python process on this machine — the same
            environment the platform trains in. It costs memory while it is up, so it is
            started on demand and can be stopped from the Notebooks page.
          </span>
        </p>
        <ComputeTargetPicker value={device} onChange={setDevice} disabled={busy} />
        <p className="form-caption">
          Every kernel inherits this. Changing it later means stopping the runtime, so it is
          asked now.
        </p>
        {runtime.error && <p className="nb-runtime-error">{runtime.error}</p>}
      </div>
    </Modal>
  );
}
