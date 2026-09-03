"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Cpu, Zap } from "lucide-react";
import { api } from "@/lib/api";
import { Badge, InlineSpinner } from "@/features/platform/ui";

/**
 * What this run will actually do, before it starts.
 *
 * The reason this is on the form and not in a log: a Keras run on a machine
 * where only torch has a GPU is a CPU run, and the first evidence used to be
 * that it took twenty times longer than the run before it. Nobody guesses the
 * cause — `tensorflow-metal` is a separate wheel — so the form says it, with
 * the fix.
 *
 * The reasons are shown, not just the values. "Batch size 4" is a number;
 * "batch size 4 — 9 GB usable, roughly 1.8 GB per sample, estimate not a
 * measurement" is something the user can disagree with, which is the point of
 * putting it in front of them rather than applying it silently.
 *
 * Re-queried when the family, device, or batch size changes, because all three
 * change the answer.
 */
export function ComputePlanBanner({
  taskType,
  modelFamily,
  device,
  batchSize
}: {
  taskType: string;
  modelFamily: string | undefined;
  device?: string;
  batchSize?: number;
}) {
  const planQuery = useQuery({
    queryKey: ["compute-plan", taskType, modelFamily, device, batchSize],
    queryFn: () =>
      api.computePlan({
        task_type: taskType,
        model_family: modelFamily ?? "",
        device,
        batch_size: batchSize
      }),
    enabled: Boolean(modelFamily),
    // The probe behind this is cached server-side for ten minutes; refetching
    // on window focus would ask a question whose answer cannot have changed.
    staleTime: 60_000,
    retry: false
  });

  if (!modelFamily) return null;
  if (planQuery.isLoading) return <InlineSpinner label="Checking this machine" />;

  const plan = planQuery.data;
  if (!plan) return null;

  return (
    <div className={`compute-banner ${plan.accelerated ? "" : "compute-banner-cpu"}`}>
      <div className="compute-banner-head">
        {plan.accelerated ? <Zap size={14} /> : <Cpu size={14} />}
        <span className="compute-banner-headline">
          {plan.device_name || plan.device.toUpperCase()}
        </span>
        <Badge
          tone={plan.accelerated ? "ok" : "warn"}
          title={`This family trains through ${plan.framework}.`}
        >
          {plan.framework}
        </Badge>
        <Badge tone="neutral" title="Resolved before the run starts.">
          batch {plan.batch_size}
        </Badge>
        <Badge tone="neutral">{plan.precision}</Badge>
      </div>

      {plan.warnings.map((warning) => (
        <p className="compute-banner-warning" key={warning}>
          <AlertTriangle size={13} aria-hidden="true" /> {warning}
        </p>
      ))}

      {/* Collapsed by default: the values answer the question, and the
          reasoning is what you read when you disagree with them. */}
      <details className="compute-banner-why">
        <summary>Why these settings</summary>
        <ul>
          {plan.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      </details>
    </div>
  );
}
