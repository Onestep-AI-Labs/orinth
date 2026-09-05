import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import {
  JobProgress,
  setForegroundJob,
  useForegroundJob,
  usePublishForegroundJob,
  type ForegroundJob
} from "@/features/platform/foreground-job";

function Consumer() {
  const job = useForegroundJob();
  return <div data-testid="consumer">{job ? `${job.label} @ ${job.percent}` : "idle"}</div>;
}

function Publisher({ job }: { job: ForegroundJob | null }) {
  usePublishForegroundJob(job);
  return null;
}

afterEach(() => {
  act(() => setForegroundJob(null));
});

describe("the foreground job channel", () => {
  it("carries a leaf component's progress up to the shell", () => {
    // The publisher is deep inside a route and the consumer is the shell that
    // renders it, which is why this is a store and not a context.
    render(
      <>
        <Consumer />
        <Publisher job={{ label: "Writing record 8,412 of 41,003", percent: 84 }} />
      </>
    );
    expect(screen.getByTestId("consumer")).toHaveTextContent("Writing record 8,412 of 41,003 @ 84");
  });

  it("clears when the publisher unmounts", () => {
    // A surface that navigates away mid-run would otherwise leave the blocking
    // overlay stuck on its last frame forever.
    const { rerender } = render(
      <>
        <Consumer />
        <Publisher job={{ label: "Uploading", percent: 10 }} />
      </>
    );
    rerender(<Consumer />);
    expect(screen.getByTestId("consumer")).toHaveTextContent("idle");
  });

  it("renders a determinate bar with its percentage and counts", () => {
    render(
      <JobProgress
        job={{ label: "Writing records", percent: 42, count: "8,412 of 41,003" }}
      />
    );
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
    expect(screen.getByText("42%")).toBeInTheDocument();
    expect(screen.getByText("8,412 of 41,003")).toBeInTheDocument();
  });

  it("still draws a bar when the work cannot say how far through it is", () => {
    render(<JobProgress job={{ label: "Uploading", percent: null }} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toBeInTheDocument();
    expect(bar).not.toHaveAttribute("aria-valuenow");
  });
});
