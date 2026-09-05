import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PrepProgress } from "@/features/datasets/prep/prep-progress";
import type { DatasetPrepStatus } from "@/types/api";

function status(overrides: Partial<DatasetPrepStatus> = {}): DatasetPrepStatus {
  return {
    state: "planning",
    job_id: null,
    staged_files: 3,
    applied_at: null,
    error: null,
    step: "planning",
    detail: "Read 312 files — looks like classification, checking",
    progress: 0.35,
    ...overrides
  } as DatasetPrepStatus;
}

describe("PrepProgress", () => {
  it("shows the server's sentence rather than a generic label", () => {
    render(<PrepProgress status={status()} />);
    expect(
      screen.getByText("Read 312 files — looks like classification, checking")
    ).toBeInTheDocument();
  });

  it("renders nothing when no run is in flight", () => {
    const { container } = render(<PrepProgress status={status({ state: "ready", step: "done" })} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing without a status at all", () => {
    const { container } = render(<PrepProgress status={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("hides the transform rung until a run actually reaches it", () => {
    // Most runs never need the sandbox. A permanently-pending stage reads as a
    // stall, so it must not appear before it is reached.
    render(<PrepProgress status={status()} />);
    expect(screen.queryByText("Reshape the rows")).not.toBeInTheDocument();

    render(<PrepProgress status={status({ step: "transforming" })} />);
    expect(screen.getByText("Reshape the rows")).toBeInTheDocument();
  });

  it("marks earlier stages done and the current one active", () => {
    render(<PrepProgress status={status({ state: "applying", step: "applying" })} />);
    expect(screen.getByText("Read the files").className).toContain("prep-stage-done");
    expect(screen.getByText("Work out the task").className).toContain("prep-stage-done");
    expect(screen.getByText("Build the splits").className).toContain("prep-stage-active");
  });

  it("is announced to assistive tech as it changes", () => {
    render(<PrepProgress status={status()} />);
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
  });

  it("draws a determinate bar with the run's percentage", () => {
    render(<PrepProgress status={status({ progress: 0.42 })} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("shows the stage's own counts beside the bar", () => {
    // A percentage cannot be checked against anything; "8,412 of 41,003" can,
    // and it is what tells the user the run is moving rather than hung.
    render(
      <PrepProgress
        status={status({ state: "applying", step: "applying", processed: 8412, total: 41003 })}
      />
    );
    expect(screen.getByText("8,412 of 41,003")).toBeInTheDocument();
  });

  it("still draws a bar when the server reports no fraction", () => {
    render(<PrepProgress status={status({ progress: null })} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toBeInTheDocument();
    expect(bar).not.toHaveAttribute("aria-valuenow");
  });
});
