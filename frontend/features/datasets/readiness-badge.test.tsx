import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReadinessBadge, readinessPresentation } from "@/features/datasets/readiness-badge";
import type { DatasetReadiness } from "@/types/api";

function readiness(overrides: Partial<DatasetReadiness> = {}): DatasetReadiness {
  return {
    state: "ready",
    trainable: true,
    summary: "Ready to train.",
    next_action: "none",
    checks: [],
    busy: false,
    ...overrides
  } as DatasetReadiness;
}

describe("ReadinessBadge", () => {
  it("labels a ready dataset", () => {
    render(<ReadinessBadge readiness={readiness()} />);
    expect(screen.getByText("Ready to train")).toBeInTheDocument();
  });

  it("distinguishes what the agent can fix from what needs a person", () => {
    expect(readinessPresentation("needs_prep").tone).toBe("warn");
    expect(readinessPresentation("needs_input").tone).toBe("fail");
  });

  it("carries the reason as the title so it is one hover away", () => {
    render(
      <ReadinessBadge
        readiness={readiness({
          state: "needs_input",
          trainable: false,
          summary: "This task needs at least 2 class labels; 0 defined."
        })}
      />
    );
    expect(screen.getByText("Needs input")).toHaveAttribute(
      "title",
      "This task needs at least 2 class labels; 0 defined."
    );
  });

  it("renders nothing rather than an empty pill when readiness is absent", () => {
    const { container } = render(<ReadinessBadge readiness={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("falls back neutrally for a state written by a newer build", () => {
    // Manifests outlive the schema; an unknown state must not blank the catalog.
    expect(readinessPresentation("something_new" as never).tone).toBe("neutral");
  });
});
