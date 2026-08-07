import { describe, expect, it } from "vitest";
import {
  activePollInterval,
  allowedTaskTypesForProject,
  areTasksCompatible,
  isActiveStatus,
  isNlpTask,
  isVisionTask,
  labelColor,
  labelFill,
  listPollInterval,
  normalizeTerminalOutput
} from "@/features/platform/utils";

describe("isActiveStatus", () => {
  it("treats completed, failed, and canceled as terminal", () => {
    expect(isActiveStatus("completed")).toBe(false);
    expect(isActiveStatus("failed")).toBe(false);
    expect(isActiveStatus("canceled")).toBe(false);
  });

  it("treats other statuses as active", () => {
    expect(isActiveStatus("running")).toBe(true);
    expect(isActiveStatus("pending")).toBe(true);
    expect(isActiveStatus("queued")).toBe(true);
  });
});

describe("activePollInterval", () => {
  it("returns false when there is no job", () => {
    expect(activePollInterval(undefined)).toBe(false);
  });

  it("returns false once the job reaches a terminal status", () => {
    expect(activePollInterval({ status: "completed" })).toBe(false);
    expect(activePollInterval({ status: "failed" })).toBe(false);
    expect(activePollInterval({ status: "canceled" })).toBe(false);
  });

  it("returns the poll interval while the job is active", () => {
    expect(activePollInterval({ status: "running" })).toBe(2500);
  });
});

describe("listPollInterval", () => {
  it("returns false for an undefined list", () => {
    expect(listPollInterval(undefined)).toBe(false);
  });

  it("returns false when every job is terminal", () => {
    expect(listPollInterval([{ status: "completed" }, { status: "failed" }, { status: "canceled" }])).toBe(false);
  });

  it("returns false for an empty list", () => {
    expect(listPollInterval([])).toBe(false);
  });

  it("returns the poll interval when at least one job is active", () => {
    expect(listPollInterval([{ status: "completed" }, { status: "running" }])).toBe(3000);
  });
});

describe("isNlpTask / isVisionTask", () => {
  it("classifies NLP task types", () => {
    expect(isNlpTask("text_classification")).toBe(true);
    expect(isNlpTask("summarization")).toBe(true);
    expect(isNlpTask("question_answering")).toBe(true);
  });

  it("classifies vision task types as not NLP", () => {
    expect(isNlpTask("classification")).toBe(false);
    expect(isNlpTask("object_detection")).toBe(false);
    expect(isNlpTask("segmentation")).toBe(false);
  });

  it("treats undefined as not NLP", () => {
    expect(isNlpTask(undefined)).toBe(false);
  });

  it("is the inverse of isNlpTask", () => {
    expect(isVisionTask("classification")).toBe(true);
    expect(isVisionTask("text_classification")).toBe(false);
  });
});

describe("areTasksCompatible", () => {
  it("is compatible when the model and dataset tasks match exactly", () => {
    expect(areTasksCompatible("classification", "classification")).toBe(true);
    expect(areTasksCompatible("text_classification", "text_classification")).toBe(true);
  });

  it("treats object_detection and segmentation as shape-compatible with each other", () => {
    expect(areTasksCompatible("object_detection", "segmentation")).toBe(true);
    expect(areTasksCompatible("segmentation", "object_detection")).toBe(true);
  });

  it("rejects mismatched, non-shape task pairs", () => {
    expect(areTasksCompatible("classification", "object_detection")).toBe(false);
    expect(areTasksCompatible("text_classification", "summarization")).toBe(false);
  });
});

describe("labelColor / labelFill", () => {
  it("maps a class index onto the data-viz ramp token", () => {
    expect(labelColor(0)).toBe("var(--color-label-0)");
    expect(labelColor(3)).toBe("var(--color-label-3)");
  });

  it("wraps around the eight-step ramp", () => {
    expect(labelColor(8)).toBe(labelColor(0));
    expect(labelColor(11)).toBe(labelColor(3));
  });

  it("builds a translucent fill without hex concatenation", () => {
    // Appending a hex alpha suffix to a var() reference yields an invalid
    // colour, which paints annotation overlays opaque black over the image.
    expect(labelFill(0)).toBe("color-mix(in oklab, var(--color-label-0) 20%, transparent)");
    expect(labelFill(1, 50)).toBe("color-mix(in oklab, var(--color-label-1) 50%, transparent)");
  });
});

describe("allowedTaskTypesForProject", () => {
  it("keeps every task type a mixed-domain project declares", () => {
    expect(
      allowedTaskTypesForProject({ task_types: ["classification", "summarization"] })
    ).toEqual(["classification", "summarization"]);
  });

  it("returns empty while the project is still resolving", () => {
    // Substituting a default here flashes vision tasks at an NLP project on
    // every page load, because useProject reports null until projects arrive.
    expect(allowedTaskTypesForProject(null)).toEqual([]);
    expect(allowedTaskTypesForProject(undefined)).toEqual([]);
  });

  it("drops unknown task types and falls back only for a resolved empty project", () => {
    expect(allowedTaskTypesForProject({ task_types: ["nonsense"] })).toEqual([
      "classification",
      "object_detection",
      "segmentation"
    ]);
    expect(allowedTaskTypesForProject({ task_types: [] })).toEqual([
      "classification",
      "object_detection",
      "segmentation"
    ]);
  });
});

describe("normalizeTerminalOutput", () => {
  it("drops the trailing backspaces Keras uses to rewind its progress bar", () => {
    // Captured verbatim from a real run log: the metrics are followed by a run
    // of 0x08 bytes, which rendered as a long trail of boxes in the UI.
    const line = "22/56 - 23s 694ms/step - accuracy: 0.3807 - loss: 1.6060" + "\b".repeat(40);
    expect(normalizeTerminalOutput(line)).toBe(
      "22/56 - 23s 694ms/step - accuracy: 0.3807 - loss: 1.6060"
    );
  });

  it("strips ANSI colour codes", () => {
    const line = `ESC[1m0sESC[0m 587ms/step - loss: 1.3129`.replaceAll("ESC", "\u001b");
    expect(normalizeTerminalOutput(line)).toBe("0s 587ms/step - loss: 1.3129");
  });

  it("lets text after a backspace overwrite, as a terminal would", () => {
    // A backspace moves the cursor; it does not erase. Deleting the control
    // character instead of applying it would yield "abcX".
    expect(normalizeTerminalOutput("abc\bX")).toBe("abX");
  });

  it("treats a carriage return as a rewrite of the line", () => {
    expect(normalizeTerminalOutput("first pass\rsecond")).toBe("secondpass");
  });

  it("leaves ordinary log lines untouched", () => {
    expect(normalizeTerminalOutput("Epoch 1/15")).toBe("Epoch 1/15");
    expect(normalizeTerminalOutput("")).toBe("");
  });
});
