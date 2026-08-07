import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Badge, Button, ButtonLink, IconButton, Select } from "./primitives";

describe("Button", () => {
  it("renders a native button with the mapped variant class", () => {
    render(<Button variant="secondary">Save draft</Button>);
    const button = screen.getByRole("button", { name: "Save draft" });
    expect(button.tagName).toBe("BUTTON");
    expect(button).toHaveClass("secondary-button");
    expect(button).toHaveAttribute("type", "button");
  });

  it("defaults to the primary variant and supports the sm size", () => {
    render(<Button size="sm">Run</Button>);
    const button = screen.getByRole("button", { name: "Run" });
    expect(button).toHaveClass("primary-button");
    expect(button).toHaveClass("button-sm");
  });

  it("fires onClick and respects disabled", async () => {
    const onClick = vi.fn();
    render(
      <Button variant="danger" onClick={onClick} disabled>
        Delete
      </Button>
    );
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onClick).not.toHaveBeenCalled();
  });
});

describe("ButtonLink", () => {
  it("renders an anchor styled as a button", () => {
    render(
      <ButtonLink variant="secondary" href="/training">
        Start training
      </ButtonLink>
    );
    const link = screen.getByRole("link", { name: "Start training" });
    expect(link).toHaveAttribute("href", "/training");
    expect(link).toHaveClass("secondary-button");
  });
});

describe("IconButton", () => {
  it("exposes its aria-label and mirrors it into title", () => {
    render(<IconButton aria-label="Refresh" />);
    const button = screen.getByRole("button", { name: "Refresh" });
    expect(button).toHaveClass("icon-button");
    expect(button).toHaveAttribute("title", "Refresh");
  });

  it("applies the danger treatment", () => {
    render(<IconButton aria-label="Remove" danger />);
    expect(screen.getByRole("button", { name: "Remove" })).toHaveClass("danger-icon-button");
  });
});

describe("Badge", () => {
  it("maps tones onto badge classes", () => {
    render(<Badge tone="ok">completed</Badge>);
    const badge = screen.getByText("completed");
    expect(badge).toHaveClass("badge");
    expect(badge).toHaveClass("badge-ok");
  });
});

describe("Select", () => {
  it("carries the shared dropdown class so every select looks the same", () => {
    render(
      <Select aria-label="Task">
        <option value="a">A</option>
      </Select>
    );
    expect(screen.getByRole("combobox", { name: "Task" })).toHaveClass("select-control");
  });

  it("keeps caller classes alongside the shared one", () => {
    render(<Select aria-label="Device" className="w-40" />);
    const select = screen.getByRole("combobox", { name: "Device" });
    expect(select).toHaveClass("select-control");
    expect(select).toHaveClass("w-40");
  });

  it("renders a chevron so it matches the custom task picker", () => {
    // The native double-arrow stepper is suppressed in CSS; without this
    // chevron a native dropdown reads as a different control from TaskSelect.
    const { container } = render(<Select aria-label="Optimizer" />);
    expect(container.querySelector(".select-shell")).not.toBeNull();
    const chevron = container.querySelector(".select-chevron");
    expect(chevron).not.toBeNull();
    // Presentational only — the native control owns interaction.
    expect(chevron).toHaveAttribute("aria-hidden", "true");
  });

  it("forwards native select props", () => {
    const onChange = vi.fn();
    render(
      <Select aria-label="Split" defaultValue="train" onChange={onChange} disabled>
        <option value="train">train</option>
      </Select>
    );
    const select = screen.getByRole("combobox", { name: "Split" }) as HTMLSelectElement;
    expect(select.value).toBe("train");
    expect(select).toBeDisabled();
  });
});
