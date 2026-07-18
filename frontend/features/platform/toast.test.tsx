import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { Toaster, toast } from "./toast";

describe("Toaster", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    act(() => {
      vi.runAllTimers();
    });
    vi.useRealTimers();
  });

  it("renders a polite live region", () => {
    render(<Toaster />);
    const viewport = screen.getByRole("status");
    expect(viewport).toHaveAttribute("aria-live", "polite");
  });

  it("shows success toasts and error toasts with role=alert", () => {
    render(<Toaster />);
    act(() => {
      toast.success("Project created");
      toast.error("Backend unreachable");
    });
    expect(screen.getByText("Project created")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Backend unreachable");
  });

  it("renders an action link when provided", () => {
    render(<Toaster />);
    act(() => {
      toast.success("Version created", { action: { label: "Start training", href: "/training" } });
    });
    const action = screen.getByRole("link", { name: "Start training" });
    expect(action).toHaveAttribute("href", "/training");
  });

  it("dismisses via the close button and auto-dismisses after the timeout", () => {
    render(<Toaster />);
    act(() => {
      toast.error("First failure");
      toast.error("Second failure");
    });
    const dismissButtons = screen.getAllByRole("button", { name: "Dismiss notification" });
    act(() => {
      dismissButtons[0].click();
    });
    expect(screen.queryByText("First failure")).not.toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(6500);
    });
    expect(screen.queryByText("Second failure")).not.toBeInTheDocument();
  });

  it("keeps at most three toasts", () => {
    render(<Toaster />);
    act(() => {
      toast.success("one");
      toast.success("two");
      toast.success("three");
      toast.success("four");
    });
    expect(screen.queryByText("one")).not.toBeInTheDocument();
    expect(screen.getByText("four")).toBeInTheDocument();
  });
});
