import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CanvasContextMenu } from "./context-menu";

/**
 * The dismiss-on-outside-press listener runs on `pointerdown` in the capture
 * phase, which is one step ahead of the `click` the rows are wired to. Without
 * the `contains` guard it tore the menu down before any command could fire, so
 * every entry looked inert — these are the tests that keep that from returning.
 */
describe("CanvasContextMenu", () => {
  const at = { x: 40, y: 40 };

  // Exactly one close is the load-bearing assertion: a second one means the
  // outside-press listener also fired, which in a real browser is the moment
  // the row is unmounted and its `click` is never dispatched.
  it("runs a command when its row is clicked, and closes exactly once", async () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    render(
      <CanvasContextMenu
        state={{ ...at, items: [{ label: "Duplicate", onSelect }] }}
        onClose={onClose}
      />
    );

    await userEvent.click(screen.getByRole("menuitem", { name: /Duplicate/ }));

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not run a disabled command", async () => {
    const onSelect = vi.fn();
    render(
      <CanvasContextMenu
        state={{ ...at, items: [{ label: "Paste", disabled: true, onSelect }] }}
        onClose={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("menuitem", { name: /Paste/ }));

    expect(onSelect).not.toHaveBeenCalled();
  });

  it("closes without running anything when the press lands outside", async () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    render(
      <div>
        <button type="button">Canvas</button>
        <CanvasContextMenu
          state={{ ...at, items: [{ label: "Delete", onSelect }] }}
          onClose={onClose}
        />
      </div>
    );

    await userEvent.click(screen.getByRole("button", { name: "Canvas" }));

    expect(onSelect).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it("closes on Escape", async () => {
    const onClose = vi.fn();
    render(
      <CanvasContextMenu
        state={{ ...at, items: [{ label: "Copy", onSelect: vi.fn() }] }}
        onClose={onClose}
      />
    );

    await userEvent.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalled();
  });
});
