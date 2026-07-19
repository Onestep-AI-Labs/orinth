import { useState } from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NumberInput } from "./index";

function Harness({ initial = 10, min, max }: { initial?: number; min?: number; max?: number }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <NumberInput value={value} onChange={setValue} min={min} max={max} />
      <output data-testid="committed">{value}</output>
    </>
  );
}

describe("NumberInput", () => {
  it("lets the field be cleared instead of snapping to 0", async () => {
    // Coercing "" to 0 on change rewrites the box under the caret, so clearing
    // a value to retype it left you editing "0" with the cursor misplaced.
    const user = userEvent.setup();
    render(<Harness initial={50} />);
    const input = screen.getByRole("spinbutton");

    await user.clear(input);

    expect(input).toHaveValue(null);
    expect(screen.getByTestId("committed")).toHaveTextContent("50");
  });

  it("commits what was typed after clearing", async () => {
    const user = userEvent.setup();
    render(<Harness initial={50} />);
    const input = screen.getByRole("spinbutton");

    await user.clear(input);
    await user.type(input, "8");

    expect(screen.getByTestId("committed")).toHaveTextContent("8");
  });

  it("restores the last good value when left empty", async () => {
    const user = userEvent.setup();
    render(<Harness initial={12} />);
    const input = screen.getByRole("spinbutton");

    await user.clear(input);
    await user.tab();

    expect(input).toHaveValue(12);
    expect(screen.getByTestId("committed")).toHaveTextContent("12");
  });

  it("clamps to the allowed range on blur", async () => {
    const user = userEvent.setup();
    render(<Harness initial={10} min={1} max={100} />);
    const input = screen.getByRole("spinbutton");

    await user.clear(input);
    await user.type(input, "500");
    await user.tab();

    expect(screen.getByTestId("committed")).toHaveTextContent("100");
  });
});
