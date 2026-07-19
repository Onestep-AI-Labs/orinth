import { useState } from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MultiSelect } from "./index";

const OPTIONS = [
  { value: "a", label: "Keyword Text Classifier" },
  { value: "b", label: "distilbert-test" },
  { value: "c", label: "roberta-test" }
];

function Harness({ initial = [] as string[] }) {
  const [values, setValues] = useState<string[]>(initial);
  return (
    <>
      <MultiSelect options={OPTIONS} values={values} onChange={setValues} placeholder="Choose one or more models…" />
      <output data-testid="selected">{values.join(",")}</output>
    </>
  );
}

describe("MultiSelect", () => {
  it("starts closed with nothing selected", () => {
    render(<Harness />);
    expect(screen.getByRole("button", { name: /Choose one or more models/ })).toBeInTheDocument();
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(screen.getByTestId("selected")).toHaveTextContent("");
  });

  it("selects several options without closing between picks", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole("button"));
    await user.click(screen.getByLabelText("distilbert-test"));
    await user.click(screen.getByLabelText("roberta-test"));

    expect(screen.getByTestId("selected")).toHaveTextContent("b,c");
  });

  it("deselects an option that is picked twice", async () => {
    const user = userEvent.setup();
    render(<Harness initial={["b"]} />);

    await user.click(screen.getByRole("button"));
    await user.click(screen.getByLabelText("distilbert-test"));

    expect(screen.getByTestId("selected")).toHaveTextContent("");
  });

  it("summarises the selection on the closed trigger", async () => {
    const user = userEvent.setup();
    render(<Harness initial={["a", "b"]} />);
    expect(screen.getByRole("button")).toHaveTextContent("Keyword Text Classifier, distilbert-test");

    // Past two, names would overflow the trigger, so it switches to a count.
    await user.click(screen.getByRole("button"));
    await user.click(screen.getByLabelText("roberta-test"));
    expect(screen.getByRole("button", { name: /3 selected/ })).toBeInTheDocument();
  });

  it("closes on Escape and on a click outside", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole("button"));
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button"));
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    await user.click(document.body);
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("disables the trigger when there is nothing to choose", () => {
    render(<MultiSelect options={[]} values={[]} onChange={() => {}} />);
    expect(screen.getByRole("button")).toBeDisabled();
  });
});
