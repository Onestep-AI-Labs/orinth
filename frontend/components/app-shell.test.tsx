import { useEffect } from "react";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider, useMutation } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("next/navigation", () => ({
  usePathname: () => "/datasets",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => new URLSearchParams()
}));

import { api } from "@/lib/api";
import { AppShell } from "@/components/app-shell";
import { usePublishForegroundJob } from "@/features/platform/foreground-job";

/** A mutation that never settles, so the shell's blocking overlay stays up. */
function Busy() {
  const mutation = useMutation({ mutationFn: () => new Promise<void>(() => {}) });
  useEffect(() => {
    mutation.mutate();
    // Firing once on mount is the point; `mutate` is stable enough for it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

function Publisher() {
  usePublishForegroundJob({
    label: "Writing record 8,412 of 41,003",
    percent: 84,
    stages: [
      { key: "detecting", label: "Read the files", state: "done" },
      { key: "applying", label: "Build the splits", state: "active" }
    ]
  });
  return null;
}

function renderShell(children: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <AppShell>{children}</AppShell>
    </QueryClientProvider>
  );
}

describe("the shell's blocking overlay", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, "projects").mockResolvedValue([] as never);
  });

  it("says 'Working' only when the work cannot describe itself", async () => {
    renderShell(<Busy />);
    expect(await screen.findByText("Working")).toBeInTheDocument();
  });

  it("shows the run's own progress bar instead, when one is published", async () => {
    // The defect this exists for: a prep run drew a determinate bar *behind*
    // this overlay, so a forty-thousand-row import showed a modal saying one
    // word over a readout nobody could see.
    renderShell(
      <>
        <Busy />
        <Publisher />
      </>
    );

    // Wait for the overlay itself, not for "Working" to vanish: it is never
    // rendered on this path, so waiting for its absence passes before the
    // overlay exists at all and asserts nothing.
    expect(await screen.findByText("Writing record 8,412 of 41,003")).toBeInTheDocument();
    expect(screen.queryByText("Working")).not.toBeInTheDocument();
    expect(screen.getByText("84%")).toBeInTheDocument();
    expect(screen.getAllByRole("progressbar")[0]).toHaveAttribute("aria-valuenow", "84");
    expect(screen.getByText("Build the splits").className).toContain("prep-stage-active");
  });
});
