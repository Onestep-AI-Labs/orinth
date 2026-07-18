"use client";

import { RouteErrorPanel } from "@/features/platform/ui/route-error";

export default function Error(props: { error: Error & { digest?: string }; reset: () => void }) {
  return <RouteErrorPanel {...props} area="the models area" />;
}
