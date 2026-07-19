import { Compass } from "lucide-react";
import { ButtonLink, EmptyState } from "@/features/platform/ui";

export default function NotFound() {
  return (
    <div className="space-y-5">
      <header className="page-header">
        <div>
          <span>
            <Compass size={22} />
          </span>
          <div>
            <h2>Page not found</h2>
            <p>The route you requested does not exist in this workspace.</p>
          </div>
        </div>
      </header>
      <section className="panel">
        <EmptyState
          centered
          icon={<Compass size={32} />}
          label="Nothing here"
          description="Check the URL or head back to your projects."
          action={<ButtonLink href="/projects">Back to projects</ButtonLink>}
        />
      </section>
    </div>
  );
}
