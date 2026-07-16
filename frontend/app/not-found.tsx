import Link from "next/link";
import { Compass } from "lucide-react";

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
        <div className="empty-state empty-state-centered">
          <span className="empty-state-icon">
            <Compass size={32} />
          </span>
          <strong>Nothing here</strong>
          <span>Check the URL or head back to the dashboard.</span>
        </div>
        <div className="flex items-center justify-center" style={{ marginTop: 16 }}>
          <Link className="primary-button" href="/">
            Back to dashboard
          </Link>
        </div>
      </section>
    </div>
  );
}
