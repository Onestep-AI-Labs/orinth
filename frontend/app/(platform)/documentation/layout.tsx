import Link from "next/link";
import { BookOpen } from "lucide-react";
import meta from "@/content/docs/_meta.json";

export default function DocumentationLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const links = Object.entries(meta).map(([slug, title]) => ({
    href: slug === "index" ? "/documentation" : `/documentation/${slug}`,
    label: title as string,
  }));

  return (
    <div className="flex min-h-[calc(100vh-theme(spacing.14))]">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r border-line shrink-0 hidden md:flex flex-col h-[calc(100vh-theme(spacing.14))] sticky top-14 overflow-y-auto bg-panel/50">
        <div className="p-4 border-b border-line flex items-center gap-2">
          <BookOpen className="text-accent" size={18} />
          <h2 className="font-semibold text-sm text-ink uppercase tracking-wider">Guide</h2>
        </div>
        <nav className="p-4 space-y-1">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="block px-3 py-2 text-sm text-ink/80 hover:text-ink hover:bg-black/5 rounded-md transition-colors font-medium"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 max-w-4xl px-8 py-10 min-w-0 mx-auto">
        <div className="prose prose-slate max-w-none prose-a:text-accent hover:prose-a:text-accent/80 prose-headings:font-semibold prose-h1:text-3xl prose-h2:text-2xl prose-h3:text-xl prose-img:rounded-xl">
          {children}
        </div>
      </main>
    </div>
  );
}
