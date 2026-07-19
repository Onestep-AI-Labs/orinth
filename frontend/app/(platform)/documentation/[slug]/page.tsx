import { promises as fs } from "fs";
import path from "path";
import { notFound } from "next/navigation";
import { MarkdownViewer } from "@/components/markdown-viewer";
import meta from "@/content/docs/_meta.json";

export async function generateStaticParams() {
  const slugs = Object.keys(meta).filter((slug) => slug !== "index");
  return slugs.map((slug) => ({
    slug,
  }));
}

export default async function DocumentationPage({
  params,
}: {
  params: { slug: string };
}) {
  const { slug } = params;

  // Check if it's a valid slug from our meta
  if (!(slug in meta)) {
    notFound();
  }

  const filePath = path.join(process.cwd(), "content", "docs", `${slug}.md`);
  let content = "";
  try {
    content = await fs.readFile(filePath, "utf8");
  } catch (error) {
    notFound();
  }

  return <MarkdownViewer content={content} />;
}
