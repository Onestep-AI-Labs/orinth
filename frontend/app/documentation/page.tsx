import { promises as fs } from "fs";
import path from "path";
import { notFound } from "next/navigation";
import { MarkdownViewer } from "@/components/markdown-viewer";

export default async function DocumentationIndexPage() {
  const filePath = path.join(process.cwd(), "content", "docs", "index.md");
  let content = "";
  try {
    content = await fs.readFile(filePath, "utf8");
  } catch (error) {
    notFound();
  }

  return <MarkdownViewer content={content} />;
}
