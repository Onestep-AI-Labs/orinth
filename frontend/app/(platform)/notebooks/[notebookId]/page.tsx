import { NotebookPage } from "@/features/notebooks/notebook-page";

export default async function Page({ params }: { params: Promise<{ notebookId: string }> }) {
  const { notebookId } = await params;
  return <NotebookPage notebookId={notebookId} />;
}
