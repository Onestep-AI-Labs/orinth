import { ProjectSettingsPage } from "@/components/platform-pages";

export default async function Page({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  return <ProjectSettingsPage projectId={projectId} />;
}
