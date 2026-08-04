import { ArchitectureStudioPage } from "@/components/platform-pages";

export default function Page({ params }: { params: { architectureId: string } }) {
  return <ArchitectureStudioPage architectureId={params.architectureId} />;
}
