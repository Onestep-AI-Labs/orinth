import { TestingDetailPage } from "@/components/platform-pages";

export default function Page({ params }: { params: { jobId: string } }) {
  return <TestingDetailPage jobId={params.jobId} />;
}
