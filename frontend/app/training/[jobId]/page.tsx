import { TrainingDetailPage } from "@/components/platform-pages";

export default function Page({ params }: { params: { jobId: string } }) {
  return <TrainingDetailPage jobId={params.jobId} />;
}
