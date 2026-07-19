import type { Metadata } from "next";
import { SignInPage } from "@/features/marketing/signin-page";

export const metadata: Metadata = {
  title: "Sign in · Onestep AI Platform"
};

export default function Page() {
  return <SignInPage />;
}
