import type { Metadata } from "next";
import { SignInPage } from "@/features/auth/auth-page";

export const metadata: Metadata = {
  title: "Sign in · Orinth"
};

export default function Page() {
  return <SignInPage />;
}
