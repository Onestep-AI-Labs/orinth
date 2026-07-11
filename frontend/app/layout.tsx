import type { Metadata } from "next";
import { AppShell } from "@/components/app-shell";
import { Providers } from "./providers";
import "./globals.css";
import "./styles/platform.css";

export const metadata: Metadata = {
  title: "Onestep AI Platform",
  description: "One workspace for image intelligence",
  icons: {
    icon: "/brand/logo_transparent.png"
  }
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
