import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Providers } from "./providers";
import "./globals.css";
import "./styles/platform.css";
import "./styles/marketing.css";

// One face. Display type is Inter 500 with tightened tracking (see DESIGN.md §3);
// --font-display aliases --font-sans in globals.css.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans"
});

export const metadata: Metadata = {
  title: "Onestep AI Platform",
  description: "One workspace for vision and NLP intelligence",
  icons: {
    icon: "/brand/logo_transparent.png"
  }
};

// The shell lives in app/(platform)/layout.tsx, not here — the marketing group
// renders its own chrome and must not mount the sidebar or the project context.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.variable}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
