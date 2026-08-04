/** @type {import('next').NextConfig} */
const backendOrigin = process.env.BACKEND_PROXY_ORIGIN ?? "http://127.0.0.1:8000";
const isDev = process.env.NODE_ENV !== "production";

// Next 14 App Router emits inline hydration scripts, so script-src needs
// 'unsafe-inline' without a nonce+strict-dynamic middleware (which would
// force whole-app dynamic rendering). /api and /media stay same-origin via
// rewrites; next/font self-hosts, so 'self' covers fonts. If
// NEXT_PUBLIC_BACKEND_URL points the browser at the backend directly, add
// that origin to connect-src and img-src.
const contentSecurityPolicy = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${isDev ? " 'unsafe-eval'" : ""}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' blob: data:",
  "font-src 'self'",
  `connect-src 'self'${isDev ? " ws: wss:" : ""}`,
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'"
].join("; ");

const securityHeaders = [
  { key: "Content-Security-Policy", value: contentSecurityPolicy },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" }
];

// The desktop app (see specs/phase-18-macos-desktop-app.md) runs this server
// from an app bundle, where the repo's node_modules do not exist. Standalone
// output emits a self-contained server.js plus a pruned node_modules. It is
// gated on the flag so `pnpm build` / `make build` keep their existing output.
// Standalone mode delegates image optimization to `sharp`, which is a native
// dependency the desktop bundle would have to ship per-architecture. The app
// serves its own bundled screenshots over loopback, where optimization buys
// nothing, so it is turned off for that build only.
const isDesktopBuild = process.env.DESKTOP_BUILD === "1";

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  ...(isDesktopBuild ? { output: "standalone", images: { unoptimized: true } } : {}),
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders
      }
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendOrigin}/api/:path*`
      },
      {
        source: "/media/:path*",
        destination: `${backendOrigin}/media/:path*`
      }
    ];
  }
};

export default nextConfig;
