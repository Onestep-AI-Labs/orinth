/** @type {import('next').NextConfig} */
const backendOrigin = process.env.BACKEND_PROXY_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
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
