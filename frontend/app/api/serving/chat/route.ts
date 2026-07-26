// Streaming proxy for the chat SSE endpoint.
//
// The app's `/api/*` config rewrite proxies to the backend, but Next's rewrite
// proxy buffers streaming responses, so chat tokens arrived all at once instead
// of streaming. A route handler that returns the upstream `ReadableStream` body
// directly streams reliably, and — being at this exact path — takes precedence
// over the catch-all rewrite (default rewrites run after filesystem routes).
// The client abort signal is forwarded so stopping generation cancels the
// upstream request, which the backend relays to the llama.cpp server.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const backendOrigin = process.env.BACKEND_PROXY_ORIGIN ?? "http://127.0.0.1:8000";

export async function POST(request: Request): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(`${backendOrigin}/api/serving/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: request.signal
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Upstream request failed";
    return new Response(JSON.stringify({ detail }), {
      status: 502,
      headers: { "Content-Type": "application/json" }
    });
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" }
    });
  }

  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no"
    }
  });
}
