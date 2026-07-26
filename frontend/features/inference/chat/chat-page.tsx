"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  Brain,
  ChevronDown,
  Globe,
  MessagesSquare,
  Send,
  Square,
  StopCircle
} from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ChatStreamEvent, WebSearchResult } from "@/lib/api/serving";
import { formatSeconds } from "@/features/platform/utils";
import { Badge, Button, Field, InlineSpinner, PageHeader, SliderField } from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import { ChatMessageContent, CopyButton } from "@/features/inference/chat/message-content";
import { ModelSourcePicker, type StartTarget } from "@/features/inference/model-source-picker";
import type { ModelInfo, ServingStatus } from "@/types/api";

type Turn = {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  citations?: WebSearchResult[];
  searching?: boolean;
};

type LastStats = {
  tokensPerSecond: number;
  timeToFirstToken: number | null;
  completionTokens: number;
};

function servingTone(state: string): "ok" | "info" | "fail" | "neutral" {
  if (state === "running") return "ok";
  if (state === "starting" || state === "stopping") return "info";
  return "neutral";
}

let turnCounter = 0;
function nextId(): string {
  turnCounter += 1;
  return `turn-${turnCounter}`;
}

export function ChatPage() {
  const queryClient = useQueryClient();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [system, setSystem] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.95);
  // Reasoning models spend many tokens inside <think>; a low cap truncates
  // before the closing tag so the answer never lands outside the block.
  const [maxTokens, setMaxTokens] = useState(2048);
  const [showThinking, setShowThinking] = useState(true);
  const [searchOn, setSearchOn] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [lastStats, setLastStats] = useState<LastStats | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);

  const statusQuery = useQuery({
    queryKey: ["serving-status"],
    queryFn: () => api.servingStatus(),
    refetchInterval: (query) => {
      const state = (query.state.data as ServingStatus | undefined)?.state;
      return state === "starting" || state === "stopping" ? 1500 : 4000;
    }
  });
  const status = statusQuery.data;
  const running = status?.state === "running";

  const modelsQuery = useQuery({
    queryKey: ["models", "catalog", "all"],
    queryFn: () => api.models(false)
  });
  const servableModels = useMemo(
    () => (modelsQuery.data ?? []).filter((model) => model.family === "llm_gguf" && model.available),
    [modelsQuery.data]
  );

  useEffect(() => {
    const node = transcriptRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [turns]);

  useEffect(() => {
    // Cancel any in-flight generation when the surface unmounts.
    return () => abortRef.current?.abort();
  }, []);

  const startServing = useMutation({
    mutationKey: ["serving-start"],
    mutationFn: (target: { model_id?: string; model_path?: string }) => api.startServing(target),
    onSuccess: async (next) => {
      toast.success(`Serving ${next.model_name ?? "model"}`);
      await queryClient.invalidateQueries({ queryKey: ["serving-status"] });
    },
    onError: (error: Error) => toast.error(error.message)
  });
  const stopServing = useMutation({
    mutationFn: () => api.stopServing(),
    onSuccess: async () => {
      toast.success("Serving stopped");
      await queryClient.invalidateQueries({ queryKey: ["serving-status"] });
    }
  });

  function appendTurn(turn: Turn) {
    setTurns((prev) => [...prev, turn]);
  }

  function updateAssistant(id: string, updater: (content: string) => string, streamingFlag?: boolean) {
    setTurns((prev) =>
      prev.map((turn) =>
        turn.id === id
          ? { ...turn, content: updater(turn.content), streaming: streamingFlag ?? turn.streaming }
          : turn
      )
    );
  }

  async function send() {
    const text = draft.trim();
    if (!text || streaming || !running) return;
    setDraft("");
    setLastStats(null);
    const userTurn: Turn = { id: nextId(), role: "user", content: text };
    const assistantId = nextId();
    const history = [...turns, userTurn];
    setTurns([...history, { id: assistantId, role: "assistant", content: "", streaming: true, searching: searchOn }]);

    const controller = new AbortController();
    abortRef.current = controller;
    setStreaming(true);

    // Web search mode: fetch results first, show them as citations, and ground
    // the model with them. Failures fall through to an ordinary answer.
    let searchContext = "";
    let citations: WebSearchResult[] = [];
    if (searchOn) {
      try {
        const result = await api.webSearch(text, 5);
        citations = result.results;
        if (result.error) toast.error(`Web search: ${result.error}`);
        if (citations.length > 0) {
          searchContext =
            "Web search results — cite relevant ones inline as [n] and list the sources you used:\n" +
            citations
              .map((c, i) => `[${i + 1}] ${c.title}\n${c.url}\n${c.snippet}`)
              .join("\n\n");
        }
      } catch (error) {
        toast.error(`Web search failed: ${(error as Error).message}`);
      }
    }
    setTurns((prev) =>
      prev.map((turn) =>
        turn.id === assistantId ? { ...turn, searching: false, citations } : turn
      )
    );

    try {
      const messages = history.map((turn) => ({ role: turn.role, content: turn.content }));
      const systemPrompt = [system.trim(), searchContext].filter(Boolean).join("\n\n");
      const stream = api.streamChat(
        {
          messages,
          system: systemPrompt || null,
          temperature,
          top_p: topP,
          max_tokens: maxTokens
        },
        controller.signal
      );
      for await (const event of stream as AsyncGenerator<ChatStreamEvent>) {
        if (event.type === "delta") {
          updateAssistant(assistantId, (content) => content + event.content);
        } else if (event.type === "usage") {
          setLastStats({
            tokensPerSecond: event.tokens_per_second,
            timeToFirstToken: event.time_to_first_token_seconds,
            completionTokens: event.completion_tokens
          });
        } else if (event.type === "error") {
          updateAssistant(
            assistantId,
            (content) => content + (content ? "\n\n" : "") + `⚠ ${event.message}`
          );
          toast.error(event.message);
          await queryClient.invalidateQueries({ queryKey: ["serving-status"] });
        }
      }
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        const message = (error as Error).message;
        updateAssistant(assistantId, (content) => content + (content ? "\n\n" : "") + `⚠ ${message}`);
        toast.error(message);
      }
    } finally {
      updateAssistant(assistantId, (content) => content, false);
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function stopGeneration() {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setTurns((prev) => prev.map((turn) => (turn.streaming ? { ...turn, streaming: false } : turn)));
  }

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Inference"
        title="Chat"
        subtitle="Converse with a served GGUF model. This is a research instrument, not a medical assistant."
        icon={<MessagesSquare size={20} />}
        actions={
          <Link className="secondary-button" href="/inference">
            <ArrowLeft size={16} /> Back to inference
          </Link>
        }
      />

      <div className="chat-layout">
        <aside className="panel chat-rail">
          <ServingControl
            status={status}
            running={running}
            servableModels={servableModels}
            onStart={(target) => startServing.mutate(target)}
            onStop={() => stopServing.mutate()}
            busy={startServing.isPending || stopServing.isPending}
          />

          <div className="chat-rail-section">
            <span className="chat-rail-label">Sampler</span>
            <SliderField label="Temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} />
            <SliderField label="Top-p" value={topP} min={0} max={1} step={0.01} onChange={setTopP} />
            <SliderField label="Max tokens" value={maxTokens} min={16} max={8192} step={16} onChange={setMaxTokens} />
          </div>

          <div className="chat-rail-section">
            <Field label="System prompt">
              <textarea
                value={system}
                onChange={(event) => setSystem(event.target.value)}
                rows={4}
                placeholder="Optional. Sets the assistant's behavior for the conversation."
              />
            </Field>
          </div>
        </aside>

        <section className="panel chat-column">
          <div className="chat-transcript" ref={transcriptRef}>
            {turns.length === 0 ? (
              <div className="chat-empty">
                <MessagesSquare size={28} />
                <strong>Start a conversation</strong>
                <span>
                  {running
                    ? "Type a message below to chat with the served model."
                    : "Start serving a GGUF model from the left rail, then send a message."}
                </span>
              </div>
            ) : (
              turns.map((turn) => <TranscriptRow key={turn.id} turn={turn} showThinking={showThinking} />)
            )}
          </div>

          <div className="chat-footer-stats">
            <div className="chat-footer-stats-inner">
              {lastStats ? (
                <span>
                  {lastStats.completionTokens} tokens · {lastStats.tokensPerSecond.toFixed(1)} tok/s
                  {lastStats.timeToFirstToken != null
                    ? ` · first token ${lastStats.timeToFirstToken.toFixed(2)}s`
                    : ""}
                </span>
              ) : (
                <span className="chat-footer-idle">Ready</span>
              )}
            </div>
          </div>

          <div className="chat-composer">
            <div className="chat-composer-inner">
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    send();
                  }
                }}
                rows={2}
                placeholder={running ? "Send a message…" : "Start serving a model to chat"}
                disabled={!running || streaming}
              />
              {streaming ? (
                <Button variant="secondary" onClick={stopGeneration}>
                  <StopCircle size={17} /> Stop
                </Button>
              ) : (
                <Button onClick={send} disabled={!running || !draft.trim()}>
                  <Send size={17} /> Send
                </Button>
              )}
            </div>
            <div className="chat-composer-chips">
              <button
                type="button"
                className={`chat-mode-chip${searchOn ? " chat-mode-chip-active" : ""}`}
                onClick={() => setSearchOn((on) => !on)}
                aria-pressed={searchOn}
                title="Search the web and cite sources"
              >
                <Globe size={14} /> Search
              </button>
              <button
                type="button"
                className={`chat-mode-chip${showThinking ? " chat-mode-chip-active" : ""}`}
                onClick={() => setShowThinking((on) => !on)}
                aria-pressed={showThinking}
                title="Show or hide the model's reasoning"
              >
                <Brain size={14} /> Thinking
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function ServingControl({
  status,
  running,
  servableModels,
  onStart,
  onStop,
  busy
}: {
  status: ServingStatus | undefined;
  running: boolean;
  servableModels: ModelInfo[];
  onStart: (target: StartTarget) => void;
  onStop: () => void;
  busy: boolean;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const state = status?.state ?? "stopped";

  function serve(target: StartTarget) {
    setPickerOpen(false);
    onStart(target);
  }

  return (
    <div className="chat-rail-section chat-serving-control">
      <span className="chat-rail-label">Served model</span>
      <div className="serving-status-row">
        <Badge tone={servingTone(state)}>{state}</Badge>
        {running && status?.uptime_seconds != null ? (
          <span className="serving-status-meta">up {formatSeconds(status.uptime_seconds)}</span>
        ) : null}
      </div>

      {running ? (
        <>
          <p className="chat-served-name">{status?.model_name}</p>
          <Button variant="secondary" onClick={onStop} disabled={busy}>
            <Square size={16} /> Stop
          </Button>
        </>
      ) : null}

      <div className="chat-model-picker">
        <button
          type="button"
          className="chat-model-picker-trigger"
          aria-haspopup="dialog"
          aria-expanded={pickerOpen}
          onClick={() => setPickerOpen((open) => !open)}
          disabled={busy}
        >
          <span>{state === "starting" ? "Starting…" : running ? "Switch model" : "Select model"}</span>
          <ChevronDown size={16} />
        </button>
        {pickerOpen && (
          <div className="chat-model-picker-panel">
            <ModelSourcePicker registeredModels={servableModels} onServe={serve} disabled={busy} />
          </div>
        )}
      </div>
    </div>
  );
}

function TranscriptRow({ turn, showThinking }: { turn: Turn; showThinking: boolean }) {
  const isAssistant = turn.role === "assistant";
  return (
    <div className="chat-turn">
      <div className="chat-turn-head">
        <span className="chat-turn-role">{isAssistant ? "Assistant" : "You"}</span>
        {isAssistant && !turn.streaming && turn.content.trim().length > 0 && (
          <CopyButton value={turn.content} label="Copy message" />
        )}
      </div>
      <div className="chat-turn-body">
        {turn.searching ? (
          <div className="chat-searching">
            <InlineSpinner label="Searching the web" />
          </div>
        ) : null}
        {turn.citations && turn.citations.length > 0 ? <Citations citations={turn.citations} /> : null}
        <ChatMessageContent content={turn.content} streaming={turn.streaming} showThinking={showThinking} />
        {turn.streaming ? <span className="chat-caret" aria-hidden="true" /> : null}
      </div>
    </div>
  );
}

function Citations({ citations }: { citations: WebSearchResult[] }) {
  return (
    <div className="chat-citations">
      <span className="chat-citations-label">
        <Globe size={13} /> Sources
      </span>
      <ol className="chat-citations-list">
        {citations.map((source, index) => (
          <li key={source.url + index}>
            <a href={source.url} target="_blank" rel="noopener noreferrer" title={source.url}>
              <span className="chat-citation-index">[{index + 1}]</span>
              {source.title}
            </a>
            <span className="chat-citation-host">{hostOf(source.url)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}
