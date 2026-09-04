import { type UIEvent, useEffect, useRef, useState } from "react";
import { ArrowUp, Square, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Markdown } from "@/components/studio/markdown";
import {
  clearPlayground,
  deltaCompletionTokens,
  deltaContent,
  deltaReasoning,
  getPlayground,
  getProgress,
  modelIsBusy,
  modelIsLive,
  postStop,
  putPlayground,
  type ModelProgress,
  type PlaygroundMetrics,
  type PlaygroundTurn,
} from "@/lib/edge-api";
import { loadTarget, publicName } from "@/lib/models";
import { useStudio } from "@/lib/studio-store";
import { cn } from "@/lib/utils";
import type { EngineKind } from "@/lib/flags";

type ChatTurn = PlaygroundTurn;


export function Playground() {
  const model = useStudio((s) => s.selected());
  const served = useStudio((s) => s.served);
  const live = modelIsLive(served, model);

  if (model?.engine === "embed") {
    return <EmbedPlayground live={live} loadedCount={served.length} />;
  }
  if (model?.engine === "tts" || model?.engine === "stt" || model?.engine === "rerank" || model?.engine === "image") {
    return <AudioHint engine={model.engine} live={live} />;
  }

  return <ChatPlayground live={live} />;
}

function EmbedPlayground({ live, loadedCount }: { live: boolean; loadedCount: number }) {
  const model = useStudio((s) => s.selected());
  const progressSnap = useStudio((s) => s.progress);
  const [input, setInput] = useState("what is the capital of France?");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ dims: number; preview: number[]; tokens: number } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const remoteBusy = modelIsBusy(progressSnap, model);
  const stopping = busy || remoteBusy;

  async function stop() {
    abortRef.current?.abort();
    if (model) {
      try {
        await postStop(loadTarget(model));
      } catch {
        /* already gone */
      }
    }
  }

  async function run() {
    const text = input.trim();
    if (!text || !model || !live || stopping) return;
    setBusy(true);
    setError(null);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const res = await fetch("/v1/embeddings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: loadTarget(model), input: text }),
        signal: ac.signal,
      });
      const body = (await res.json().catch(() => ({}))) as {
        error?: { message?: string };
        data?: { embedding?: number[] }[];
        usage?: { prompt_tokens?: number };
      };
      if (!res.ok) throw new Error(body.error?.message || `HTTP ${res.status}`);
      const vector = body.data?.[0]?.embedding ?? [];
      setResult({
        dims: vector.length,
        preview: vector.slice(0, 8),
        tokens: Number(body.usage?.prompt_tokens || 0),
      });
    } catch (err) {
      if (ac.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      if (abortRef.current === ac) abortRef.current = null;
      setBusy(false);
    }
  }

  if (!live) {
    return (
      <div className="flex flex-1 flex-col items-start justify-center">
        <p className="max-w-md text-sm text-muted-foreground">
          Serve this embedding model. It answers POST /v1/embeddings on the same /v1 as your chat
          models — RAG does not unload them.
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-1 flex-col">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
        <p className="pt-6 text-sm text-muted-foreground">
          POST /v1/embeddings on the already-loaded engine. Keep a chat model served too — embedding
          inference is a separate process, so it does not add delay to the next chat token.
          {loadedCount > 1 ? ` ${loadedCount} loaded on this origin.` : ""}
        </p>
        {result ? (
          <article className="mr-auto max-w-[85%] rounded-2xl bg-card px-4 py-3 text-sm shadow-[var(--shadow-border)]">
            <p className="mb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">embedding</p>
            <p className="font-mono text-xs leading-relaxed">
              {result.dims} dims
              {result.tokens ? ` · ${result.tokens} tokens` : ""}
            </p>
            <p className="mt-2 font-mono text-xs break-all text-muted-foreground">
              [{result.preview.map((n) => n.toFixed(4)).join(", ")}
              {result.dims > result.preview.length ? ", …" : ""}]
            </p>
          </article>
        ) : null}
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
      </div>
      <form
        className="mt-4 flex items-end gap-2 rounded-2xl border border-border bg-card p-2 shadow-[var(--shadow-border)]"
        onSubmit={(e) => {
          e.preventDefault();
          void run();
        }}
      >
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={`Embed · ${model?.name ?? "OpenAI /v1"}`}
          className="min-h-14 border-0 bg-transparent shadow-none focus-visible:ring-0"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!stopping) void run();
            }
          }}
        />
        {stopping ? (
          <Button type="button" size="icon" className="size-11 shrink-0" onClick={() => void stop()} aria-label="Stop">
            <Square />
          </Button>
        ) : (
          <Button type="submit" size="icon" className="size-11 shrink-0" disabled={!input.trim()} aria-label="Embed">
            <ArrowUp />
          </Button>
        )}
      </form>
    </div>
  );
}

function AudioHint({ engine, live }: { engine: EngineKind; live: boolean }) {
  const route =
    engine === "tts"
      ? "/v1/audio/speech"
      : engine === "stt"
        ? "/v1/audio/transcriptions"
        : engine === "rerank"
          ? "/v1/rerank"
          : "/v1/images/generations";
  return (
    <div className="flex flex-1 flex-col items-start justify-center">
      <p className="max-w-md text-sm text-muted-foreground">
        {live
          ? `Playground is text-only. This ${engine.toUpperCase()} model is served — POST ${route} from any OpenAI-compatible client. See Endpoint for more information on how to use this model.`
          : `Serve this ${engine.toUpperCase()} model to expose ${route}. Playground stays text-only. See Endpoint for more information on how to use this model.`}
      </p>
    </div>
  );
}

function ChatPlayground({ live }: { live: boolean }) {
  const model = useStudio((s) => s.selected());
  const served = useStudio((s) => s.served);
  const flags = useStudio((s) => s.flags);
  const progressSnap = useStudio((s) => s.progress);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ModelProgress | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const stickRef = useRef(true);
  const programmaticRef = useRef(false);
  const readyRef = useRef(false);
  const genTokensRef = useRef(0);
  const remoteBusy = modelIsBusy(progressSnap, model);
  const stopping = busy || remoteBusy;

  useEffect(() => {
    let cancelled = false;
    void getPlayground()
      .then((loaded) => {
        if (cancelled) return;
        setTurns(loaded);
        readyRef.current = true;
      })
      .catch(() => {
        if (!cancelled) readyRef.current = true;
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!readyRef.current) return;
    const handle = window.setTimeout(() => {
      void putPlayground(turns);
    }, 250);
    return () => window.clearTimeout(handle);
  }, [turns]);

  function follow() {
    const el = scrollerRef.current;
    if (!el || !stickRef.current) return;
    programmaticRef.current = true;
    el.scrollTop = el.scrollHeight;
    requestAnimationFrame(() => {
      programmaticRef.current = false;
    });
  }

  function onScroll(ev: UIEvent<HTMLDivElement>) {
    if (programmaticRef.current) return;
    const el = ev.currentTarget;
    const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickRef.current = gap < 72;
  }

  useEffect(() => {
    follow();
  }, [turns, stopping, progress]);

  useEffect(() => {
    if (!stopping || !model) {
      setProgress(null);
      return;
    }
    let stop = false;
    async function tick() {
      try {
        const snap = await getProgress(publicName(model!));
        if (!stop) {
          const row = snap.models[0] ?? null;
          setProgress(row);
          const n = row?.generation?.tokens ?? 0;
          if (n > 0) genTokensRef.current = n;
        }
      } catch {
        /* preview or network blip */
      }
      if (!stop) window.setTimeout(tick, 200);
    }
    void tick();
    return () => {
      stop = true;
    };
  }, [stopping, model]);

  async function halt() {
    abortRef.current?.abort();
    if (model) {
      try {
        await postStop(loadTarget(model));
      } catch {
        /* already gone */
      }
    }
  }

  async function rewind(index: number) {
    if (stopping) await halt();
    setTurns((prev) => prev.slice(0, index));
  }

  async function reset() {
    if (stopping) await halt();
    setTurns([]);
    try {
      await clearPlayground();
    } catch {
      /* RAM miss is fine */
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || !model || !live || stopping) return;
    setInput("");
    setError(null);
    stickRef.current = true;
    const next = [...turns, { role: "user" as const, text }];
    setTurns(next);
    setBusy(true);
    let assistant = "";
    let reasoning = "";
    const ac = new AbortController();
    abortRef.current = ac;
    genTokensRef.current = 0;
    const started = performance.now();
    let firstAt = 0;
    let usageTokens = 0;
    const modelLabel = publicName(model);
    const foldReason = Boolean(flags.streamReasonToResponse);
    const paint = (value: string, thinkText = "", metrics?: PlaygroundMetrics) =>
      setTurns([
        ...next,
        {
          role: "assistant",
          text: value,
          thinking: foldReason ? undefined : thinkText || undefined,
          metrics,
        },
      ]);
    const finish = (value: string, thinkText = "") => {
      const end = performance.now();
      const ttft = ((firstAt || end) - started) / 1000;
      const gen = firstAt ? Math.max(0, (end - firstAt) / 1000) : 0;
      const tokens = usageTokens || genTokensRef.current || estimateTokens(`${value} ${thinkText}`);
      const tps = gen > 0 ? tokens / gen : 0;
      paint(value, thinkText, { ttft, gen, tokens, tps, model: modelLabel });
    };
    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: loadTarget(model),
          messages: next.map((t) => ({ role: t.role, content: t.text })),
          max_tokens: Number(flags.maxTokens) || 512,
          stream: true,
        }),
        signal: ac.signal,
      });
      const ctype = res.headers.get("content-type") || "";
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
        throw new Error(body.error?.message || `HTTP ${res.status}`);
      }
      if (ctype.includes("text/event-stream") && res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let acc = "";
        paint("");
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          acc += decoder.decode(value, { stream: true });
          const frames = acc.split("\n\n");
          acc = frames.pop() ?? "";
          for (const frame of frames) {
            for (const line of frame.split("\n")) {
              if (!line.startsWith("data:")) continue;
              const data = line.slice(5).trim();
              if (!data || data === "[DONE]") continue;
              try {
                const payload = JSON.parse(data);
                const piece = deltaContent(payload);
                const think = deltaReasoning(payload);
                const used = deltaCompletionTokens(payload);
                if (used) usageTokens = used;
                if (foldReason) {
                  if (think) assistant += think;
                  if (piece) {
                    if (piece === assistant || (assistant && piece.trim() === assistant.trim())) {
                      /* trailing dump of the reasoning stream — already shown */
                    } else if (assistant && piece.startsWith(assistant)) {
                      assistant = piece;
                    } else {
                      assistant += piece;
                    }
                  }
                  if (think || piece) {
                    if (!firstAt) firstAt = performance.now();
                    paint(assistant, "");
                  }
                } else {
                  if (think) reasoning += think;
                  if (piece) assistant += piece;
                  if (piece || think) {
                    if (!firstAt) firstAt = performance.now();
                    paint(assistant, reasoning);
                  }
                }
              } catch {
                /* ignore malformed chunk */
              }
            }
          }
        }
        if (foldReason) {
          if (!assistant) finish(reasoning.trim() || "(empty)", "");
          else finish(assistant, "");
        } else if (!assistant) finish(reasoning.trim() ? "" : "(empty)", reasoning.trim());
        else finish(assistant, reasoning);
      } else {
        const body = (await res.json()) as {
          choices?: { message?: { content?: string; reasoning_content?: string } }[];
          usage?: { completion_tokens?: number };
        };
        usageTokens = Number(body.usage?.completion_tokens || 0);
        if (!firstAt) firstAt = performance.now();
        const msg = body.choices?.[0]?.message;
        const reply = (msg?.content || "").trim();
        const thinkText = (msg?.reasoning_content || "").trim();
        if (foldReason) {
          const merged =
            reply && thinkText && (reply === thinkText || reply.startsWith(thinkText) || thinkText.startsWith(reply))
              ? reply.length >= thinkText.length
                ? reply
                : thinkText
              : `${thinkText}${reply}`.trim();
          finish(merged || "(empty)", "");
        } else {
          finish(reply || (thinkText ? "" : "(empty)"), thinkText);
        }
      }
    } catch (err) {
      if (ac.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      if (abortRef.current === ac) abortRef.current = null;
      setBusy(false);
      setProgress(null);
    }
  }

  if (!live && turns.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-start justify-center">
        <p className="max-w-md text-sm text-muted-foreground">
          {served.length
            ? `${served.length} model${served.length === 1 ? "" : "s"} already loaded. Serve this one to hot-load it beside them on the same /v1.`
            : "Serve a model first. Each Serve hot-loads into the same OpenAI /v1 — you can keep several loaded at once."}
        </p>
      </div>
    );
  }

  const prefill = progress?.phase === "prefill" ? progress.prompt : null;
  const prefillPct = progress?.phase === "prefill" ? Math.round((progress.progress ?? 0) * 100) : null;

  return (
    <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-1 flex-col">
      <div className="mb-2 flex items-center justify-end">
        {turns.length ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => void reset()}
            aria-label="Clear conversation"
          >
            <Trash2 className="size-3.5" />
            Clear
          </Button>
        ) : null}
      </div>
      <div
        ref={scrollerRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1"
      >
        {turns.length === 0 ? (
          <p className="pt-6 text-sm text-muted-foreground">
            Streaming POST /v1/chat/completions on the already-loaded engine. {served.length} loaded on this origin.
          </p>
        ) : (
          turns.map((turn, i) => (
            <article
              key={`${turn.role}-${i}`}
              className={cn(
                "group relative rounded-2xl px-4 py-3 pr-10 text-sm",
                turn.role === "user"
                  ? "ml-auto max-w-[85%] bg-secondary text-foreground"
                  : "mr-auto w-full max-w-full bg-card shadow-[var(--shadow-border)]",
              )}
            >
              <button
                type="button"
                className="absolute top-2 right-2 inline-flex size-7 items-center justify-center rounded-md text-muted-foreground opacity-70 transition-opacity hover:bg-accent hover:text-foreground group-hover:opacity-100"
                aria-label="Delete this message and everything after"
                title="Delete from here"
                onClick={() => void rewind(i)}
              >
                <X className="size-3.5" />
              </button>
              <p className="mb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">
                {turn.role === "assistant" && turn.thinking && !turn.text ? "thinking" : turn.role}
              </p>
              {turn.role === "assistant" && !turn.text && !turn.thinking && prefill ? (
                <PrefillMeter
                  processed={prefill.processed_tokens}
                  total={prefill.total_tokens}
                  pct={prefillPct}
                />
              ) : (
                <>
                  {turn.thinking ? (
                    <p className="mb-2 text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground italic">
                      {turn.thinking}
                    </p>
                  ) : null}
                  {turn.text ? (
                    <div className="leading-relaxed">
                      <Markdown text={turn.text} />
                      {stopping && i === turns.length - 1 && turn.role === "assistant" ? (
                        <span className="ml-0.5 inline-block h-3 w-0.5 animate-pulse bg-foreground align-middle" />
                      ) : null}
                    </div>
                  ) : stopping && i === turns.length - 1 && turn.role === "assistant" ? (
                    <span className="inline-block h-3 w-0.5 animate-pulse bg-foreground align-middle" />
                  ) : null}
                  {turn.role === "assistant" && turn.metrics && !(stopping && i === turns.length - 1) ? (
                    <MetricsLine metrics={turn.metrics} />
                  ) : null}
                </>
              )}
            </article>
          ))
        )}
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        {!live ? (
          <p className="text-xs text-muted-foreground">Serve this model to keep chatting. Transcript stays until Edge exits.</p>
        ) : null}
      </div>
      <form
        className="mt-4 flex items-end gap-2 rounded-2xl border border-border bg-card p-2 shadow-[var(--shadow-border)]"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={`Message · ${model?.name ?? "OpenAI /v1"}`}
          className="min-h-14 border-0 bg-transparent shadow-none focus-visible:ring-0"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!stopping) void send();
            }
          }}
        />
        {stopping ? (
          <Button type="button" size="icon" className="size-11 shrink-0" onClick={() => void halt()} aria-label="Stop">
            <Square />
          </Button>
        ) : (
          <Button type="submit" size="icon" className="size-11 shrink-0" disabled={!live || !input.trim()} aria-label="Send">
            <ArrowUp />
          </Button>
        )}
      </form>
    </div>
  );
}

function PrefillMeter({
  processed,
  total,
  pct,
}: {
  processed: number;
  total: number | null;
  pct: number | null;
}) {
  const width = pct ?? 8;
  return (
    <div>
      <p className="text-xs text-muted-foreground">
        Processing prompt
        {total ? ` ${processed.toLocaleString("en-US")} / ${total.toLocaleString("en-US")}` : ` ${processed.toLocaleString("en-US")} tokens`}
        {pct != null ? ` · ${pct}%` : ""}
      </p>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-secondary">
        <div className="h-full rounded-full bg-ok transition-[width] duration-150" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function MetricsLine({ metrics }: { metrics: PlaygroundMetrics }) {
  const bits = [
    { tip: "Time to first token", text: `TTFT ${metrics.ttft.toFixed(2)}s` },
    { tip: "Generation time after first token", text: `Gen ${metrics.gen.toFixed(2)}s` },
    { tip: "Completion tokens", text: `${metrics.tokens} tok` },
    { tip: "Tokens per second", text: `${metrics.tps.toFixed(1)} T/s` },
    { tip: "Model", text: metrics.model },
  ];
  return (
    <p className="mt-3 flex flex-wrap items-center gap-x-1.5 font-mono text-[10px] tabular-nums text-muted-foreground">
      {bits.map((bit, i) => (
        <span key={bit.tip}>
          {i ? <span aria-hidden> · </span> : null}
          <span title={bit.tip}>{bit.text}</span>
        </span>
      ))}
    </p>
  );
}

function estimateTokens(text: string): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(words ? Math.round(words * 1.3) : 0, 0);
}