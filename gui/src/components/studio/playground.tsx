import { useState } from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { modelIsLive } from "@/lib/edge-api";
import { useStudio } from "@/lib/studio-store";
import { cn } from "@/lib/utils";

type ChatTurn = { role: "user" | "assistant"; text: string };

export function Playground() {
  const model = useStudio((s) => s.selected());
  const served = useStudio((s) => s.served);
  const live = modelIsLive(served, model);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function send() {
    const text = input.trim();
    if (!text || !model || !live || busy) return;
    setInput("");
    setError(null);
    const next = [...turns, { role: "user" as const, text }];
    setTurns(next);
    setBusy(true);
    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model.repo,
          messages: next.map((t) => ({ role: t.role, content: t.text })),
          max_tokens: 256,
          stream: false,
        }),
      });
      const body = (await res.json()) as {
        error?: { message?: string };
        choices?: { message?: { content?: string } }[];
      };
      if (!res.ok) {
        throw new Error(body.error?.message || `HTTP ${res.status}`);
      }
      const reply = body.choices?.[0]?.message?.content ?? "(empty)";
      setTurns([...next, { role: "assistant", text: reply }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  if (!live) {
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

  return (
    <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-1 flex-col">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
        {turns.length === 0 ? (
          <p className="pt-6 text-sm text-muted-foreground">
            POST /v1/chat/completions with model {model?.repo}. {served.length} loaded on this origin.
          </p>
        ) : (
          turns.map((turn, i) => (
            <article
              key={`${turn.role}-${i}`}
              className={cn(
                "max-w-[85%] rounded-2xl px-4 py-3 text-sm",
                turn.role === "user"
                  ? "ml-auto bg-secondary text-foreground"
                  : "mr-auto bg-card shadow-[var(--shadow-border)]",
              )}
            >
              <p className="mb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">
                {turn.role}
              </p>
              <p className="leading-relaxed whitespace-pre-wrap">{turn.text}</p>
            </article>
          ))
        )}
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
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
              void send();
            }
          }}
        />
        <Button type="submit" size="icon" className="size-11 shrink-0" disabled={busy || !input.trim()} aria-label="Send">
          <ArrowUp />
        </Button>
      </form>
    </div>
  );
}
