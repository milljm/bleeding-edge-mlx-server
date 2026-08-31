import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { guiCommand, loadCommand } from "@/lib/command";
import { publicName } from "@/lib/models";
import { useStudio } from "@/lib/studio-store";

export function EndpointPanel() {
  const served = useStudio((s) => s.served);
  const model = useStudio((s) => s.selected());
  const flags = useStudio((s) => s.flags);
  const gateway = useStudio((s) => s.gateway);

  if (!model) return null;

  const base = gateway.url;
  const id = publicName(model);
  const embed = model.engine === "embed";

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6">
      <div>
        <h2 className="text-lg font-medium tracking-tight">OpenAI endpoint</h2>
        <p className="mt-1 max-w-xl text-sm text-muted-foreground">
          Point a client at <span className="font-mono font-medium text-foreground">{base}</span>. One
          gateway, many hot-loaded models — pass the basename as <span className="font-mono">model</span>
          (not the full disk path). Chat uses <span className="font-mono">stream: true</span>. Embeddings
          use <span className="font-mono">POST /v1/embeddings</span> on a separate loaded process, so RAG
          does not stall a chat model that is already up. Prompt-processing progress is a float{" "}
          <span className="font-mono">0.0–1.0</span> on <span className="font-mono">GET /v1/progress</span>{" "}
          (and SSE <span className="font-mono">/v1/progress/stream</span>). Edge strips Harmony{" "}
          <span className="font-mono">{"<|channel|>"}</span> tokens from chat content. Bind with{" "}
          <span className="font-mono">edge-gui --host 0.0.0.0</span> for remote clients.
        </p>
      </div>
      {served.length ? (
        <ul className="space-y-1 text-sm text-ok">
          {served.map((m) => (
            <li key={m.repo}>
              loaded {m.repo} · {m.engine === "vlm" ? "mlx-vlm" : m.engine === "embed" ? "mlx-embed" : "mlx-lm"}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted-foreground">
          Nothing loaded. Serve hot-loads into mlx-edge without replacing models that are already up.
        </p>
      )}
      <CopyBlock label="start GUI + gateway" code={guiCommand(gateway)} />
      <CopyBlock label="hot-load this model" code={loadCommand(model, flags)} />
      {embed ? (
        <CopyBlock
          label="curl embeddings"
          code={`curl ${base}/embeddings \\\n  -H 'content-type: application/json' \\\n  -d '{"model":"${id}","input":"what is the capital of France?"}'`}
        />
      ) : (
        <CopyBlock
          label="curl chat (stream)"
          code={`curl -N ${base}/chat/completions \\\n  -H 'content-type: application/json' \\\n  -d '{"model":"${id}","messages":[{"role":"user","content":"hello"}],"stream":true}'`}
        />
      )}
      <CopyBlock label="list models" code={`curl ${base}/models`} />
      <CopyBlock
        label="processing progress (0.0–1.0)"
        code={`# snapshot — models[].progress and top-level progress are floats 0.0–1.0
curl ${base.replace(/\/v1$/, "")}/v1/progress
curl ${base.replace(/\/v1$/, "")}/v1/progress?model=${id}

# live SSE (EventSource in a browser, or curl -N)
curl -N ${base.replace(/\/v1$/, "")}/v1/progress/stream`}
      />
      <CopyBlock
        label="progress from another app"
        code={`const es = new EventSource("${base.replace(/\/v1$/, "")}/v1/progress/stream");
es.onmessage = (ev) => {
  const snap = JSON.parse(ev.data);
  // snap.progress → 0.0 idle / unknown, 1.0 prefill done (decode)
  const row = snap.models.find((m) => m.id === "${id}") || snap.models[0];
  const p = row?.progress ?? snap.progress; // 0.0 .. 1.0
};`}
      />
      <CopyBlock
        label="engine logs (SSE)"
        code={`curl -N ${base.replace(/\/v1$/, "")}/v1/logs/stream
const logs = new EventSource("${base.replace(/\/v1$/, "")}/v1/logs/stream");
logs.onmessage = (ev) => console.log(JSON.parse(ev.data).lines);`}
      />
    </div>
  );
}

function CopyBlock({ label, code }: { label: string; code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="overflow-hidden rounded-2xl bg-card shadow-[var(--shadow-border)]">
      <div className="flex items-center justify-between gap-2 px-4 py-2">
        <p className="font-mono text-xs tracking-wide text-muted-foreground uppercase">{label}</p>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Copy"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(code);
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1200);
            } catch {
              /* ignore */
            }
          }}
        >
          {copied ? <Check /> : <Copy />}
        </Button>
      </div>
      <pre className="overflow-x-auto px-4 pb-4 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap">
        {code}
      </pre>
    </div>
  );
}