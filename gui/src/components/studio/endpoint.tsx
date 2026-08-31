import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { guiCommand, loadCommand } from "@/lib/command";
import { useStudio } from "@/lib/studio-store";

export function EndpointPanel() {
  const served = useStudio((s) => s.served);
  const model = useStudio((s) => s.selected());
  const flags = useStudio((s) => s.flags);
  const gateway = useStudio((s) => s.gateway);

  if (!model) return null;

  const base = gateway.url;

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6">
      <div>
        <h2 className="text-lg font-medium tracking-tight">OpenAI endpoint</h2>
        <p className="mt-1 max-w-xl text-sm text-muted-foreground">
          Point a chat client at <span className="font-mono font-medium text-foreground">{base}</span>. One
          gateway, many hot-loaded models — pass <span className="font-mono">model</span> to pick which
          engine answers. Bind with <span className="font-mono">edge-gui --host 0.0.0.0</span> for remote
          clients.
        </p>
      </div>
      {served.length ? (
        <ul className="space-y-1 text-sm text-ok">
          {served.map((m) => (
            <li key={m.repo}>
              loaded {m.repo} · {m.engine === "vlm" ? "mlx-vlm" : "mlx-lm"}
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
      <CopyBlock
        label="curl chat"
        code={`curl ${base}/chat/completions \\\n  -H 'content-type: application/json' \\\n  -d '{"model":"${model.repo}","messages":[{"role":"user","content":"hello"}]}'`}
      />
      <CopyBlock label="list models" code={`curl ${base}/models`} />
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
