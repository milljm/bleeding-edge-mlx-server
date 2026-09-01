import { useState } from "react";
import { RefreshCw, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { engineLabel } from "@/lib/command";
import { fetchTemplate, modelIsLive } from "@/lib/edge-api";
import { flagsDirty, flagsFor, type FlagDef, type FlagGroup } from "@/lib/flags";
import { formatContext, loadTarget } from "@/lib/models";
import { useStudio } from "@/lib/studio-store";

const GROUP_LABEL: Record<FlagGroup, string> = {
  server: "Server",
  sampling: "Sampling",
  thinking: "Thinking",
  template: "Chat template",
};

export function FlagPanel() {
  const model = useStudio((s) => s.selected());
  const flags = useStudio((s) => s.flags);
  const served = useStudio((s) => s.served);
  const setFlag = useStudio((s) => s.setFlag);
  const resetFlags = useStudio((s) => s.resetFlags);
  const reloadServe = useStudio((s) => s.reloadServe);
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);

  if (!model) {
    return <p className="text-sm text-muted-foreground">Pick a model in the sidebar, or add a folder to watch.</p>;
  }

  const visible = flagsFor(model.engine, false);
  const extra = flagsFor(model.engine, true);
  const groups: FlagGroup[] = ["server", "sampling", "thinking"];
  const live = modelIsLive(served, model);
  const loaded = served.find((row) => modelIsLive([row], model));
  const dirty = Boolean(live && flagsDirty(model.engine, flags, loaded?.flags));

  return (
    <div className="mx-auto w-full max-w-3xl space-y-8">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-medium tracking-tight">Engine switches</h2>
          <p className="mt-1 max-w-lg text-sm text-muted-foreground">
            Settings for {model.name} only — they stick when you close Edge.
            {formatContext(model.context) ? ` Context window ${formatContext(model.context)}.` : ""}{" "}
            These map 1:1 onto {engineLabel(model.engine)} flags.
            {model.engine === "embed"
              ? " Embedding models answer POST /v1/embeddings — they do not chat."
              : live
                ? dirty
                  ? " Flags changed on a running model — Reload to apply them."
                  : " Change a flag and Reload to apply it without unloading the others."
                : " Serve hot-loads this model beside any that are already up."}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {dirty ? (
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await reloadServe();
                } finally {
                  setBusy(false);
                }
              }}
            >
              <RefreshCw className={busy ? "animate-spin" : undefined} />
              Reload model
            </Button>
          ) : null}
          <Button type="button" variant="ghost" size="sm" onClick={resetFlags}>
            <RotateCcw />
            Reset
          </Button>
        </div>
      </div>
      {model.engine === "embed" ? (
        <p className="rounded-2xl bg-card px-4 py-3 text-sm text-muted-foreground shadow-[var(--shadow-border)]">
          This is an embedding model. Serve it, then POST <span className="font-mono text-foreground">/v1/embeddings</span>.
          Keep a chat model loaded too — RAG does not unload it.
        </p>
      ) : null}
      {model.engine === "lm" ? <TemplateCard /> : null}
      {groups.map((group) => {
        const defs = visible.filter((d) => (d.group ?? "server") === group);
        if (defs.length === 0) return null;
        return (
          <section key={group} className="space-y-4">
            <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
              {GROUP_LABEL[group]}
            </h3>
            <FlagGrid defs={defs} values={flags} onChange={setFlag} />
          </section>
        );
      })}
      <div>
        <button
          type="button"
          className="text-xs font-medium tracking-wide text-muted-foreground uppercase hover:text-foreground"
          onClick={() => setAdvanced((v) => !v)}
        >
          {advanced ? "Hide advanced" : "Show advanced"}
        </button>
        {advanced ? (
          <div className="mt-4">
            <FlagGrid defs={extra} values={flags} onChange={setFlag} />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function TemplateCard() {
  const model = useStudio((s) => s.selected());
  const flags = useStudio((s) => s.flags);
  const setFlag = useStudio((s) => s.setFlag);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  if (!model) return null;
  const bundled = Boolean(model.hasChatTemplate);
  const override = String(flags.chatTemplate || "").trim();

  async function pull() {
    if (!model) return;
    setBusy(true);
    setNote(null);
    try {
      const info = await fetchTemplate({ model: loadTarget(model), repo: model.repo });
      if (!info.chat_template) {
        setNote("No template on the Hub for this repo. Paste Jinja below, then Reload.");
        return;
      }
      setFlag("chatTemplate", info.chat_template);
      setNote(info.bundled ? "Using the checkpoint template." : `Loaded from ${info.source || "Hugging Face"}. Reload to apply.`);
    } catch (err) {
      setNote(err instanceof Error ? err.message : "Fetch failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-3 rounded-2xl bg-card px-4 py-4 shadow-[var(--shadow-border)]">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Chat template</h3>
          <p className="mt-1 max-w-lg text-sm text-muted-foreground">
            {bundled
              ? "This checkpoint already has a tokenizer chat_template. mlx-lm will apply it."
              : "No chat_template in this checkpoint. MiniMax / gpt-oss will leak <|channel|> tokens unless you apply one. Edge also strips those tokens from /v1 content."}
          </p>
        </div>
        <Button type="button" variant="secondary" size="sm" disabled={busy} onClick={() => void pull()}>
          {busy ? "Pulling…" : "Pull from Hugging Face"}
        </Button>
      </div>
      {note ? <p className="text-xs text-ok">{note}</p> : null}
      <div className="space-y-2">
        <Label htmlFor="chatTemplate">Jinja override</Label>
        <Textarea
          id="chatTemplate"
          value={String(flags.chatTemplate || "")}
          onChange={(e) => setFlag("chatTemplate", e.target.value)}
          placeholder="Leave empty to use the checkpoint (or the template Edge injects on Serve when missing)."
          className="min-h-28 font-mono text-xs"
        />
      </div>
      <div className="flex items-center justify-between gap-3">
        <div>
          <Label htmlFor="useDefaultChatTemplate" className="text-foreground">
            Default chat template
          </Label>
          <p className="mt-1 text-xs text-muted-foreground">Force the tokenizer default instead of the override.</p>
        </div>
        <Switch
          id="useDefaultChatTemplate"
          checked={Boolean(flags.useDefaultChatTemplate)}
          onCheckedChange={(checked) => setFlag("useDefaultChatTemplate", checked)}
        />
      </div>
      {override ? (
        <p className="text-xs text-muted-foreground">{override.length.toLocaleString("en-US")} characters · Reload to apply</p>
      ) : null}
    </section>
  );
}

function FlagGrid({
  defs,
  values,
  onChange,
}: {
  defs: FlagDef[];
  values: Record<string, string | number | boolean>;
  onChange: (key: string, value: string | number | boolean) => void;
}) {
  return (
    <div className="grid gap-5 sm:grid-cols-2">
      {defs.map((def) => (
        <FlagField key={def.key} def={def} value={values[def.key] ?? def.default} onChange={onChange} />
      ))}
    </div>
  );
}

function FlagField({
  def,
  value,
  onChange,
}: {
  def: FlagDef;
  value: string | number | boolean;
  onChange: (key: string, value: string | number | boolean) => void;
}) {
  if (def.type === "bool") {
    return (
      <div className="flex items-center justify-between gap-3 rounded-xl border border-border bg-card px-3 py-3 sm:col-span-2">
        <div>
          <Label htmlFor={def.key} className="text-foreground">
            {def.label}
          </Label>
          <p className="mt-1 text-xs text-muted-foreground">{def.help}</p>
        </div>
        <Switch
          id={def.key}
          checked={Boolean(value)}
          onCheckedChange={(checked) => onChange(def.key, checked)}
        />
      </div>
    );
  }

  if (def.type === "select") {
    return (
      <div className="space-y-2">
        <Label htmlFor={def.key}>{def.label}</Label>
        <select
          id={def.key}
          className="flex h-11 w-full rounded-md border border-input bg-card px-3 text-sm"
          value={String(value)}
          onChange={(e) => onChange(def.key, e.target.value)}
        >
          {def.options?.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-muted-foreground">{def.help}</p>
      </div>
    );
  }

  if (def.type === "number" && def.min != null && def.max != null && def.step != null && def.max <= 200) {
    const num = Number(value);
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <Label htmlFor={def.key}>{def.label}</Label>
          <span className="font-mono text-xs text-muted-foreground tabular-nums">{num}</span>
        </div>
        <Slider
          id={def.key}
          min={def.min}
          max={def.max}
          step={def.step}
          value={[num]}
          onValueChange={([next]) => onChange(def.key, next ?? def.default)}
        />
        <p className="text-xs text-muted-foreground">{def.help}</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <Label htmlFor={def.key}>{def.label}</Label>
      <Input
        id={def.key}
        type={def.type === "number" ? "number" : "text"}
        min={def.min}
        max={def.max}
        step={def.step}
        value={String(value)}
        onChange={(e) =>
          onChange(def.key, def.type === "number" ? Number(e.target.value) : e.target.value)
        }
        className="h-11 font-mono text-sm"
      />
      <p className="text-xs text-muted-foreground">{def.help}</p>
    </div>
  );
}
