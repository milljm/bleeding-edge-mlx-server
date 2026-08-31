import { type FormEvent, type ReactNode, useMemo, useState } from "react";
import { ChevronRight, Folder, PanelLeft, Plus, RefreshCw, Search, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DIR_PLACEHOLDER } from "@/lib/models";
import { modelIsLive } from "@/lib/edge-api";
import { useStudio } from "@/lib/studio-store";
import { cn } from "@/lib/utils";

export function Sidebar({
  onNavigate,
  onCollapse,
}: {
  onNavigate?: () => void;
  onCollapse?: () => void;
}) {
  const watchDirs = useStudio((s) => s.watchDirs);
  const models = useStudio((s) => s.models);
  const selectedId = useStudio((s) => s.selectedId);
  const served = useStudio((s) => s.served);
  const dirDraft = useStudio((s) => s.dirDraft);
  const modelDraft = useStudio((s) => s.modelDraft);
  const modelEngine = useStudio((s) => s.modelEngine);
  const scanning = useStudio((s) => s.scanning);
  const scanErrors = useStudio((s) => s.scanErrors);
  const addWatchDir = useStudio((s) => s.addWatchDir);
  const removeWatchDir = useStudio((s) => s.removeWatchDir);
  const setDirDraft = useStudio((s) => s.setDirDraft);
  const addModel = useStudio((s) => s.addModel);
  const setModelDraft = useStudio((s) => s.setModelDraft);
  const setModelEngine = useStudio((s) => s.setModelEngine);
  const selectModel = useStudio((s) => s.selectModel);
  const scanWatchDirs = useStudio((s) => s.scanWatchDirs);
  const [query, setQuery] = useState("");

  const loadedCount = served.length;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return models;
    return models.filter(
      (m) =>
        m.name.toLowerCase().includes(q) ||
        m.repo.toLowerCase().includes(q) ||
        m.engine.includes(q),
    );
  }, [models, query]);

  function submitDir(e: FormEvent) {
    e.preventDefault();
    if (!dirDraft.trim()) return;
    void addWatchDir(dirDraft);
  }

  function submitModel(e: FormEvent) {
    e.preventDefault();
    addModel(modelDraft, modelEngine);
  }

  const emptyHint = !watchDirs.length
    ? "Add a folder to watch. Edge lists MLX models it finds (config.json + weights). Hugging Face ids can be added below without a folder."
    : scanning
      ? "Scanning folders…"
      : "No models in these folders. Point at a directory that contains MLX checkpoints, or add a Hugging Face id below.";

  return (
    <aside className="flex h-full min-h-0 flex-col bg-card paper">
      <div className="flex items-center gap-2 px-4 pb-3 pt-4">
        <div className="min-w-0 flex-1">
          <p className="font-display text-2xl italic leading-none tracking-tight">Edge</p>
          <p className="mt-1 text-xs text-muted-foreground">Front-end for mlx-edge</p>
        </div>
        {onCollapse ? (
          <Button type="button" variant="ghost" size="icon-sm" aria-label="Collapse sidebar" onClick={onCollapse}>
            <PanelLeft />
          </Button>
        ) : null}
      </div>

      <Section title="Watch" count={watchDirs.length} defaultOpen>
        <form onSubmit={submitDir} className="flex gap-2 px-4">
          <Input
            value={dirDraft}
            onChange={(e) => setDirDraft(e.target.value)}
            placeholder={DIR_PLACEHOLDER}
            className="h-9 font-mono text-xs"
            aria-label="Model directory"
          />
          <Button
            type="submit"
            variant="secondary"
            size="icon-sm"
            className="size-9"
            aria-label="Add directory"
            disabled={!dirDraft.trim()}
          >
            <Plus />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="size-9"
            aria-label="Rescan folders"
            disabled={scanning || watchDirs.length === 0}
            onClick={() => void scanWatchDirs()}
          >
            <RefreshCw className={cn(scanning && "animate-spin")} />
          </Button>
        </form>
        <ul className="mt-2 space-y-1 px-3 pb-3">
          {watchDirs.map((dir) => (
            <li
              key={dir}
              className="flex items-center gap-2 rounded-md px-1 py-1.5 text-xs text-muted-foreground"
            >
              <Folder className="size-3.5 shrink-0" />
              <span className="min-w-0 flex-1 truncate font-mono">{dir}</span>
              <button
                type="button"
                className="rounded p-1 hover:bg-accent hover:text-foreground"
                aria-label={`Remove ${dir}`}
                onClick={() => void removeWatchDir(dir)}
              >
                <Trash2 className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
        {scanErrors.length ? (
          <p className="px-4 pb-3 text-xs text-destructive">
            {scanErrors.map((err) => `${err.dir}: ${err.message}`).join(" · ")}
          </p>
        ) : null}
      </Section>

      <Section title="Models" count={models.length} defaultOpen className="min-h-0 flex-1 border-t border-border">
        <div className="px-4 pb-2">
          <div className="relative">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter models"
              className="h-9 pl-8 text-xs"
              aria-label="Filter models"
            />
          </div>
          {loadedCount > 0 ? (
            <p className="mt-2 text-xs text-ok">{loadedCount} hot-loaded on /v1</p>
          ) : null}
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {filtered.length === 0 ? (
            <p className="px-2 py-6 text-sm text-muted-foreground">{emptyHint}</p>
          ) : (
            <ul className="space-y-1">
              {filtered.map((model) => {
                const active = model.id === selectedId;
                const live = modelIsLive(served, model);
                return (
                  <li key={model.id}>
                    <button
                      type="button"
                      onClick={() => {
                        selectModel(model.id);
                        onNavigate?.();
                      }}
                      className={cn(
                        "flex w-full items-start gap-2 rounded-lg px-2 py-2 text-left transition-colors duration-150",
                        active
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-accent",
                      )}
                    >
                      <span
                        className={cn(
                          "mt-1.5 size-1.5 shrink-0 rounded-full",
                          live ? "bg-ok" : active ? "bg-primary-foreground/40" : "bg-border",
                        )}
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium">{model.name}</span>
                        <span className="mt-0.5 flex flex-wrap items-center gap-1.5">
                          <Badge
                            variant={model.engine === "vlm" ? "warn" : "default"}
                            className={active ? "border-primary-foreground/20 bg-primary-foreground/10 text-primary-foreground" : undefined}
                          >
                            {model.engine === "vlm" ? "vlm" : "lm"}
                          </Badge>
                          <span className={cn("text-xs", active ? "text-primary-foreground/70" : "text-muted-foreground")}>
                            {live ? "loaded · " : ""}
                            {model.quant} · {model.size}
                          </span>
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        <form onSubmit={submitModel} className="space-y-2 border-t border-border px-4 py-3">
          <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Add model</p>
          <Input
            value={modelDraft}
            onChange={(e) => setModelDraft(e.target.value)}
            placeholder="mlx-community/Qwen3-8B-4bit"
            className="h-9 font-mono text-xs"
            aria-label="Model repo"
          />
          <div className="flex gap-2">
            <select
              value={modelEngine}
              onChange={(e) => setModelEngine(e.target.value === "vlm" ? "vlm" : "lm")}
              className="h-9 flex-1 rounded-md border border-input bg-card px-2 font-mono text-xs"
              aria-label="Engine"
            >
              <option value="lm">mlx-lm</option>
              <option value="vlm">mlx-vlm</option>
            </select>
            <Button type="submit" variant="secondary" size="sm" disabled={!modelDraft.trim()}>
              <Plus />
              Add
            </Button>
          </div>
        </form>
      </Section>
    </aside>
  );
}

function Section({
  title,
  count,
  defaultOpen = true,
  className,
  children,
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={cn("flex flex-col", className)}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 px-4 py-2 text-left text-xs font-medium tracking-wide text-muted-foreground uppercase hover:text-foreground"
      >
        <ChevronRight
          className={cn(
            "size-3.5 transition-transform duration-150",
            open && "rotate-90",
          )}
        />
        <span>{title}</span>
        {count != null ? <span className="text-muted-foreground/80">{count}</span> : null}
      </button>
      {open ? <div className="flex min-h-0 flex-1 flex-col">{children}</div> : null}
    </section>
  );
}