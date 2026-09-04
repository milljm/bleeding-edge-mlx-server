import { type CSSProperties, type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ChevronRight, CircleStop, Folder, PanelLeft, Pause, Play, Plus, RefreshCw, Search, Square, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatContext, DIR_PLACEHOLDER, HF_HUB_WATCH, SUGGESTED_WATCH, loadTarget, modelOrigin, originLabel, sortLoadedFirst, type ModelRec } from "@/lib/models";
import {
  getHubProgress,
  getHubStatus,
  modelGeneration,
  modelIsBusy,
  modelIsLive,
  modelIsPrefill,
  modelLoadProgress,
  postHubCancel,
  postHubDownload,
  postHubPause,
  postHubResume,
  postHubSearch,
  postStop,
  type HubProgress,
  type HubQuant,
  type ProgressSnapshot,
} from "@/lib/edge-api";
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
  const scanning = useStudio((s) => s.scanning);
  const scanErrors = useStudio((s) => s.scanErrors);
  const loadingIds = useStudio((s) => s.loadingIds);
  const pinKeys = useStudio((s) => s.pinKeys);
  const failed = useStudio((s) => s.failed);
  const progress = useStudio((s) => s.progress);
  const addWatchDir = useStudio((s) => s.addWatchDir);
  const removeWatchDir = useStudio((s) => s.removeWatchDir);
  const setDirDraft = useStudio((s) => s.setDirDraft);
  const selectModel = useStudio((s) => s.selectModel);
  const startServe = useStudio((s) => s.startServe);
  const stopServe = useStudio((s) => s.stopServe);
  const scanWatchDirs = useStudio((s) => s.scanWatchDirs);
  const [query, setQuery] = useState("");

  const loadedCount = served.length;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = q
      ? models.filter(
          (m) =>
            m.name.toLowerCase().includes(q) ||
            m.repo.toLowerCase().includes(q) ||
            m.engine.includes(q),
        )
      : models;
    return sortLoadedFirst(list, (m) => modelIsLive(served, m) || loadingIds.includes(m.id) || pinKeys.includes(m.id));
  }, [models, query, served, loadingIds, pinKeys]);

  function submitDir(e: FormEvent) {
    e.preventDefault();
    if (!dirDraft.trim()) return;
    void addWatchDir(dirDraft);
  }

  const emptyHint = !watchDirs.length
    ? "Add a folder to watch, or Hugging Face / LM Studio / Ollama. MLX-Edge lists checkpoints it finds (config.json + weights)."
    : scanning
      ? "Scanning folders…"
      : "No models in these folders. Point at a directory that contains MLX checkpoints.";

  return (
    <aside className="flex h-full min-h-0 flex-col bg-card paper">
      <div className="flex items-center gap-2 px-4 pb-3 pt-4">
        <div className="min-w-0 flex-1">
          <p className="font-display text-2xl italic leading-none tracking-tight">MLX-Edge</p>
          <p className="mt-1 text-xs text-muted-foreground">A front-end for mlx-lm, mlx-vlm</p>
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
        {SUGGESTED_WATCH.some((s) => !watchDirs.includes(s.path)) ? (
          <div className="mt-2 flex flex-wrap gap-1.5 px-4">
            {SUGGESTED_WATCH.filter((s) => !watchDirs.includes(s.path)).map((s) => (
              <button
                key={s.path}
                type="button"
                className="rounded-full border border-border bg-background px-2.5 py-1 text-[11px] text-muted-foreground hover:border-foreground/30 hover:text-foreground"
                onClick={() => void addWatchDir(s.path)}
              >
                {s.label}
              </button>
            ))}
          </div>
        ) : null}
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

      <HubPanel />

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
        <div className="min-h-0 flex-1 overflow-y-auto overflow-x-visible px-2 pb-3 pt-2">
          {filtered.length === 0 ? (
            <p className="px-2 py-6 text-sm text-muted-foreground">{emptyHint}</p>
          ) : (
            <ul className="space-y-1">
              {filtered.map((model) => (
                <ModelCard
                  key={model.id}
                  model={model}
                  active={model.id === selectedId}
                  live={modelIsLive(served, model)}
                  error={failed[model.id]}
                  loading={loadingIds.includes(model.id)}
                  busy={modelIsBusy(progress, model)}
                  progress={progress}
                  onSelect={() => {
                    selectModel(model.id);
                    onNavigate?.();
                  }}
                  onToggle={async () => {
                    try {
                      if (modelIsBusy(progress, model)) await postStop(loadTarget(model));
                      else if (modelIsLive(served, model)) await stopServe(model.id);
                      else await startServe(model.id);
                    } catch (err) {
                      toast.error(err instanceof Error ? err.message : "Failed");
                    }
                  }}
                />
              ))}
            </ul>
          )}
        </div>
      </Section>
    </aside>
  );
}

function useLoadFill(active: boolean, reported: number | null) {
  const [eased, setEased] = useState(0);
  const started = useRef<number | null>(null);

  useEffect(() => {
    if (!active) {
      started.current = null;
      setEased(0);
      return;
    }
    started.current = started.current ?? performance.now();
    const reduce =
      typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setEased(0.55);
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      const elapsed = (now - (started.current ?? now)) / 1000;
      setEased(0.92 * (1 - Math.exp(-elapsed / 14)));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  if (!active) return 0;
  return Math.min(0.96, Math.max(reported ?? 0, eased));
}

function useTokenBubble(liveTokens: number, generating: boolean) {
  const [display, setDisplay] = useState(0);
  const [phase, setPhase] = useState<"hidden" | "live" | "hold" | "leave">("hidden");
  const last = useRef(0);
  const timers = useRef<{ hold: number | null; leave: number | null }>({ hold: null, leave: null });

  function clearTimers() {
    if (timers.current.hold != null) window.clearTimeout(timers.current.hold);
    if (timers.current.leave != null) window.clearTimeout(timers.current.leave);
    timers.current = { hold: null, leave: null };
  }

  useEffect(() => {
    if (generating && liveTokens > 0) {
      last.current = liveTokens;
      setDisplay(liveTokens);
      setPhase("live");
      clearTimers();
    }
  }, [generating, liveTokens]);

  useEffect(() => {
    if (generating) return;
    if (last.current <= 0) return;
    setDisplay(last.current);
    setPhase("hold");
    clearTimers();
    timers.current.hold = window.setTimeout(() => {
      setPhase("leave");
      timers.current.leave = window.setTimeout(() => {
        setPhase("hidden");
        setDisplay(0);
        last.current = 0;
      }, 500);
    }, 2000);
    return clearTimers;
  }, [generating]);

  return { display, phase };
}

function TokenBubble({ tokens, generating }: { tokens: number; generating: boolean }) {
  const { display, phase } = useTokenBubble(tokens, generating);
  if (phase === "hidden" || display <= 0) return null;
  const label = `${display.toLocaleString("en-US")} tokens`;
  return (
    <span
      className={cn("tok-bubble", phase === "leave" && "is-leaving")}
      title={label}
      aria-label={`${label} generated`}
    >
      {display.toLocaleString("en-US")}
    </span>
  );
}

function FeatureChips({ features }: { features?: ModelRec["features"] }) {
  if (!features) return null;
  const shown = (["tool", "vision", "reason"] as const).filter((key) => features[key]);
  if (!shown.length) return null;
  return (
    <>
      {shown.map((key) => (
        <Badge key={key} className="px-1.5 py-0 text-[10px] tracking-wide">
          {key}
        </Badge>
      ))}
    </>
  );
}

function ModelCard({
  model,
  active,
  live,
  error,
  loading,
  busy,
  progress,
  onSelect,
  onToggle,
}: {
  model: ModelRec;
  active: boolean;
  live: boolean;
  error?: string;
  loading: boolean;
  busy: boolean;
  progress: ProgressSnapshot | null;
  onSelect: () => void;
  onToggle: () => void | Promise<void>;
}) {
  const reported = modelLoadProgress(progress, model);
  const fill = useLoadFill(loading, reported);
  const tinted = live && !loading;
  const context = formatContext(model.context);
  const gen = modelGeneration(progress, model);
  const prefill = modelIsPrefill(progress, model);
  const status = prefill
    ? "processing"
    : busy
      ? "generating"
      : loading
        ? "loading"
        : live
          ? "loaded"
          : error
            ? "failed"
            : undefined;
  const controlLabel = busy ? `Stop ${model.name}` : live ? `Unload ${model.name}` : `Serve ${model.name}`;

  return (
    <li className="relative overflow-visible">
      <div
        className={cn(
          "relative flex w-full overflow-visible rounded-lg ring-inset transition-colors duration-150",
          tinted && "bg-live/20 text-foreground",
          error && !tinted && !loading && "bg-destructive/40 text-foreground ring-1 ring-destructive/60",
          loading && "text-foreground",
          active && "ring-2 ring-primary",
        )}
      >
        {loading ? (
          <span
            className="load-fill"
            style={{ "--load-pct": String(fill) } as CSSProperties}
            role="progressbar"
            aria-label={`Loading ${model.name}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(fill * 100)}
          />
        ) : null}
        <TokenBubble tokens={gen.tokens} generating={gen.generating} />
        <button
          type="button"
          onClick={onSelect}
          aria-busy={loading || busy || undefined}
          className={cn(
            "relative flex min-w-0 flex-1 items-start gap-2 rounded-l-lg px-2 py-2 text-left",
            tinted && "hover:bg-live/10",
            error && !tinted && !loading && "hover:bg-destructive/20",
            !tinted && !error && !loading && "hover:bg-accent",
          )}
        >
          <span
            className={cn(
              "relative mt-1.5 size-1.5 shrink-0 rounded-full",
              busy
                ? "busy-dot"
                : loading
                  ? "animate-pulse bg-live"
                  : live
                    ? "bg-live"
                    : error
                      ? "bg-destructive"
                      : "bg-border",
            )}
            title={status}
          />
          <span className="relative min-w-0 flex-1">
            <span className="flex min-w-0 items-center justify-between gap-1.5">
              <span className="min-w-0 flex-1 truncate text-sm font-medium">{model.name}</span>
              <Badge className="shrink-0 px-1.5 py-0 text-[10px] tracking-wide">
                {originLabel(modelOrigin(model))}
              </Badge>
            </span>
            <span className="mt-0.5 flex flex-wrap items-center gap-1.5">
              <Badge variant={model.engine === "vlm" ? "warn" : model.engine === "lm" ? "default" : "accent"}>
                {model.engine}
              </Badge>
              <FeatureChips features={model.features} />
              <span className="text-xs text-muted-foreground">
                {status ? `${status} · ` : ""}
                {model.quant} · {model.size}
                {context ? ` · ${context}` : ""}
              </span>
            </span>
          </span>
        </button>
        <button
          type="button"
          disabled={loading}
          aria-label={controlLabel}
          title={controlLabel}
          className={cn(
            "relative z-[1] flex w-9 shrink-0 items-center justify-center self-stretch rounded-r-lg border-l border-border/50",
            "transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
            loading && "pointer-events-none opacity-40",
            busy || live
              ? "bg-destructive text-primary-foreground hover:opacity-90"
              : "bg-secondary text-foreground hover:bg-accent",
          )}
          onClick={() => void onToggle()}
        >
          {busy ? (
            <CircleStop className="size-3.5" />
          ) : live ? (
            <Square className="size-3.5" />
          ) : (
            <Play className={cn("size-3.5", loading && "animate-pulse")} />
          )}
        </button>
      </div>
    </li>
  );
}

function HubPanel() {
  const watchDirs = useStudio((s) => s.watchDirs);
  const addWatchDir = useStudio((s) => s.addWatchDir);
  const scanWatchDirs = useStudio((s) => s.scanWatchDirs);
  const [draft, setDraft] = useState("");
  const [token, setToken] = useState(false);
  const [help, setHelp] = useState(
    "Create a Hugging Face account, copy a token from huggingface.co/settings/tokens, then launch edge-gui with HF_TOKEN set.",
  );
  const [results, setResults] = useState<HubQuant[]>([]);
  const [picked, setPicked] = useState("");
  const [busy, setBusy] = useState<"search" | null>(null);
  const [jobs, setJobs] = useState<HubProgress[]>([]);
  const seenDone = useRef(new Set<string>());

  const active = jobs.filter((j) => j.phase === "downloading" || j.phase === "paused");
  const pickedBusy = active.some((j) => j.repo === picked);
  const locked = !token;

  useEffect(() => {
    void getHubStatus()
      .then((s) => {
        setToken(s.token);
        if (s.help) setHelp(s.help);
      })
      .catch(() => setToken(false));
  }, []);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const snap = await getHubProgress();
        if (stop) return;
        setJobs(snap.jobs);
        for (const job of snap.jobs) {
          const key = `${job.repo}:${job.phase}`;
          if (job.phase === "done" && !seenDone.current.has(job.repo)) {
            seenDone.current.add(job.repo);
            toast.success(`Downloaded ${job.repo}`);
            if (!watchDirs.includes(HF_HUB_WATCH)) await addWatchDir(HF_HUB_WATCH);
            else await scanWatchDirs();
          } else if (job.phase === "error" && !seenDone.current.has(key)) {
            seenDone.current.add(key);
            toast.error(job.error || "Download failed");
          } else if (job.phase === "cancelled" && !seenDone.current.has(key)) {
            seenDone.current.add(key);
            toast.message("Download cancelled");
          }
        }
      } catch {
        /* keep last snapshot */
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 400);
    return () => {
      stop = true;
      window.clearInterval(id);
    };
  }, [addWatchDir, scanWatchDirs, watchDirs]);

  async function lookup(e?: FormEvent) {
    e?.preventDefault();
    const query = draft.trim();
    if (!query || locked) return;
    setBusy("search");
    try {
      const out = await postHubSearch(query);
      setToken(out.token);
      setResults(out.results);
      const prefer = out.results.find((r) => r.quant === "4-bit") ?? out.results[0];
      setPicked(prefer?.id ?? "");
      if (!out.results.length) toast.error("No MLX quants for that repo");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Hub search failed");
    } finally {
      setBusy(null);
    }
  }

  async function download() {
    if (!picked || locked || pickedBusy) return;
    try {
      seenDone.current.delete(picked);
      const snap = await postHubDownload(picked);
      setJobs((prev) => {
        const rest = prev.filter((j) => j.repo !== snap.repo);
        return [...rest, snap];
      });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Download failed");
    }
  }

  const tokenTip = (
    <span>
      {help}{" "}
      <a
        href="https://huggingface.co/settings/tokens"
        target="_blank"
        rel="noreferrer"
        className="underline"
      >
        huggingface.co/settings/tokens
      </a>
    </span>
  );

  return (
    <Section title="Hugging Face" defaultOpen>
      <form onSubmit={lookup} className="flex gap-2 px-4">
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="https://huggingface.co/mlx-community/…"
          className="h-9 font-mono text-xs"
          aria-label="Hugging Face model URL"
          disabled={locked}
        />
        <Tooltip>
          <TooltipTrigger asChild>
            <span>
              <Button
                type="submit"
                variant="secondary"
                size="icon-sm"
                className="size-9"
                aria-label="Find MLX quants"
                disabled={locked || !draft.trim() || busy !== null}
              >
                <Search className={cn(busy === "search" && "animate-spin")} />
              </Button>
            </span>
          </TooltipTrigger>
          {locked ? <TooltipContent className="max-w-xs">{tokenTip}</TooltipContent> : null}
        </Tooltip>
      </form>
      {results.length ? (
        <div className="mt-2 space-y-2 px-4 pb-3">
          <select
            className="flex h-9 w-full rounded-md border border-input bg-card px-2 font-mono text-xs"
            value={picked}
            onChange={(e) => setPicked(e.target.value)}
            aria-label="MLX quant"
            disabled={locked}
          >
            {results.map((row) => (
              <option key={row.id} value={row.id}>
                {row.quant} · {row.id}
              </option>
            ))}
          </select>
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="block">
                <button
                  type="button"
                  disabled={locked || !picked || pickedBusy}
                  onClick={() => void download()}
                  className="flex min-h-9 w-full items-center justify-center rounded-lg border border-border px-3 text-xs font-medium disabled:opacity-40"
                >
                  Download
                </button>
              </span>
            </TooltipTrigger>
            {pickedBusy ? (
              <TooltipContent>Already downloading this quant</TooltipContent>
            ) : null}
          </Tooltip>
          {active.map((job) => (
            <DownloadBubble
              key={job.repo}
              job={job}
              onPause={() => void postHubPause(job.repo)}
              onResume={() => void postHubResume(job.repo)}
              onCancel={() => void postHubCancel(job.repo)}
            />
          ))}
        </div>
      ) : (
        <p className="px-4 pb-3 pt-2 text-[11px] text-muted-foreground">
          {locked ? (
            <>
              Set <span className="font-mono">HF_TOKEN</span> when launching Edge.{" "}
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noreferrer"
                className="underline hover:text-foreground"
              >
                Create a token
              </a>
              .
            </>
          ) : (
            "Paste a Hub URL or org/name. Dropdown lists MLX quants."
          )}
        </p>
      )}
    </Section>
  );
}

function hubPctLabel(job: HubProgress): string {
  if (job.pct == null || !job.total) return "";
  if (job.pct < 1) return `${job.pct.toFixed(1)}%`;
  return `${Math.round(job.pct)}%`;
}

function DownloadBubble({
  job,
  onPause,
  onResume,
  onCancel,
}: {
  job: HubProgress;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
}) {
  const fill = Math.max(0, Math.min(1, job.ratio ?? 0));
  const pct = hubPctLabel(job);
  const label = [job.phase === "paused" ? "Paused" : job.name || job.repo, pct, job.detail]
    .filter(Boolean)
    .join(" · ");
  return (
    <div className="relative flex min-h-9 overflow-hidden rounded-lg border border-border">
      <div className="relative flex min-w-0 flex-1 items-center px-3 text-xs font-medium">
        {job.total ? (
          <span
            className="load-fill"
            style={{ "--load-pct": String(fill) } as CSSProperties}
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(fill * 100)}
          />
        ) : (
          <span className="load-fill animate-pulse" style={{ "--load-pct": "0.15" } as CSSProperties} />
        )}
        <span className="relative z-[1] truncate px-1">{label}</span>
      </div>
      <button
        type="button"
        className="relative z-[1] flex w-9 shrink-0 items-center justify-center border-l border-border bg-secondary hover:bg-accent"
        aria-label={job.phase === "paused" ? "Resume download" : "Pause download"}
        onClick={job.phase === "paused" ? onResume : onPause}
      >
        {job.phase === "paused" ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}
      </button>
      <button
        type="button"
        className="relative z-[1] flex w-9 shrink-0 items-center justify-center border-l border-border bg-destructive text-primary-foreground hover:opacity-90"
        aria-label="Cancel download"
        onClick={onCancel}
      >
        <Square className="size-3.5" />
      </button>
    </div>
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
