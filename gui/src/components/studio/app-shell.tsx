import { useEffect, useState } from "react";
import { Check, Copy, Menu, PanelLeft, Play, RefreshCw, Square, X } from "lucide-react";
import { toast, Toaster } from "sonner";
import { EndpointPanel } from "@/components/studio/endpoint";
import { FlagPanel } from "@/components/studio/flag-panel";
import { Playground } from "@/components/studio/playground";
import { Sidebar } from "@/components/studio/sidebar";
import { ThemeToggle } from "@/components/studio/theme-toggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import { getPrefs, modelIsLive } from "@/lib/edge-api";
import { flagsDirty } from "@/lib/flags";
import { engineLabel, loadedSummary } from "@/lib/command";
import { useStudio, type StudioTab } from "@/lib/studio-store";
import { cn } from "@/lib/utils";

const SIDEBAR_MIN = 280;
const SIDEBAR_MAX = 480;
const SIDEBAR_DEFAULT = 320;

export function AppShell() {
  useRehydrateStudio();
  const [navOpen, setNavOpen] = useState(false);
  const sidebar = useSidebarLayout();
  const model = useStudio((s) => s.selected());
  const served = useStudio((s) => s.served);
  const flags = useStudio((s) => s.flags);
  const gateway = useStudio((s) => s.gateway);
  const tab = useStudio((s) => s.tab);
  const setTab = useStudio((s) => s.setTab);
  const startServe = useStudio((s) => s.startServe);
  const stopServe = useStudio((s) => s.stopServe);
  const reloadServe = useStudio((s) => s.reloadServe);
  const loadingId = useStudio((s) => s.loadingId);
  const failed = useStudio((s) => s.failed);

  const live = modelIsLive(served, model);
  const loaded = served.find((row) => modelIsLive([row], model));
  const dirty = Boolean(live && model && flagsDirty(model.engine, flags, loaded?.flags));
  const loadingThis = Boolean(model && loadingId === model.id);
  const failedThis = model ? failed[model.id] : undefined;

  async function onServe() {
    if (loadingThis) return;
    try {
      if (live) await stopServe();
      else await startServe();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Serve failed");
    }
  }

  async function onReload() {
    if (loadingThis || !live) return;
    try {
      await reloadServe();
      toast.success("Model reloaded");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Reload failed");
    }
  }

  return (
    <TooltipProvider>
      <div
        className={cn(
          "relative flex h-dvh overflow-hidden bg-background paper text-foreground",
          sidebar.dragging && "select-none",
        )}
      >
        <div
          className={cn(
            "relative hidden shrink-0 overflow-hidden md:block",
            !sidebar.dragging &&
              "transition-[width] duration-[var(--motion-fast)] ease-[var(--ease-smooth-out)]",
            sidebar.open ? "border-r border-border" : "border-r-0",
          )}
          style={{ width: sidebar.open ? sidebar.width : 0 }}
        >
          <div className="h-full min-w-0 overflow-hidden" style={{ width: sidebar.width }}>
            <Sidebar onCollapse={() => sidebar.setOpen(false)} />
          </div>
          {sidebar.open ? (
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize sidebar"
              tabIndex={0}
              className="absolute top-0 right-0 z-20 h-full w-1.5 cursor-col-resize hover:bg-foreground/20"
              onPointerDown={(e) => {
                e.preventDefault();
                sidebar.setDragging(true);
                const startX = e.clientX;
                const startW = sidebar.width;
                const move = (ev: PointerEvent) => {
                  const next = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, startW + ev.clientX - startX));
                  sidebar.setWidth(next);
                };
                const up = () => {
                  sidebar.setDragging(false);
                  window.removeEventListener("pointermove", move);
                  window.removeEventListener("pointerup", up);
                };
                window.addEventListener("pointermove", move);
                window.addEventListener("pointerup", up);
              }}
            />
          ) : null}
        </div>

        {navOpen ? (
          <div className="fixed inset-0 z-40 md:hidden">
            <button
              type="button"
              className="absolute inset-0 bg-background/70"
              aria-label="Close sidebar"
              onClick={() => setNavOpen(false)}
            />
            <div className="relative h-full w-[min(20rem,88vw)] border-r border-border bg-card paper shadow-[var(--shadow-border)]">
              <div className="absolute top-2 right-2 z-10">
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-11"
                  aria-label="Close"
                  onClick={() => setNavOpen(false)}
                >
                  <X />
                </Button>
              </div>
              <Sidebar onNavigate={() => setNavOpen(false)} />
            </div>
          </div>
        ) : null}

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex items-center gap-2 border-b border-border px-3 py-2">
            <Button
              variant="ghost"
              size="icon"
              className="size-11 md:hidden"
              aria-label="Open models"
              onClick={() => setNavOpen(true)}
            >
              <Menu />
            </Button>
            {!sidebar.open ? (
              <Button
                variant="ghost"
                size="icon-sm"
                className="hidden md:inline-flex"
                aria-label="Show sidebar"
                onClick={() => sidebar.setOpen(true)}
              >
                <PanelLeft />
              </Button>
            ) : null}
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <h1 className="truncate text-sm font-medium sm:text-base">
                  {model?.name ?? "No model"}
                </h1>
                {model ? <Badge>{engineLabel(model.engine)}</Badge> : null}
                {live ? (
                  <Badge variant="ok" className="gap-1.5">
                    <span className="size-1.5 animate-pulse rounded-full bg-ok" />
                    loaded
                  </Badge>
                ) : failedThis ? (
                  <Badge variant="bleed">failed</Badge>
                ) : null}
                {served.length > 1 ? <Badge variant="ok">{served.length} loaded</Badge> : null}
              </div>
              <p className="truncate font-mono text-xs text-muted-foreground">
                {failedThis && !live ? failedThis : model?.repo}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              {dirty ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => void onReload()}
                  disabled={!model || loadingThis}
                  aria-label="Reload model"
                >
                  <RefreshCw className={cn("size-3.5", loadingThis && "animate-spin")} />
                  Reload
                </Button>
              ) : null}
              <Button
                type="button"
                variant={live ? "destructive" : failedThis ? "destructive" : "default"}
                size="sm"
                onClick={() => void onServe()}
                disabled={!model || loadingThis}
              >
                {live ? (
                  <>
                    <Square className="size-3.5" />
                    Unload
                  </>
                ) : (
                  <>
                    <Play className={cn("size-3.5", loadingThis && "animate-pulse")} />
                    {loadingThis ? "Loading" : "Serve"}
                  </>
                )}
              </Button>
              <ThemeToggle />
            </div>
          </header>

          <Tabs
            value={tab}
            onValueChange={(value) => setTab(value as StudioTab)}
            className="flex min-h-0 flex-1 flex-col gap-0"
          >
            <div className="border-b border-border px-3 py-2">
              <TabsList>
                <TabsTrigger value="settings">Settings</TabsTrigger>
                <TabsTrigger value="playground">Playground</TabsTrigger>
                <TabsTrigger value="endpoint">Endpoint</TabsTrigger>
              </TabsList>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-6">
              <TabsContent value="settings">
                <FlagPanel />
              </TabsContent>
              <TabsContent value="playground" className="flex h-full min-h-80 flex-col">
                <Playground />
              </TabsContent>
              <TabsContent value="endpoint">
                <EndpointPanel />
              </TabsContent>
            </div>
          </Tabs>

          <footer className="flex items-center justify-between gap-3 border-t border-border px-4 py-2 font-mono text-xs text-muted-foreground">
            <span className="min-w-0 truncate">{loadedSummary(served)}</span>
            <ServingUrl url={gateway.url} bind={gateway.bind} />
          </footer>
        </div>
      </div>
      <ThemeToaster />
    </TooltipProvider>
  );
}

function ServingUrl({ url, bind }: { url: string; bind: string }) {
  const [copied, setCopied] = useState(false);
  const remote = bind.startsWith("0.0.0.0:") || bind.startsWith("[::]:") || bind.startsWith(":::");
  return (
    <button
      type="button"
      className="flex min-w-0 shrink-0 items-center gap-1.5 text-right hover:text-foreground"
      title={remote ? `Bound on ${bind}. Copy OpenAI base URL` : "Copy OpenAI base URL"}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(url);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        } catch {
          /* ignore */
        }
      }}
    >
      <span className="truncate">
        Serving on <strong className="font-medium text-foreground">{url}</strong>
      </span>
      {copied ? <Check className="size-3.5 shrink-0 text-ok" /> : <Copy className="size-3.5 shrink-0" />}
    </button>
  );
}

function ThemeToaster() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  useEffect(() => {
    const root = document.documentElement;
    const read = () => setTheme(root.dataset.theme === "light" ? "light" : "dark");
    read();
    const obs = new MutationObserver(read);
    obs.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);
  return (
    <Toaster
      theme={theme}
      position="top-center"
      toastOptions={{
        className: "bg-popover text-popover-foreground shadow-[var(--shadow-border)] border-0",
      }}
    />
  );
}

function useSidebarLayout() {
  const [open, setOpen] = useState(true);
  const [width, setWidth] = useState(SIDEBAR_DEFAULT);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    try {
      if (window.localStorage.getItem("edge-sidebar") === "0") setOpen(false);
      const n = Number(window.localStorage.getItem("edge-sidebar-w"));
      if (Number.isFinite(n)) setWidth(Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, n)));
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("edge-sidebar", open ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [open]);

  useEffect(() => {
    try {
      window.localStorage.setItem("edge-sidebar-w", String(Math.round(width)));
    } catch {
      /* ignore */
    }
  }, [width]);

  return { open, setOpen, width, setWidth, dragging, setDragging };
}

function useRehydrateStudio() {
  const syncServed = useStudio((s) => s.syncServed);
  const scanWatchDirs = useStudio((s) => s.scanWatchDirs);
  const applyPrefs = useStudio((s) => s.applyPrefs);
  useEffect(() => {
    const api = useStudio.persist;
    let cancelled = false;
    void Promise.resolve(api.rehydrate())
      .catch(() => undefined)
      .then(async () => {
        if (cancelled) return;
        try {
          applyPrefs(await getPrefs());
        } catch {
          /* localStorage watch dirs still apply */
        }
        if (cancelled) return;
        try {
          await scanWatchDirs();
        } catch {
          /* empty watch list is fine */
        }
        if (cancelled) return;
        try {
          await syncServed();
        } catch {
          /* preview without a gateway is still usable */
        }
      });
    return () => {
      cancelled = true;
    };
  }, [syncServed, scanWatchDirs, applyPrefs]);
}
