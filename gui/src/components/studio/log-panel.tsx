import { useEffect, useRef, useState } from "react";
import { Pause, Play, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { clearLogs, getLogs, type LogLevel, type LogLine } from "@/lib/edge-api";
import { publicName } from "@/lib/models";
import { useStudio } from "@/lib/studio-store";
import { cn } from "@/lib/utils";

type Filter = "all" | "model" | "error";

const LEVEL_CLASS: Record<LogLevel, string> = {
  error: "text-destructive",
  warn: "text-warn",
  progress: "text-ok",
  http: "text-foreground",
  info: "text-muted-foreground",
};

export function LogPanel() {
  const model = useStudio((s) => s.selected());
  const served = useStudio((s) => s.served);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [stick, setStick] = useState(true);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  const scroller = useRef<HTMLDivElement>(null);
  const needle = model ? publicName(model) : "";
  pausedRef.current = paused;

  useEffect(() => {
    let stop = false;
    const stream = new EventSource("/v1/logs/stream");
    stream.onmessage = (event) => {
      if (pausedRef.current) return;
      try {
        const body = JSON.parse(event.data) as { seq?: number; lines?: LogLine[] };
        const incoming = body.lines ?? [];
        if (!incoming.length) return;
        setLines((prev) => {
          const seen = new Set(prev.map((row) => row.seq));
          const next = [...prev];
          for (const row of incoming) {
            if (!seen.has(row.seq)) next.push(row);
          }
          return next.slice(-2000);
        });
      } catch {
        /* ignore malformed */
      }
    };
    void getLogs()
      .then((snap) => {
        if (stop) return;
        setLines(snap.lines);
      })
      .catch(() => undefined);
    return () => {
      stop = true;
      stream.close();
    };
  }, []);

  useEffect(() => {
    if (!stick || paused) return;
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines, stick, filter, paused]);

  const visible = lines.filter((row) => {
    if (filter === "error") return row.level === "error" || row.level === "warn";
    if (filter === "model" && needle) return row.model.toLowerCase().includes(needle.toLowerCase());
    return true;
  });

  return (
    <div className="mx-auto flex h-full min-h-0 w-full max-w-5xl flex-1 flex-col">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="mr-auto text-lg font-medium tracking-tight">Logging</h2>
        <FilterChip label="All" active={filter === "all"} onClick={() => setFilter("all")} />
        <FilterChip
          label={needle ? needle : "This model"}
          active={filter === "model"}
          onClick={() => setFilter("model")}
          disabled={!needle}
        />
        <FilterChip label="Errors" active={filter === "error"} onClick={() => setFilter("error")} />
        <label className="flex items-center gap-1.5 px-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={stick}
            onChange={(e) => setStick(e.target.checked)}
            className="accent-foreground"
          />
          Follow
        </label>
        <Button
          type="button"
          variant={paused ? "secondary" : "ghost"}
          size="sm"
          onClick={() => setPaused((v) => !v)}
          aria-pressed={paused}
        >
          {paused ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}
          {paused ? "Resume" : "Pause"}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => {
            setLines([]);
            void clearLogs();
          }}
        >
          <Trash2 className="size-3.5" />
          Clear
        </Button>
      </div>
      <div
        ref={scroller}
        className="min-h-0 flex-1 overflow-auto rounded-2xl bg-card px-3 py-3 font-mono text-xs leading-5 shadow-[var(--shadow-border)]"
      >
        {paused ? (
          <p className="mb-2 text-xs text-warn">Paused — missed lines are dropped. Resume continues from live.</p>
        ) : null}
        {visible.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {served.length
              ? "Waiting for engine output. Serve, chat, or embed and the child stdout lands here."
              : "Serve a model to stream engine logs."}
          </p>
        ) : (
          <ol className="space-y-0.5">
            {visible.map((row) => (
              <li key={row.seq} className="flex gap-3">
                <span className="w-16 shrink-0 text-muted-foreground/70 tabular-nums">
                  {formatTime(row.ts)}
                </span>
                <span className="w-28 shrink-0 truncate text-muted-foreground">{row.model}</span>
                <span className={cn("min-w-0 flex-1 whitespace-pre-wrap break-all", LEVEL_CLASS[row.level] || LEVEL_CLASS.info)}>
                  {highlight(row.text, row.level)}
                </span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}

function FilterChip({
  label,
  active,
  onClick,
  disabled,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "min-h-11 max-w-[12rem] truncate rounded-full px-3 py-2 text-xs font-medium",
        active ? "bg-accent text-foreground" : "text-muted-foreground hover:text-foreground",
        disabled && "opacity-40",
      )}
    >
      {label}
    </button>
  );
}

function formatTime(ts: number) {
  const d = new Date(ts * 1000);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, "0")).join(":");
}

function highlight(text: string, _level: LogLevel) {
  const parts = text.split(/(\b\d+\/\d+\b|\b(?:GET|POST|PUT|DELETE|PATCH)\b|\b(?:ERROR|WARNING|INFO|DEBUG)\b)/g);
  return parts.map((part, i) => {
    if (/^\d+\/\d+$/.test(part)) {
      return (
        <span key={i} className="text-ok">
          {part}
        </span>
      );
    }
    if (part === "ERROR") {
      return (
        <span key={i} className="font-medium text-destructive">
          {part}
        </span>
      );
    }
    if (part === "WARNING") {
      return (
        <span key={i} className="text-warn">
          {part}
        </span>
      );
    }
    if (part === "GET" || part === "POST" || part === "PUT" || part === "DELETE" || part === "PATCH") {
      return (
        <span key={i} className="text-foreground">
          {part}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}
