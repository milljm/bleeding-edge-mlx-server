import { type CSSProperties, useEffect, useState } from "react";
import { getHost } from "@/lib/edge-api";
import { cn } from "@/lib/utils";

const GIG = 1024 ** 3;

function gbLabel(bytes: number) {
  const n = bytes / GIG;
  if (n >= 10) return String(Math.round(n));
  return n.toFixed(1).replace(/\.0$/, "");
}

function ResourceRow({
  label,
  value,
  ratio,
  kind,
  tall,
}: {
  label: string;
  value: string;
  ratio: number;
  kind: "memory" | "gpu";
  tall?: boolean;
}) {
  const pct = Math.max(0, Math.min(1, ratio));
  return (
    <div
      className={cn("relative min-h-0 flex-1 overflow-hidden", tall ? "h-full" : "")}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct * 100)}
    >
      <span
        className={cn("resource-fill", kind === "gpu" && "is-gpu")}
        style={{ "--load-pct": String(pct) } as CSSProperties}
      />
      <span className="relative z-[1] flex h-full items-center justify-between gap-2 px-2.5 font-mono text-[11px] tracking-wide">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums text-foreground">{value}</span>
      </span>
    </div>
  );
}

export function ResourceMeter() {
  const [used, setUsed] = useState<number | null>(null);
  const [total, setTotal] = useState(0);
  const [gpu, setGpu] = useState<number | null>(null);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const snap = await getHost();
        if (stop) return;
        const nextUsed = snap.memory.used_bytes;
        const nextTotal = snap.memory.total_bytes;
        setTotal(nextTotal);
        setUsed((prev) => {
          if (prev === null) return nextUsed;
          if (Math.abs(nextUsed - prev) >= GIG) return nextUsed;
          return prev;
        });
        if (snap.gpu) {
          const pct = Math.round(snap.gpu.percent);
          setGpu((prev) => (prev === null || prev !== pct ? pct : prev));
        } else {
          setGpu(null);
        }
      } catch {
        /* keep last sample */
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 1000);
    return () => {
      stop = true;
      window.clearInterval(id);
    };
  }, []);

  const split = gpu !== null;
  const memRatio = total > 0 && used !== null ? used / total : 0;
  const memText =
    used === null || total <= 0 ? "—" : `${gbLabel(used)} / ${gbLabel(total)} GB`;

  return (
    <div className="px-2 pb-3">
      <div
        className={cn(
          "relative flex h-14 w-full flex-col overflow-hidden rounded-lg bg-secondary/50 ring-1 ring-border/60",
          split && "divide-y divide-border/50",
        )}
      >
        <ResourceRow label="Memory" value={memText} ratio={memRatio} kind="memory" tall={!split} />
        {split ? <ResourceRow label="GPU" value={`${gpu}%`} ratio={(gpu ?? 0) / 100} kind="gpu" /> : null}
      </div>
    </div>
  );
}
