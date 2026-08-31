import { useState } from "react";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { engineLabel } from "@/lib/command";
import { modelIsLive } from "@/lib/edge-api";
import { flagsFor, type FlagDef, type FlagGroup } from "@/lib/flags";
import { useStudio } from "@/lib/studio-store";

const GROUP_LABEL: Record<FlagGroup, string> = {
  server: "Server",
  sampling: "Sampling",
  thinking: "Thinking",
};

export function FlagPanel() {
  const model = useStudio((s) => s.selected());
  const flags = useStudio((s) => s.flags);
  const served = useStudio((s) => s.served);
  const setFlag = useStudio((s) => s.setFlag);
  const resetFlags = useStudio((s) => s.resetFlags);
  const [advanced, setAdvanced] = useState(false);

  if (!model) {
    return <p className="text-sm text-muted-foreground">Pick a model in the sidebar.</p>;
  }

  const visible = flagsFor(model.engine, false);
  const extra = flagsFor(model.engine, true);
  const groups: FlagGroup[] = ["server", "sampling", "thinking"];
  const live = modelIsLive(served, model);

  return (
    <div className="mx-auto w-full max-w-3xl space-y-8">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-medium tracking-tight">Engine switches</h2>
          <p className="mt-1 max-w-lg text-sm text-muted-foreground">
            These map 1:1 onto {engineLabel(model.engine)} flags. Serve passes them through mlx-edge.
            {live ? " Unload and Serve again after changing flags." : " Serve hot-loads this model beside any that are already up."}
          </p>
        </div>
        <Button type="button" variant="ghost" size="sm" onClick={resetFlags}>
          <RotateCcw />
          Reset
        </Button>
      </div>
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
