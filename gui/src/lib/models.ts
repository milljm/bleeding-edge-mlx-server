import type { EngineKind } from "./flags.ts";

export type ModelRec = {
  id: string;
  name: string;
  repo: string;
  path: string;
  engine: EngineKind;
  /** Scan guess, before a per-model Engine override. */
  detectedEngine?: EngineKind;
  size: string;
  quant: string;
  context?: number | null;
  watchDir: string;
  source: "scan" | "manual";
  hasChatTemplate?: boolean;
};

/** Common watch path (LM Studio). Kept as-is — never treated as a wipe sentinel. */
export const LEGACY_DEFAULT_WATCH = "~/.lmstudio/models";

export const DIR_PLACEHOLDER = "/path/to/models";

export function migrateWatchDirs(dirs?: string[] | null): string[] {
  if (!dirs || dirs.length === 0) return [];
  return [...new Set(dirs.map((d) => d.trim()).filter((d) => d.length > 0))];
}

export function slugModelId(engine: EngineKind, repo: string, source: ModelRec["source"] = "scan") {
  const base = `${engine}-${repo.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`.replace(/-+$/g, "");
  return source === "manual" ? `custom-${base}` : base;
}

export function publicName(model: Pick<ModelRec, "path" | "repo" | "name">) {
  const fromPath = model.path.split(/[/\\]/).filter(Boolean).pop();
  const fromRepo = model.repo.split(/[/\\]/).filter(Boolean).pop();
  return fromPath || fromRepo || model.name;
}

export function formatContext(n?: number | null) {
  if (!n || n <= 0) return null;
  if (n % 1024 === 0 && n >= 1024) return `${n / 1024}k`;
  return n.toLocaleString("en-US");
}

export function flagKey(model: Pick<ModelRec, "repo" | "id" | "path">) {
  return model.repo || model.id || model.path;
}

export function applyEngineOverrides(
  models: ModelRec[],
  engineByModel: Partial<Record<string, EngineKind>> = {},
): ModelRec[] {
  return models.map((model) => {
    const detected = model.detectedEngine ?? model.engine;
    const override = engineByModel[flagKey(model)];
    return { ...model, detectedEngine: detected, engine: override ?? detected };
  });
}

export function modelFromRepo(repo: string, engine: EngineKind, watchDir: string): ModelRec | null {
  const trimmed = repo.trim().replace(/^['"]|['"]$/g, "");
  if (!trimmed) return null;
  const name = trimmed.split("/").filter(Boolean).pop() ?? trimmed;
  const base = watchDir.replace(/\/+$/, "");
  return {
    id: slugModelId(engine, trimmed, "manual"),
    name,
    repo: trimmed,
    path: base ? `${base}/${trimmed}` : trimmed,
    engine,
    size: "—",
    quant: "local",
    watchDir: base,
    source: "manual",
  };
}

export function mergeCatalog(scanned: ModelRec[], extra: ModelRec[] = []): ModelRec[] {
  const out: ModelRec[] = [];
  const seen = new Set<string>();
  const seenRepos = new Set<string>();
  for (const model of [...scanned, ...extra]) {
    const repoKey = model.repo.toLowerCase();
    if (seen.has(model.id) || seenRepos.has(repoKey)) continue;
    seen.add(model.id);
    seenRepos.add(repoKey);
    out.push(model);
  }
  return out;
}

/** Identifier passed to mlx-edge load/unload: filesystem path for scans, Hub id for extras. */
export function loadTarget(model: Pick<ModelRec, "source" | "path" | "repo">) {
  return model.source === "manual" ? model.repo : model.path || model.repo;
}

/** Loaded / in-use models float to the top; both groups are A–Z (a/A together). */
export function sortLoadedFirst<T extends { name: string }>(
  models: T[],
  isLive: (model: T) => boolean,
): T[] {
  return [...models].sort((a, b) => {
    const live = Number(isLive(b)) - Number(isLive(a));
    if (live !== 0) return live;
    return a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
  });
}

