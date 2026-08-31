import type { EngineKind } from "./flags.ts";

export type ModelRec = {
  id: string;
  name: string;
  repo: string;
  path: string;
  engine: EngineKind;
  size: string;
  quant: string;
  watchDir: string;
  source: "scan" | "manual";
};

/** Pre-0.4 persisted default. Migrated to an empty watch list. */
export const LEGACY_DEFAULT_WATCH = "~/.lmstudio/models";

export const DIR_PLACEHOLDER = "/path/to/models";

export function migrateWatchDirs(dirs?: string[] | null): string[] {
  if (!dirs || dirs.length === 0) return [];
  if (dirs.length === 1 && dirs[0] === LEGACY_DEFAULT_WATCH) return [];
  return dirs.filter((d) => d.trim().length > 0);
}

export function slugModelId(engine: EngineKind, repo: string, source: ModelRec["source"] = "scan") {
  const base = `${engine}-${repo.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`.replace(/-+$/g, "");
  return source === "manual" ? `custom-${base}` : base;
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