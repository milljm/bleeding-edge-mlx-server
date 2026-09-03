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
  features?: {
    tool?: boolean;
    vision?: boolean;
    reason?: boolean;
  };
};

/** Common watch path (LM Studio). Kept as-is — never treated as a wipe sentinel. */
export const LEGACY_DEFAULT_WATCH = "~/.lmstudio/models";

/** Default Hugging Face hub cache — mlx-lm / mlx-vlm download here. */
export const HF_HUB_WATCH = "~/.cache/huggingface/hub";

/** Ollama local weights (GGUF blobs + manifests). */
export const OLLAMA_WATCH = "~/.ollama/models";

export const DIR_PLACEHOLDER = "~/.cache/huggingface/hub";

export const SUGGESTED_WATCH: { label: string; path: string }[] = [
  { label: "Hugging Face", path: HF_HUB_WATCH },
  { label: "LM Studio", path: LEGACY_DEFAULT_WATCH },
  { label: "Ollama", path: OLLAMA_WATCH },
];

export function migrateWatchDirs(dirs?: string[] | null): string[] {
  if (!dirs || dirs.length === 0) return [];
  return [...new Set(dirs.map((d) => d.trim()).filter((d) => d.length > 0))];
}

export type ModelOrigin = "huggingface" | "lmstudio" | "ollama" | "local";

export function modelOrigin(model: Pick<ModelRec, "path" | "watchDir" | "repo">): ModelOrigin {
  const blob = `${model.watchDir} ${model.path} ${model.repo}`.replace(/\\/g, "/").toLowerCase();
  if (blob.includes("huggingface") || blob.includes("models--") || /\/hub\//.test(blob)) return "huggingface";
  if (blob.includes("lmstudio") || blob.includes(".lmstudio")) return "lmstudio";
  if (blob.includes("ollama")) return "ollama";
  return "local";
}

export function originLabel(origin: ModelOrigin): string {
  return {
    huggingface: "Hugging Face",
    lmstudio: "LM Studio",
    ollama: "Ollama",
    local: "Local",
  }[origin];
}

/** Open the upstream model page. Only when the watch path is clearly Hub or Ollama. */
export function modelCardLink(model: Pick<ModelRec, "repo" | "path" | "watchDir" | "name">): {
  href: string;
  host: "huggingface" | "ollama";
} | null {
  const watch = (model.watchDir || "").replace(/\\/g, "/").toLowerCase();
  const path = (model.path || "").replace(/\\/g, "/").toLowerCase();
  const blob = `${watch} ${path}`;
  if (blob.includes("huggingface") || blob.includes("models--") || /\/hub\//.test(blob)) {
    const id = huggingFaceId(model);
    if (id) return { href: `https://huggingface.co/${id}`, host: "huggingface" };
  }
  if (blob.includes("ollama")) {
    const name = ollamaLibraryName(model);
    if (name) return { href: `https://ollama.com/library/${encodeURIComponent(name)}`, host: "ollama" };
  }
  return null;
}

function huggingFaceId(model: Pick<ModelRec, "repo" | "path">): string | null {
  const repo = model.repo.trim();
  if (/^[^/]+\/[^/]+$/.test(repo)) return repo;
  const match = model.path.replace(/\\/g, "/").match(/models--([^/]+)--([^/]+)/);
  return match ? `${match[1]}/${match[2]}` : null;
}

function ollamaLibraryName(model: Pick<ModelRec, "repo" | "path" | "name">): string | null {
  const match = model.path.replace(/\\/g, "/").match(/\/library\/([^/]+)/i);
  if (match) return match[1];
  const leaf = (model.repo.split("/").pop() || model.name || "").trim();
  return leaf || null;
}

export function publicName(model: Pick<ModelRec, "path" | "repo" | "name">) {
  const fromPath = model.path.split(/[/\\]/).filter(Boolean).pop();
  const fromRepo = model.repo.split(/[/\\]/).filter(Boolean).pop();
  // Hugging Face snapshots are content-addressed SHAs — clients want chatterbox, not 05e904….
  if (fromPath && /^[0-9a-f]{7,40}$/i.test(fromPath)) return fromRepo || model.name || fromPath;
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

