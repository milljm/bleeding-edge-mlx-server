import type { EngineKind } from "./flags";

export type ModelRec = {
  id: string;
  name: string;
  repo: string;
  path: string;
  engine: EngineKind;
  size: string;
  quant: string;
  watchDir: string;
  source: "seed" | "scan";
};

export const DEFAULT_WATCH = "~/.lmstudio/models";

export const SEED_MODELS: ModelRec[] = [
  {
    id: "qwen3-8b-4bit",
    name: "Qwen3 8B",
    repo: "mlx-community/Qwen3-8B-4bit",
    path: "~/.lmstudio/models/mlx-community/Qwen3-8B-4bit",
    engine: "lm",
    size: "4.5 GB",
    quant: "4-bit",
    watchDir: DEFAULT_WATCH,
    source: "seed",
  },
  {
    id: "llama-3.2-3b-4bit",
    name: "Llama 3.2 3B Instruct",
    repo: "mlx-community/Llama-3.2-3B-Instruct-4bit",
    path: "~/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit",
    engine: "lm",
    size: "1.8 GB",
    quant: "4-bit",
    watchDir: DEFAULT_WATCH,
    source: "seed",
  },
  {
    id: "mistral-7b-v03-4bit",
    name: "Mistral 7B Instruct v0.3",
    repo: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    path: "~/.lmstudio/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    engine: "lm",
    size: "4.1 GB",
    quant: "4-bit",
    watchDir: DEFAULT_WATCH,
    source: "seed",
  },
  {
    id: "qwen25-vl-7b-4bit",
    name: "Qwen2.5 VL 7B Instruct",
    repo: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    path: "~/.lmstudio/models/mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    engine: "vlm",
    size: "4.8 GB",
    quant: "4-bit",
    watchDir: DEFAULT_WATCH,
    source: "seed",
  },
  {
    id: "qwen2-vl-2b-4bit",
    name: "Qwen2 VL 2B Instruct",
    repo: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    path: "~/.lmstudio/models/mlx-community/Qwen2-VL-2B-Instruct-4bit",
    engine: "vlm",
    size: "1.3 GB",
    quant: "4-bit",
    watchDir: DEFAULT_WATCH,
    source: "seed",
  },
  {
    id: "deepseek-ocr-4bit",
    name: "DeepSeek OCR",
    repo: "mlx-community/DeepSeek-OCR-4bit",
    path: "~/.lmstudio/models/mlx-community/DeepSeek-OCR-4bit",
    engine: "vlm",
    size: "2.1 GB",
    quant: "4-bit",
    watchDir: DEFAULT_WATCH,
    source: "seed",
  },
];

export function isLmStudioPath(dir: string) {
  const n = dir.replace(/\\/g, "/").toLowerCase().replace(/\/+$/, "");
  return n.includes(".lmstudio/models") || n.endsWith("lmstudio/models");
}

export function slugModelId(engine: EngineKind, repo: string) {
  return `custom-${engine}-${repo.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`.replace(
    /-+$/g,
    "",
  );
}

export function modelFromRepo(
  repo: string,
  engine: EngineKind,
  watchDir: string,
): ModelRec | null {
  const trimmed = repo.trim().replace(/^['"]|['"]$/g, "");
  if (!trimmed) return null;
  const name = trimmed.split("/").filter(Boolean).pop() ?? trimmed;
  const base = watchDir.replace(/\/+$/, "") || DEFAULT_WATCH;
  return {
    id: slugModelId(engine, trimmed),
    name,
    repo: trimmed,
    path: `${base}/${trimmed}`,
    engine,
    size: "—",
    quant: "local",
    watchDir: base,
    source: "scan",
  };
}

export function modelsForDirs(dirs: string[], extra: ModelRec[] = []): ModelRec[] {
  const out: ModelRec[] = [];
  for (const raw of dirs) {
    const dir = raw.trim();
    if (!dir) continue;
    if (isLmStudioPath(dir) || dir === DEFAULT_WATCH) {
      out.push(
        ...SEED_MODELS.map((m) => ({
          ...m,
          watchDir: dir,
          path: `${dir.replace(/\/+$/, "")}/${m.repo}`,
        })),
      );
    }
  }
  out.push(...extra.filter((m) => dirs.some((d) => d.trim() === m.watchDir)));
  const seen = new Set<string>();
  return out.filter((m) => {
    const key = m.id;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
