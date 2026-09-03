import { type EngineKind } from "./flags";
import { type ServedRuntime } from "./edge-api";

export function engineLabel(engine: EngineKind) {
  if (engine === "vlm") return "mlx-vlm";
  if (engine === "embed") return "mlx-embed";
  if (engine === "tts") return "mlx-tts";
  if (engine === "stt") return "mlx-stt";
  if (engine === "rerank") return "mlx-rerank";
  if (engine === "image") return "mlx-image";
  return "mlx-lm";
}

export function loadedSummary(loaded: ServedRuntime[]): string {
  if (!loaded.length) return "idle · play on a model card to hot-load /v1";
  if (loaded.length === 1) return `1 loaded · ${loaded[0]?.name || loaded[0]?.repo}`;
  return `${loaded.length} loaded`;
}
