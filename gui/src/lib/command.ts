import { flagArgs, type EngineKind, type FlagValues } from "./flags";
import { loadTarget, type ModelRec } from "./models";
import { DEFAULT_GATEWAY, openaiUrl, type GatewayInfo, type ServedRuntime } from "./edge-api";

function shQuote(value: string): string {
  if (value === "") return '""';
  if (/^[A-Za-z0-9_./:@%+=,-]+$/.test(value)) return value;
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

export function guiCommand(gateway: GatewayInfo = DEFAULT_GATEWAY): string {
  return ["edge-gui", "--host", gateway.host, "--port", String(gateway.port)].map(shQuote).join(" ");
}

export function gatewayCommand(gateway: GatewayInfo = DEFAULT_GATEWAY): string {
  return ["mlx-edge", "serve", "--host", gateway.host, "--port", String(gateway.port)].map(shQuote).join(" ");
}

export function loadCommand(model: ModelRec, flags: FlagValues): string {
  const args = [
    "mlx-edge",
    "load",
    "--engine",
    model.engine,
    "--model",
    loadTarget(model),
    ...flagArgs(model.engine, flags, ["host", "port"]),
  ];
  return args.map(shQuote).join(" ");
}

export function serveCommand(model: ModelRec, flags: FlagValues, gateway: GatewayInfo = DEFAULT_GATEWAY): string {
  return `${guiCommand(gateway)}  # then Serve, or:\n${loadCommand(model, flags)}`;
}

export function engineLabel(engine: EngineKind) {
  return engine === "vlm" ? "mlx-vlm" : "mlx-lm";
}

export function loadedSummary(loaded: ServedRuntime[]): string {
  if (!loaded.length) return "idle · press Serve to hot-load /v1";
  if (loaded.length === 1) return `1 loaded · ${loaded[0]?.repo}`;
  return `${loaded.length} loaded`;
}

export { openaiUrl };
