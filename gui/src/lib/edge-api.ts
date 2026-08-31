import type { EngineKind, FlagValues } from "./flags";
import type { ModelRec } from "./models";

export type ServedRuntime = {
  id: string;
  name: string;
  repo: string;
  engine: EngineKind;
  host: string;
  port: number;
  startedAt: number;
  flags?: FlagValues;
};

export type GatewayInfo = {
  host: string;
  port: number;
  bind: string;
  url: string;
};

export const DEFAULT_GATEWAY: GatewayInfo = {
  host: "127.0.0.1",
  port: 8080,
  bind: "127.0.0.1:8080",
  url: "http://127.0.0.1:8080/v1",
};

export function openaiUrl(host = "127.0.0.1", port: number | string = 8080) {
  return `http://${host}:${port}/v1`;
}

export function modelIsLive(
  served: ServedRuntime[],
  model?: { id: string; repo: string; path?: string; name?: string } | null,
) {
  if (!model) return false;
  const needles = [model.repo, model.id, model.path, model.name].filter((n): n is string => Boolean(n));
  return served.some((s) => [s.repo, s.id, s.name].some((id) => needles.some((n) => sameModel(id, n))));
}

export function sameModel(a: string, b: string) {
  if (!a || !b) return false;
  const x = normalizeName(a);
  const y = normalizeName(b);
  if (!x || !y) return false;
  if (x === y) return true;
  if (x.endsWith(`/${y}`) || y.endsWith(`/${x}`)) return true;
  return x.split("/").pop() === y.split("/").pop();
}

function normalizeName(name: string) {
  return name.trim().replace(/\\/g, "/").replace(/\/+$/g, "").toLowerCase();
}

async function parseJson(res: Response) {
  const body = (await res.json().catch(() => ({}))) as {
    error?: { message?: string };
  };
  if (!res.ok) {
    throw new Error(body.error?.message || `HTTP ${res.status}`);
  }
  return body;
}

export async function getHealth(): Promise<{ models: string[]; gateway: GatewayInfo }> {
  const res = await fetch("/health");
  const body = (await parseJson(res)) as {
    models?: string[];
    host?: string;
    port?: number;
    bind?: string;
    url?: string;
  };
  const host = body.host || DEFAULT_GATEWAY.host;
  const port = Number(body.port || DEFAULT_GATEWAY.port);
  return {
    models: body.models ?? [],
    gateway: {
      host,
      port,
      bind: body.bind || `${host}:${port}`,
      url: body.url || openaiUrl(host, port),
    },
  };
}

export async function listServed(gateway: GatewayInfo): Promise<ServedRuntime[]> {
  const res = await fetch("/v1/models");
  const body = (await parseJson(res)) as {
    data?: { id?: string; owned_by?: string; created?: number }[];
  };
  return (body.data ?? []).map((row) => {
    const repo = String(row.id || "");
    return {
      id: repo,
      name: repo.split("/").filter(Boolean).pop() || repo,
      repo,
      engine: row.owned_by === "mlx-vlm" ? "vlm" : "lm",
      host: gateway.host,
      port: gateway.port,
      startedAt: (row.created ?? 0) * 1000,
    };
  });
}

export type StudioPrefs = {
  watchDirs: string[];
  flagsByModel: Record<string, FlagValues>;
};

export async function getPrefs(): Promise<StudioPrefs> {
  const res = await fetch("/v1/prefs");
  const body = (await parseJson(res)) as Partial<StudioPrefs>;
  return {
    watchDirs: Array.isArray(body.watchDirs) ? body.watchDirs.map(String) : [],
    flagsByModel: body.flagsByModel && typeof body.flagsByModel === "object" ? body.flagsByModel : {},
  };
}

export async function putPrefs(prefs: StudioPrefs): Promise<StudioPrefs> {
  const res = await fetch("/v1/prefs", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(prefs),
  });
  const body = (await parseJson(res)) as Partial<StudioPrefs>;
  return {
    watchDirs: Array.isArray(body.watchDirs) ? body.watchDirs.map(String) : prefs.watchDirs,
    flagsByModel: body.flagsByModel && typeof body.flagsByModel === "object" ? body.flagsByModel : prefs.flagsByModel,
  };
}

export async function postLoad(input: { engine: EngineKind; model: string; args?: string[] }) {
  const res = await fetch("/v1/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return parseJson(res);
}

export async function postUnload(model: string) {
  const res = await fetch("/v1/unload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  return parseJson(res);
}

export type ScanError = { dir: string; message: string };

export async function postScan(dirs: string[]): Promise<{ models: ModelRec[]; errors: ScanError[] }> {
  const res = await fetch("/v1/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dirs }),
  });
  const body = (await parseJson(res)) as { models?: ModelRec[]; errors?: ScanError[] };
  return { models: body.models ?? [], errors: body.errors ?? [] };
}
