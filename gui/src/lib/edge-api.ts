import { engineFromOwnedBy, type EngineKind, type FlagValues } from "./flags";
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

export type HostMemory = {
  used_bytes: number;
  total_bytes: number;
  ratio: number;
};

export type HostGpu = {
  percent: number;
  source: string;
};

export type HostSnapshot = {
  object: "edge.host";
  generated_at: number;
  memory: HostMemory;
  gpu: HostGpu | null;
};

export async function getHost(): Promise<HostSnapshot> {
  const res = await fetch("/v1/host");
  const body = (await parseJson(res)) as Partial<HostSnapshot>;
  const mem = body.memory || { used_bytes: 0, total_bytes: 0, ratio: 0 };
  const gpuRaw = body.gpu;
  const gpu =
    gpuRaw && typeof gpuRaw.percent === "number"
      ? { percent: Math.max(0, Math.min(100, gpuRaw.percent)), source: String(gpuRaw.source || "") }
      : null;
  return {
    object: "edge.host",
    generated_at: Number(body.generated_at || 0),
    memory: {
      used_bytes: Math.max(0, Number(mem.used_bytes || 0)),
      total_bytes: Math.max(0, Number(mem.total_bytes || 0)),
      ratio: Math.max(0, Math.min(1, Number(mem.ratio || 0))),
    },
    gpu,
  };
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
      engine: engineFromOwnedBy(row.owned_by),
      host: gateway.host,
      port: gateway.port,
      startedAt: (row.created ?? 0) * 1000,
    };
  });
}

export type StudioPrefs = {
  watchDirs: string[];
  flagsByModel: Record<string, FlagValues>;
  engineByModel: Partial<Record<string, EngineKind>>;
  lockedByModel: Record<string, boolean>;
};

function asLocked(raw: unknown): Record<string, boolean> {
  if (!raw || typeof raw !== "object") return {};
  const out: Record<string, boolean> = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (key && value === true) out[key] = true;
  }
  return out;
}

export async function getPrefs(): Promise<StudioPrefs> {
  const res = await fetch("/v1/prefs");
  const body = (await parseJson(res)) as Partial<StudioPrefs>;
  return {
    watchDirs: Array.isArray(body.watchDirs) ? body.watchDirs.map(String) : [],
    flagsByModel: body.flagsByModel && typeof body.flagsByModel === "object" ? body.flagsByModel : {},
    engineByModel:
      body.engineByModel && typeof body.engineByModel === "object" ? body.engineByModel : {},
    lockedByModel: asLocked(body.lockedByModel),
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
    engineByModel:
      body.engineByModel && typeof body.engineByModel === "object" ? body.engineByModel : prefs.engineByModel,
    lockedByModel: asLocked(body.lockedByModel ?? prefs.lockedByModel),
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

export async function postStop(model?: string) {
  const res = await fetch("/v1/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(model ? { model } : {}),
  });
  return parseJson(res);
}

export type PlaygroundMetrics = {
  ttft: number;
  gen: number;
  tokens: number;
  tps: number;
  model: string;
};

export type PlaygroundTurn = {
  role: "user" | "assistant";
  text: string;
  thinking?: string;
  metrics?: PlaygroundMetrics;
};

export async function getPlayground(): Promise<PlaygroundTurn[]> {
  const res = await fetch("/v1/playground");
  const body = (await parseJson(res)) as { turns?: PlaygroundTurn[] };
  return Array.isArray(body.turns) ? body.turns.filter((t) => t && (t.role === "user" || t.role === "assistant")) : [];
}

export async function putPlayground(turns: PlaygroundTurn[]) {
  const res = await fetch("/v1/playground", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ turns }),
  });
  return parseJson(res);
}

export async function clearPlayground() {
  const res = await fetch("/v1/playground", { method: "DELETE" });
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

export type HubQuant = { id: string; quant: string; downloads: number };

export type HubProgress = {
  repo: string;
  name: string;
  phase: "idle" | "downloading" | "paused" | "done" | "error" | "cancelled";
  bytes: number;
  total: number;
  ratio: number;
  pct: number | null;
  detail: string;
  error: string;
  path: string;
  token: boolean;
};

function asHubProgress(body: Partial<HubProgress> | null | undefined): HubProgress {
  return {
    repo: String(body?.repo || ""),
    name: String(body?.name || (body?.repo || "").split("/").pop() || ""),
    phase: (body?.phase as HubProgress["phase"]) || "idle",
    bytes: Number(body?.bytes || 0),
    total: Number(body?.total || 0),
    ratio: Number(body?.ratio || 0),
    pct: body?.pct == null || Number.isNaN(Number(body.pct)) ? null : Number(body.pct),
    detail: String(body?.detail || ""),
    error: String(body?.error || ""),
    path: String(body?.path || ""),
    token: Boolean(body?.token),
  };
}

export async function getHubStatus(): Promise<{ token: boolean; help: string }> {
  const res = await fetch("/v1/hub");
  const body = (await parseJson(res)) as { token?: boolean; help?: string };
  return { token: Boolean(body.token), help: String(body.help || "") };
}

export async function getHubProgress(): Promise<{ token: boolean; jobs: HubProgress[] }> {
  const res = await fetch("/v1/hub/progress");
  const body = (await parseJson(res)) as { token?: boolean; jobs?: Partial<HubProgress>[] };
  const jobs = Array.isArray(body.jobs) ? body.jobs.map((row) => asHubProgress(row)) : [];
  return { token: Boolean(body.token), jobs };
}

export async function postHubSearch(query: string): Promise<{
  repo: string;
  stem: string;
  token: boolean;
  results: HubQuant[];
}> {
  const res = await fetch("/v1/hub/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const body = (await parseJson(res)) as {
    repo?: string;
    stem?: string;
    token?: boolean;
    results?: HubQuant[];
  };
  return {
    repo: body.repo ?? "",
    stem: body.stem ?? "",
    token: Boolean(body.token),
    results: Array.isArray(body.results) ? body.results : [],
  };
}

export async function postHubDownload(repo: string): Promise<HubProgress> {
  const res = await fetch("/v1/hub/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
  return asHubProgress((await parseJson(res)) as Partial<HubProgress>);
}

export async function postHubPause(repo: string): Promise<HubProgress> {
  const res = await fetch("/v1/hub/pause", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
  return asHubProgress((await parseJson(res)) as Partial<HubProgress>);
}

export async function postHubResume(repo: string): Promise<HubProgress> {
  const res = await fetch("/v1/hub/resume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
  return asHubProgress((await parseJson(res)) as Partial<HubProgress>);
}

export async function postHubCancel(repo: string): Promise<HubProgress> {
  const res = await fetch("/v1/hub/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
  return asHubProgress((await parseJson(res)) as Partial<HubProgress>);
}

export async function postHubDelete(repo: string): Promise<{ repo: string; path: string }> {
  const res = await fetch("/v1/hub/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
  const body = (await parseJson(res)) as { repo?: string; path?: string };
  return { repo: body.repo ?? repo, path: body.path ?? "" };
}

export type ProgressPhase = "idle" | "loading" | "prefill" | "decode" | "done" | "error";
export type ProgressStatus = "ready" | "processing" | "complete" | "error";

export type PromptProgress = {
  processed_tokens: number;
  total_tokens: number | null;
  ratio: number;
  cached_tokens: number | null;
  started_at: number | null;
  updated_at: number | null;
  tokens_per_second: number | null;
};

export type GenerationProgress = {
  tokens: number;
  started_at: number | null;
  updated_at: number | null;
  tokens_per_second: number | null;
};

export type ModelProgress = {
  id: string;
  engine: EngineKind;
  phase: ProgressPhase;
  status: ProgressStatus;
  stream: boolean | null;
  progress: number;
  prompt: PromptProgress;
  generation: GenerationProgress;
  error: string | null;
};

export type ProgressSnapshot = {
  object: "edge.progress";
  version: 1;
  generated_at: number;
  active: boolean;
  progress: number;
  models: ModelProgress[];
};

export async function getProgress(model?: string): Promise<ProgressSnapshot> {
  const url = model ? `/v1/progress?model=${encodeURIComponent(model)}` : "/v1/progress";
  const res = await fetch(url);
  const body = (await parseJson(res)) as Partial<ProgressSnapshot>;
  return normalizeProgress(body);
}

/** EventSource for `GET /v1/progress/stream`. Same snapshot as `getProgress`. */
export function subscribeProgress(onSnap: (snap: ProgressSnapshot) => void): () => void {
  const es = new EventSource("/v1/progress/stream");
  es.onmessage = (ev) => {
    try {
      onSnap(normalizeProgress(JSON.parse(String(ev.data || "{}")) as Partial<ProgressSnapshot>));
    } catch {
      /* skip a torn frame */
    }
  };
  return () => es.close();
}

function progressRow(
  snap: ProgressSnapshot | null | undefined,
  model?: { id: string; repo: string; path?: string; name?: string } | null,
): ModelProgress | undefined {
  if (!snap || !model) return undefined;
  const needles = [model.repo, model.id, model.path, model.name].filter((n): n is string => Boolean(n));
  return snap.models.find((row) => needles.some((n) => sameModel(row.id, n)));
}

export function modelIsBusy(
  snap: ProgressSnapshot | null | undefined,
  model?: { id: string; repo: string; path?: string; name?: string } | null,
): boolean {
  const row = progressRow(snap, model);
  if (!row) return false;
  return row.status === "processing" && (row.phase === "prefill" || row.phase === "decode");
}

export function modelIsPrefill(
  snap: ProgressSnapshot | null | undefined,
  model?: { id: string; repo: string; path?: string; name?: string } | null,
): boolean {
  const row = progressRow(snap, model);
  return Boolean(row && row.status === "processing" && row.phase === "prefill");
}

export function modelGeneration(
  snap: ProgressSnapshot | null | undefined,
  model?: { id: string; repo: string; path?: string; name?: string } | null,
): { tokens: number; generating: boolean } {
  const row = progressRow(snap, model);
  if (!row) return { tokens: 0, generating: false };
  const generating = row.status === "processing" && (row.phase === "prefill" || row.phase === "decode");
  const tokens = Number(row.generation?.tokens || 0);
  return { tokens: Number.isFinite(tokens) ? tokens : 0, generating };
}

export function modelLoadProgress(
  snap: ProgressSnapshot | null | undefined,
  model?: { id: string; repo: string; path?: string; name?: string } | null,
): number | null {
  if (!snap || !model) return null;
  const needles = [model.repo, model.id, model.path, model.name].filter((n): n is string => Boolean(n));
  const row = snap.models.find((item) => needles.some((n) => sameModel(item.id, n)));
  if (!row || row.phase !== "loading") return null;
  return clamp01(Number(row.progress));
}

function normalizeProgress(body: Partial<ProgressSnapshot>): ProgressSnapshot {
  return {
    object: "edge.progress",
    version: 1,
    generated_at: Number(body.generated_at || Date.now() / 1000),
    active: Boolean(body.active),
    progress: clamp01(Number(body.progress)),
    models: Array.isArray(body.models)
      ? body.models.map((row) => ({
          ...(row as ModelProgress),
          progress: clamp01(Number((row as ModelProgress).progress)),
        }))
      : [],
  };
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

export type LogLevel = "info" | "warn" | "error" | "http" | "progress";
export type LogLine = {
  seq: number;
  ts: number;
  model: string;
  engine: string;
  level: LogLevel;
  text: string;
};

export async function getLogs(model?: string): Promise<{ seq: number; lines: LogLine[] }> {
  const url = model ? `/v1/logs?model=${encodeURIComponent(model)}` : "/v1/logs";
  const res = await fetch(url);
  const body = (await parseJson(res)) as { seq?: number; lines?: LogLine[] };
  return { seq: Number(body.seq || 0), lines: Array.isArray(body.lines) ? body.lines : [] };
}

export async function clearLogs() {
  await fetch("/v1/logs/clear", { method: "POST" }).catch(() => undefined);
}

export type TemplateInfo = {
  path: string;
  repo: string;
  bundled: boolean;
  source: string | null;
  chat_template: string | null;
  preset?: string | null;
  tried?: string[];
};

export async function fetchTemplate(input: { model: string; repo?: string }): Promise<TemplateInfo> {
  const res = await fetch("/v1/template", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return (await parseJson(res)) as TemplateInfo;
}

export function deltaContent(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const choices = (payload as { choices?: { delta?: { content?: unknown }; message?: { content?: unknown } }[] }).choices;
  const choice = choices?.[0];
  const delta = choice?.delta?.content;
  if (typeof delta === "string") return delta;
  const message = choice?.message?.content;
  return typeof message === "string" ? message : "";
}

function pickReason(a: string, b: string): string {
  if (a && b && (a === b || a.includes(b) || b.includes(a))) return a.length >= b.length ? a : b;
  return a || b;
}

export function deltaReasoning(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const choices = (
    payload as {
      choices?: {
        delta?: { reasoning_content?: unknown; reasoning?: unknown };
        message?: { reasoning_content?: unknown; reasoning?: unknown };
      }[];
    }
  ).choices;
  const choice = choices?.[0];
  const delta = choice?.delta;
  const fromDelta = pickReason(
    typeof delta?.reasoning_content === "string" ? delta.reasoning_content : "",
    typeof delta?.reasoning === "string" ? delta.reasoning : "",
  );
  if (fromDelta) return fromDelta;
  const message = choice?.message;
  return pickReason(
    typeof message?.reasoning_content === "string" ? message.reasoning_content : "",
    typeof message?.reasoning === "string" ? message.reasoning : "",
  );
}

export function deltaCompletionTokens(payload: unknown): number {
  if (!payload || typeof payload !== "object") return 0;
  const usage = (payload as { usage?: { completion_tokens?: unknown } }).usage;
  const n = Number(usage?.completion_tokens);
  return Number.isFinite(n) && n > 0 ? n : 0;
}
