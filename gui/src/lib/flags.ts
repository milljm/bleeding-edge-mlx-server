export type EngineKind = "lm" | "vlm" | "embed";
export type FlagType = "number" | "text" | "bool" | "select";
export type FlagGroup = "server" | "sampling" | "thinking" | "template";

export type FlagDef = {
  key: string;
  flag: string;
  label: string;
  help: string;
  type: FlagType;
  engines: EngineKind[];
  group?: FlagGroup;
  advanced?: boolean;
  always?: boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  default: string | number | boolean;
};

export const FLAG_DEFS: FlagDef[] = [
  {
    key: "allowedOrigins",
    flag: "--allowed-origins",
    label: "Allowed origins",
    help: "CORS allow-list. * permits every origin.",
    type: "text",
    engines: ["lm"],
    group: "server",
    default: "*",
  },
  {
    key: "temp",
    flag: "--temp",
    label: "Temperature",
    help: "Sampling temperature. 0 is greedy. A request can override this.",
    type: "number",
    engines: ["lm"],
    group: "sampling",
    min: 0,
    max: 2,
    step: 0.05,
    default: 0,
  },
  {
    key: "topP",
    flag: "--top-p",
    label: "Top-p",
    help: "Nucleus sampling. 1 disables. A request can override this.",
    type: "number",
    engines: ["lm"],
    group: "sampling",
    min: 0,
    max: 1,
    step: 0.01,
    default: 1,
  },
  {
    key: "topK",
    flag: "--top-k",
    label: "Top-k",
    help: "Top-k sampling. 0 disables. A request can override this.",
    type: "number",
    engines: ["lm"],
    group: "sampling",
    min: 0,
    max: 200,
    step: 1,
    default: 0,
  },
  {
    key: "minP",
    flag: "--min-p",
    label: "Min-p",
    help: "Min-p sampling. 0 disables. A request can override this.",
    type: "number",
    engines: ["lm"],
    group: "sampling",
    min: 0,
    max: 1,
    step: 0.01,
    default: 0,
  },
  {
    key: "maxTokens",
    flag: "--max-tokens",
    label: "Max context tokens",
    help: "Generation cap. Filled from the model's context window when known.",
    type: "number",
    engines: ["lm", "vlm"],
    group: "sampling",
    min: 16,
    max: 262144,
    step: 16,
    default: 512,
  },
  {
    key: "enableThinking",
    flag: "--enable-thinking",
    label: "Thinking",
    help: "Enable thinking blocks by default (mlx-vlm).",
    type: "bool",
    engines: ["vlm"],
    group: "thinking",
    default: false,
  },
  {
    key: "adapterPath",
    flag: "--adapter-path",
    label: "Adapter path",
    help: "Optional LoRA / adapter weights.",
    type: "text",
    engines: ["lm", "vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "draftModel",
    flag: "--draft-model",
    label: "Draft model",
    help: "Speculative decoding drafter.",
    type: "text",
    engines: ["lm", "vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "numDraftTokens",
    flag: "--num-draft-tokens",
    label: "Draft tokens",
    help: "Tokens proposed by the drafter.",
    type: "number",
    engines: ["lm"],
    advanced: true,
    min: 1,
    max: 16,
    step: 1,
    default: 3,
  },
  {
    key: "chatTemplate",
    flag: "--chat-template",
    label: "Chat template",
    help: "Jinja chat template. Pull from Hugging Face if the checkpoint did not ship one (MiniMax / gpt-oss). mlx-vlm uses the same override.",
    type: "text",
    engines: ["lm", "vlm"],
    group: "template",
    default: "",
  },
  {
    key: "chatTemplateArgs",
    flag: "--chat-template-args",
    label: "Chat template args",
    help: "JSON object passed to apply_chat_template.",
    type: "text",
    engines: ["lm", "vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "useDefaultChatTemplate",
    flag: "--use-default-chat-template",
    label: "Default chat template",
    help: "Force the tokenizer default template instead of a custom one.",
    type: "bool",
    engines: ["lm", "vlm"],
    group: "template",
    default: false,
  },
  {
    key: "trustRemoteCode",
    flag: "--trust-remote-code",
    label: "Trust remote code",
    help: "Allow custom tokenizer / model code from Hub.",
    type: "bool",
    engines: ["lm", "vlm", "embed"],
    advanced: true,
    default: false,
  },
  {
    key: "logLevel",
    flag: "--log-level",
    label: "Log level",
    help: "Server logging verbosity.",
    type: "select",
    engines: ["lm", "vlm", "embed"],
    advanced: true,
    options: [
      { value: "DEBUG", label: "DEBUG" },
      { value: "INFO", label: "INFO" },
      { value: "WARNING", label: "WARNING" },
      { value: "ERROR", label: "ERROR" },
    ],
    default: "INFO",
  },
  {
    key: "decodeConcurrency",
    flag: "--decode-concurrency",
    label: "Decode concurrency",
    help: "Parallel decode slots.",
    type: "number",
    engines: ["lm"],
    advanced: true,
    min: 1,
    max: 128,
    step: 1,
    default: 32,
  },
  {
    key: "promptConcurrency",
    flag: "--prompt-concurrency",
    label: "Prompt concurrency",
    help: "Parallel prompt processing.",
    type: "number",
    engines: ["lm"],
    advanced: true,
    min: 1,
    max: 64,
    step: 1,
    default: 8,
  },
  {
    key: "prefillStepSize",
    flag: "--prefill-step-size",
    label: "Prefill step",
    help: "Tokens per prefill step.",
    type: "number",
    engines: ["lm", "vlm"],
    advanced: true,
    min: 64,
    max: 8192,
    step: 64,
    default: 2048,
  },
  {
    key: "promptCacheSize",
    flag: "--prompt-cache-size",
    label: "Prompt cache",
    help: "Distinct KV caches to keep.",
    type: "number",
    engines: ["lm"],
    advanced: true,
    min: 0,
    max: 64,
    step: 1,
    default: 10,
  },
  {
    key: "pipeline",
    flag: "--pipeline",
    label: "Pipeline parallel",
    help: "Pipeline instead of tensor parallel.",
    type: "bool",
    engines: ["lm"],
    advanced: true,
    default: false,
  },
  {
    key: "visionCacheSize",
    flag: "--vision-cache-size",
    label: "Vision cache",
    help: "Cached vision feature slots.",
    type: "number",
    engines: ["vlm"],
    advanced: true,
    min: 0,
    max: 128,
    step: 1,
    default: 20,
  },
  {
    key: "thinkingBudget",
    flag: "--thinking-budget",
    label: "Thinking budget",
    help: "Max tokens inside a thinking block.",
    type: "number",
    engines: ["vlm"],
    group: "thinking",
    advanced: true,
    min: 0,
    max: 8192,
    step: 32,
    default: 0,
  },
  {
    key: "kvBits",
    flag: "--kv-bits",
    label: "KV bits",
    help: "KV cache quantization bits. Empty is off.",
    type: "text",
    engines: ["vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "kvGroupSize",
    flag: "--kv-group-size",
    label: "KV group size",
    help: "Group size for uniform KV quantization.",
    type: "number",
    engines: ["vlm"],
    advanced: true,
    min: 8,
    max: 128,
    step: 8,
    default: 64,
  },
  {
    key: "maxKvSize",
    flag: "--max-kv-size",
    label: "Max KV size",
    help: "KV cache cap in tokens. Empty is unbounded.",
    type: "text",
    engines: ["vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "imageModel",
    flag: "--image-model",
    label: "Image model",
    help: "Preload an image generation model.",
    type: "text",
    engines: ["vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "ttsModel",
    flag: "--tts-model",
    label: "TTS model",
    help: "Preload a speech model.",
    type: "text",
    engines: ["vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "sttModel",
    flag: "--stt-model",
    label: "STT model",
    help: "Preload a transcription model.",
    type: "text",
    engines: ["vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "embeddingModel",
    flag: "--embedding-model",
    label: "Embedding model",
    help: "Preload embeddings on a VLM server (dedicated embed models use engine embed instead).",
    type: "text",
    engines: ["vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "rerankerModel",
    flag: "--reranker-model",
    label: "Reranker model",
    help: "Preload a reranker.",
    type: "text",
    engines: ["vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "apiKey",
    flag: "--api-key",
    label: "API key",
    help: "Optional bearer token for the server.",
    type: "text",
    engines: ["vlm", "embed"],
    advanced: true,
    default: "",
  },
];

export type FlagValues = Record<string, string | number | boolean>;

export function defaultFlags(): FlagValues {
  const out: FlagValues = {};
  for (const def of FLAG_DEFS) out[def.key] = def.default;
  return out;
}

export function mergeFlags(partial?: FlagValues | null): FlagValues {
  return { ...defaultFlags(), ...(partial ?? {}) };
}

export function flagsFor(engine: EngineKind, advanced: boolean) {
  return FLAG_DEFS.filter(
    (d) => d.engines.includes(engine) && (advanced ? Boolean(d.advanced) : !d.advanced),
  );
}

function isUnset(def: FlagDef, value: string | number | boolean) {
  if (def.always) return false;
  if (def.type === "text") return String(value).trim() === "" || value === def.default;
  if (def.type === "bool") return !value;
  if (def.type === "select") return value === def.default;
  return value === def.default;
}

export function flagArgs(engine: EngineKind, values: FlagValues, omit: string[] = []): string[] {
  const args: string[] = [];
  const skip = new Set(omit);
  for (const def of FLAG_DEFS) {
    if (!def.engines.includes(engine)) continue;
    if (skip.has(def.key)) continue;
    const value = values[def.key] ?? def.default;
    if (def.type === "bool") {
      if (value) args.push(def.flag);
      continue;
    }
    if (isUnset(def, value)) continue;
    args.push(def.flag, String(value));
  }
  return args;
}

export function flagsDirty(engine: EngineKind, current: FlagValues, loaded?: FlagValues | null) {
  if (!loaded) return false;
  return JSON.stringify(flagArgs(engine, current, ["host", "port"])) !== JSON.stringify(flagArgs(engine, loaded, ["host", "port"]));
}

export function flagsForModel(
  model?: { engine: EngineKind; context?: number | null } | null,
  saved?: FlagValues | null,
): FlagValues {
  const base = defaultFlags();
  const context = model?.context;
  if (model?.engine !== "embed" && typeof context === "number" && context > 0) {
    const cap = FLAG_DEFS.find((d) => d.key === "maxTokens");
    const max = typeof cap?.max === "number" ? cap.max : context;
    const min = typeof cap?.min === "number" ? cap.min : 16;
    base.maxTokens = Math.min(max, Math.max(min, context));
  }
  return mergeFlags({ ...base, ...(saved ?? {}) });
}

export function ownedBy(engine: EngineKind) {
  if (engine === "vlm") return "mlx-vlm";
  if (engine === "embed") return "mlx-embed";
  return "mlx-lm";
}

export function engineFromOwnedBy(value?: string | null): EngineKind {
  if (value === "mlx-vlm") return "vlm";
  if (value === "mlx-embed") return "embed";
  return "lm";
}
