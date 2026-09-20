export type EngineKind = "lm" | "vlm" | "embed" | "tts" | "stt" | "rerank" | "image";
export type FlagType = "number" | "text" | "bool" | "select";
export type FlagGroup = "server" | "sampling" | "thinking" | "template" | "replace";

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
  /** Edge-only: stored in prefs, never passed to mlx-lm / mlx-vlm. */
  edge?: boolean;
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
    help: "Websites allowed to call this model. * allows all.",
    type: "text",
    engines: ["lm"],
    group: "server",
    default: "*",
  },
  {
    key: "temp",
    flag: "--temp",
    label: "Temperature",
    help: "Higher values are more creative, 0 is deterministic.",
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
    help: "Limits word choice; 1 turns it off.",
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
    help: "Limits word choice to the k most likely words; 0 turns it off.",
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
    help: "Drops unlikely words; 0 turns it off.",
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
    help: "Upper limit; filled from the model's context window when known.",
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
    help: "Turn on thinking blocks by default.",
    type: "bool",
    engines: ["vlm"],
    group: "thinking",
    default: false,
  },
  {
    key: "streamReasonToResponse",
    flag: "--stream-reason-to-response",
    label: "Reason as response",
    help: "Show the model's thinking as the reply itself, for models that only answer on their reasoning channel.",
    type: "bool",
    engines: ["lm", "vlm"],
    group: "thinking",
    edge: true,
    default: false,
  },
  {
    key: "replaceReasoning",
    flag: "--replace-reasoning",
    label: "Reasoning replace",
    help: "Replace text inside the model's thinking.",
    type: "text",
    engines: ["lm", "vlm"],
    group: "replace",
    edge: true,
    default: "",
  },
  {
    key: "replaceResponse",
    flag: "--replace-response",
    label: "Response replace",
    help: "Replace text inside the reply.",
    type: "text",
    engines: ["lm", "vlm"],
    group: "replace",
    edge: true,
    default: "",
  },
  {
    key: "adapterPath",
    flag: "--adapter-path",
    label: "Adapter path",
    help: "LoRA adapter folder. Leave empty if unused.",
    type: "text",
    engines: ["lm", "vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "draftModel",
    flag: "--draft-model",
    label: "Draft model",
    help: "A smaller companion model that speeds up generation.",
    type: "text",
    engines: ["lm", "vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "numDraftTokens",
    flag: "--num-draft-tokens",
    label: "Draft tokens",
    help: "Draft tokens proposed per step.",
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
    help: "Custom chat template. Pull one from Hugging Face if the model shipped none.",
    type: "text",
    engines: ["lm"],
    group: "template",
    default: "",
  },
  {
    key: "chatTemplateArgs",
    flag: "--chat-template-args",
    label: "Chat template args",
    help: "Extra template options as JSON.",
    type: "text",
    engines: ["lm"],
    advanced: true,
    default: "",
  },
  {
    key: "useDefaultChatTemplate",
    flag: "--use-default-chat-template",
    label: "Default chat template",
    help: "Use the built-in template instead of a custom one.",
    type: "bool",
    engines: ["lm"],
    group: "template",
    default: false,
  },
  {
    key: "trustRemoteCode",
    flag: "--trust-remote-code",
    label: "Trust remote code",
    help: "Allow custom code bundled with the model.",
    type: "bool",
    engines: ["lm", "vlm", "embed", "tts", "stt", "rerank", "image"],
    advanced: true,
    default: false,
  },
  {
    key: "logLevel",
    flag: "--log-level",
    label: "Log level",
    help: "How much detail the log shows. DEBUG logs every generated token.",
    type: "select",
    engines: ["lm", "vlm", "embed", "tts", "stt", "rerank", "image"],
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
    help: "How many replies can generate at once.",
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
    help: "How many prompts can process at once.",
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
    help: "How much of the prompt is read per step.",
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
    help: "How many conversations to keep cached.",
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
    help: "Use pipeline parallelism across GPUs.",
    type: "bool",
    engines: ["lm"],
    advanced: true,
    default: false,
  },
  {
    key: "visionCacheSize",
    flag: "--vision-cache-size",
    label: "Vision cache",
    help: "How many images to keep cached.",
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
    help: "Cap on thinking tokens.",
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
    label: "Cache bits",
    help: "Compress the prompt cache. Empty is off.",
    type: "text",
    engines: ["vlm"],
    advanced: true,
    default: "",
  },
  {
    key: "kvGroupSize",
    flag: "--kv-group-size",
    label: "KV group size",
    help: "Compression group size.",
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
    label: "Max cache size",
    help: "Prompt cache size cap. Empty is unlimited.",
    type: "text",
    engines: ["vlm"],
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
    if (def.edge) continue;
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
  if ((model?.engine === "lm" || model?.engine === "vlm") && typeof context === "number" && context > 0) {
    const cap = FLAG_DEFS.find((d) => d.key === "maxTokens");
    const max = typeof cap?.max === "number" ? cap.max : context;
    const min = typeof cap?.min === "number" ? cap.min : 16;
    base.maxTokens = Math.min(max, Math.max(min, context));
  }
  return mergeFlags({ ...base, ...(saved ?? {}) });
}

export function engineFromOwnedBy(value?: string | null): EngineKind {
  if (value === "mlx-vlm") return "vlm";
  if (value === "mlx-embed") return "embed";
  if (value === "mlx-tts") return "tts";
  if (value === "mlx-stt") return "stt";
  if (value === "mlx-rerank") return "rerank";
  if (value === "mlx-image") return "image";
  return "lm";
}
