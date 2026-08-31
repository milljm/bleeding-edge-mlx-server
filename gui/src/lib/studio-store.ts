import { create } from "zustand";
import { persist } from "zustand/middleware";
import { defaultFlags, flagArgs, mergeFlags, type EngineKind, type FlagValues } from "./flags";
import {
  DEFAULT_WATCH,
  modelFromRepo,
  modelsForDirs,
  type ModelRec,
} from "./models";
import {
  DEFAULT_GATEWAY,
  getHealth,
  listServed,
  modelIsLive,
  postLoad,
  postUnload,
  type GatewayInfo,
  type ServedRuntime,
} from "./edge-api";

export type StudioTab = "settings" | "playground" | "endpoint";

type StudioState = {
  watchDirs: string[];
  extraModels: ModelRec[];
  models: ModelRec[];
  selectedId: string | null;
  flags: FlagValues;
  served: ServedRuntime[];
  gateway: GatewayInfo;
  tab: StudioTab;
  dirDraft: string;
  modelDraft: string;
  modelEngine: EngineKind;
  addWatchDir: (dir: string) => void;
  removeWatchDir: (dir: string) => void;
  setDirDraft: (value: string) => void;
  addModel: (repo: string, engine: EngineKind) => boolean;
  setModelDraft: (value: string) => void;
  setModelEngine: (engine: EngineKind) => void;
  selectModel: (id: string) => void;
  setFlag: (key: string, value: string | number | boolean) => void;
  resetFlags: () => void;
  setTab: (tab: StudioTab) => void;
  startServe: () => Promise<void>;
  stopServe: () => Promise<void>;
  syncServed: () => Promise<void>;
  selected: () => ModelRec | undefined;
  isLoaded: (id?: string | null) => boolean;
};

function rebuild(dirs: string[], extra: ModelRec[], selectedId: string | null) {
  const models = modelsForDirs(dirs, extra);
  const still = models.some((m) => m.id === selectedId);
  return {
    models,
    extraModels: extra,
    selectedId: still ? selectedId : (models[0]?.id ?? null),
  };
}

export const useStudio = create<StudioState>()(
  persist(
    (set, get) => ({
      watchDirs: [DEFAULT_WATCH],
      extraModels: [],
      models: modelsForDirs([DEFAULT_WATCH], []),
      selectedId: modelsForDirs([DEFAULT_WATCH], [])[0]?.id ?? null,
      flags: defaultFlags(),
      served: [],
      gateway: DEFAULT_GATEWAY,
      tab: "settings",
      dirDraft: "",
      modelDraft: "",
      modelEngine: "lm",
      addWatchDir: (dir) => {
        const trimmed = dir.trim();
        if (!trimmed) return;
        const watchDirs = get().watchDirs.includes(trimmed)
          ? get().watchDirs
          : [...get().watchDirs, trimmed];
        set({
          watchDirs,
          dirDraft: "",
          ...rebuild(watchDirs, get().extraModels, get().selectedId),
        });
      },
      removeWatchDir: (dir) => {
        const watchDirs = get().watchDirs.filter((d) => d !== dir);
        const extraModels = get().extraModels.filter((m) => m.watchDir !== dir);
        set({ watchDirs, ...rebuild(watchDirs, extraModels, get().selectedId) });
      },
      setDirDraft: (dirDraft) => set({ dirDraft }),
      addModel: (repo, engine) => {
        const watchDir = get().watchDirs[0] ?? DEFAULT_WATCH;
        const rec = modelFromRepo(repo, engine, watchDir);
        if (!rec) return false;
        if (get().models.some((m) => m.id === rec.id || m.repo === rec.repo)) {
          set({ selectedId: rec.id, modelDraft: "", tab: "settings" });
          return true;
        }
        const extraModels = [...get().extraModels, rec];
        set({
          modelDraft: "",
          ...rebuild(get().watchDirs, extraModels, rec.id),
          tab: "settings",
        });
        return true;
      },
      setModelDraft: (modelDraft) => set({ modelDraft }),
      setModelEngine: (modelEngine) => set({ modelEngine }),
      selectModel: (id) => set({ selectedId: id, tab: "settings" }),
      setFlag: (key, value) => set({ flags: { ...get().flags, [key]: value } }),
      resetFlags: () => set({ flags: defaultFlags() }),
      setTab: (tab) => set({ tab }),
      selected: () => get().models.find((m) => m.id === get().selectedId),
      isLoaded: (id) => {
        const model = get().models.find((m) => m.id === id) ?? (id ? { id, repo: id } : null);
        return modelIsLive(get().served, model);
      },
      startServe: async () => {
        const model = get().selected();
        if (!model) return;
        const flags = get().flags;
        await postLoad({
          engine: model.engine,
          model: model.path || model.repo,
          args: flagArgs(model.engine, flags, ["host", "port"]),
        });
        const gateway = (await getHealth()).gateway;
        const served = await listServed(gateway);
        set({ served, gateway, tab: "playground" });
      },
      stopServe: async () => {
        const model = get().selected();
        if (!model) return;
        await postUnload(model.repo);
        const gateway = (await getHealth()).gateway;
        const served = await listServed(gateway);
        set({ served, gateway });
      },
      syncServed: async () => {
        const { gateway } = await getHealth();
        const served = await listServed(gateway);
        set({ served, gateway });
      },
    }),
    {
      name: "edge-studio",
      skipHydration: true,
      partialize: (s) => ({
        watchDirs: s.watchDirs,
        extraModels: s.extraModels,
        selectedId: s.selectedId,
        flags: s.flags,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        state.flags = mergeFlags(state.flags);
        const next = rebuild(state.watchDirs ?? [DEFAULT_WATCH], state.extraModels ?? [], state.selectedId);
        state.models = next.models;
        state.extraModels = next.extraModels;
        state.selectedId = next.selectedId;
      },
    },
  ),
);
