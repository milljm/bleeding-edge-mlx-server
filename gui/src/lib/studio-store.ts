import { create } from "zustand";
import { persist } from "zustand/middleware";
import { defaultFlags, flagArgs, mergeFlags, type EngineKind, type FlagValues } from "./flags";
import { loadTarget, mergeCatalog, migrateWatchDirs, modelFromRepo, type ModelRec } from "./models";
import {
  DEFAULT_GATEWAY,
  getHealth,
  listServed,
  modelIsLive,
  postLoad,
  postScan,
  postUnload,
  type GatewayInfo,
  type ScanError,
  type ServedRuntime,
} from "./edge-api";

export type StudioTab = "settings" | "playground" | "endpoint";

type StudioState = {
  watchDirs: string[];
  extraModels: ModelRec[];
  scanned: ModelRec[];
  models: ModelRec[];
  selectedId: string | null;
  flags: FlagValues;
  served: ServedRuntime[];
  gateway: GatewayInfo;
  tab: StudioTab;
  dirDraft: string;
  modelDraft: string;
  modelEngine: EngineKind;
  scanning: boolean;
  scanErrors: ScanError[];
  addWatchDir: (dir: string) => Promise<void>;
  removeWatchDir: (dir: string) => Promise<void>;
  setDirDraft: (value: string) => void;
  addModel: (repo: string, engine: EngineKind) => boolean;
  setModelDraft: (value: string) => void;
  setModelEngine: (engine: EngineKind) => void;
  selectModel: (id: string) => void;
  setFlag: (key: string, value: string | number | boolean) => void;
  resetFlags: () => void;
  setTab: (tab: StudioTab) => void;
  scanWatchDirs: () => Promise<void>;
  startServe: (opts?: { stay?: boolean }) => Promise<void>;
  stopServe: () => Promise<void>;
  reloadServe: () => Promise<void>;
  syncServed: () => Promise<void>;
  selected: () => ModelRec | undefined;
  isLoaded: (id?: string | null) => boolean;
};

function catalog(scanned: ModelRec[], extra: ModelRec[], selectedId: string | null) {
  const models = mergeCatalog(scanned, extra);
  const still = models.some((m) => m.id === selectedId);
  return {
    models,
    extraModels: extra,
    scanned,
    selectedId: still ? selectedId : (models[0]?.id ?? null),
  };
}

function attachFlags(next: ServedRuntime[], prev: ServedRuntime[], justLoaded?: { model: ModelRec; flags: FlagValues }) {
  return next.map((row) => {
    if (justLoaded && modelIsLive([row], justLoaded.model)) {
      return { ...row, flags: justLoaded.flags };
    }
    const old = prev.find((p) => modelIsLive([row], { id: p.id, repo: p.repo }));
    return { ...row, flags: old?.flags };
  });
}

export const useStudio = create<StudioState>()(
  persist(
    (set, get) => ({
      watchDirs: [],
      extraModels: [],
      scanned: [],
      models: [],
      selectedId: null,
      flags: defaultFlags(),
      served: [],
      gateway: DEFAULT_GATEWAY,
      tab: "settings",
      dirDraft: "",
      modelDraft: "",
      modelEngine: "lm",
      scanning: false,
      scanErrors: [],
      addWatchDir: async (dir) => {
        const trimmed = dir.trim();
        if (!trimmed) return;
        const watchDirs = get().watchDirs.includes(trimmed) ? get().watchDirs : [...get().watchDirs, trimmed];
        set({ watchDirs, dirDraft: "" });
        await get().scanWatchDirs();
      },
      removeWatchDir: async (dir) => {
        const watchDirs = get().watchDirs.filter((d) => d !== dir);
        const extraModels = get().extraModels.filter((m) => m.watchDir !== dir);
        set({ watchDirs, extraModels });
        await get().scanWatchDirs();
      },
      setDirDraft: (dirDraft) => set({ dirDraft }),
      addModel: (repo, engine) => {
        const watchDir = get().watchDirs[0] ?? "";
        const rec = modelFromRepo(repo, engine, watchDir);
        if (!rec) return false;
        if (get().models.some((m) => m.id === rec.id || m.repo === rec.repo)) {
          const existing = get().models.find((m) => m.id === rec.id || m.repo === rec.repo);
          set({ selectedId: existing?.id ?? rec.id, modelDraft: "", tab: "settings" });
          return true;
        }
        const extraModels = [...get().extraModels, rec];
        set({
          modelDraft: "",
          tab: "settings",
          ...catalog(get().scanned, extraModels, rec.id),
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
      scanWatchDirs: async () => {
        const dirs = get().watchDirs;
        if (!dirs.length) {
          set({
            scanning: false,
            scanErrors: [],
            ...catalog([], get().extraModels, get().selectedId),
          });
          return;
        }
        set({ scanning: true });
        try {
          const { models, errors } = await postScan(dirs);
          set({
            scanning: false,
            scanErrors: errors,
            ...catalog(models, get().extraModels, get().selectedId),
          });
        } catch (err) {
          set({
            scanning: false,
            scanErrors: [{ dir: dirs.join(", "), message: err instanceof Error ? err.message : "Scan failed" }],
          });
        }
      },
      startServe: async (opts) => {
        const model = get().selected();
        if (!model) return;
        const flags = get().flags;
        await postLoad({
          engine: model.engine,
          model: loadTarget(model),
          args: flagArgs(model.engine, flags, ["host", "port"]),
        });
        const gateway = (await getHealth()).gateway;
        const listed = await listServed(gateway);
        set({
          served: attachFlags(listed, get().served, { model, flags }),
          gateway,
          tab: opts?.stay ? get().tab : "playground",
        });
      },
      stopServe: async () => {
        const model = get().selected();
        if (!model) return;
        await postUnload(loadTarget(model));
        const gateway = (await getHealth()).gateway;
        const listed = await listServed(gateway);
        set({ served: attachFlags(listed, get().served), gateway });
      },
      reloadServe: async () => {
        await get().startServe({ stay: true });
      },
      syncServed: async () => {
        const { gateway } = await getHealth();
        const listed = await listServed(gateway);
        set({ served: attachFlags(listed, get().served), gateway });
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
        state.watchDirs = migrateWatchDirs(state.watchDirs);
        const next = catalog([], state.extraModels ?? [], state.selectedId);
        state.scanned = [];
        state.models = next.models;
        state.extraModels = next.extraModels;
        state.selectedId = next.selectedId;
        state.scanning = false;
        state.scanErrors = [];
      },
    },
  ),
);
