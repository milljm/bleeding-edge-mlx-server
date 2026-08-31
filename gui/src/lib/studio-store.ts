import { create } from "zustand";
import { persist } from "zustand/middleware";
import { defaultFlags, flagArgs, flagsForModel, mergeFlags, type FlagValues } from "./flags";
import { flagKey, loadTarget, mergeCatalog, migrateWatchDirs, type ModelRec } from "./models";
import {
  DEFAULT_GATEWAY,
  getHealth,
  getPrefs,
  listServed,
  modelIsLive,
  postLoad,
  postScan,
  postUnload,
  putPrefs,
  type GatewayInfo,
  type ScanError,
  type ServedRuntime,
} from "./edge-api";

export type StudioTab = "settings" | "playground" | "endpoint" | "logging";

type StudioState = {
  watchDirs: string[];
  extraModels: ModelRec[];
  scanned: ModelRec[];
  models: ModelRec[];
  selectedId: string | null;
  flags: FlagValues;
  flagsByModel: Record<string, FlagValues>;
  served: ServedRuntime[];
  gateway: GatewayInfo;
  tab: StudioTab;
  dirDraft: string;
  scanning: boolean;
  scanErrors: ScanError[];
  loadingId: string | null;
  failed: Record<string, string>;
  hydrated: boolean;
  addWatchDir: (dir: string) => Promise<void>;
  removeWatchDir: (dir: string) => Promise<void>;
  setDirDraft: (value: string) => void;
  selectModel: (id: string) => void;
  setFlag: (key: string, value: string | number | boolean) => void;
  resetFlags: () => void;
  setTab: (tab: StudioTab) => void;
  setHydrated: () => void;
  applyPrefs: (prefs: { watchDirs?: string[]; flagsByModel?: Record<string, FlagValues> }) => void;
  persistPrefs: () => Promise<void>;
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
      flagsByModel: {},
      served: [],
      gateway: DEFAULT_GATEWAY,
      tab: "settings",
      dirDraft: "",
      scanning: false,
      scanErrors: [],
      loadingId: null,
      failed: {},
      hydrated: false,
      addWatchDir: async (dir) => {
        const trimmed = dir.trim();
        if (!trimmed) return;
        const watchDirs = get().watchDirs.includes(trimmed) ? get().watchDirs : [...get().watchDirs, trimmed];
        set({ watchDirs, dirDraft: "" });
        await get().persistPrefs();
        await get().scanWatchDirs();
      },
      removeWatchDir: async (dir) => {
        const watchDirs = get().watchDirs.filter((d) => d !== dir);
        set({ watchDirs, extraModels: [] });
        await get().persistPrefs();
        await get().scanWatchDirs();
      },
      setDirDraft: (dirDraft) => set({ dirDraft }),
      selectModel: (id) => {
        const model = get().models.find((m) => m.id === id);
        const flags = flagsForModel(model, model ? get().flagsByModel[flagKey(model)] : undefined);
        set({ selectedId: id, flags, tab: "settings" });
      },
      setFlag: (key, value) => {
        const model = get().selected();
        const flags = { ...get().flags, [key]: value };
        const flagsByModel = { ...get().flagsByModel };
        if (model) flagsByModel[flagKey(model)] = flags;
        set({ flags, flagsByModel });
        void get().persistPrefs();
      },
      resetFlags: () => {
        const model = get().selected();
        const flags = flagsForModel(model, null);
        const flagsByModel = { ...get().flagsByModel };
        if (model) flagsByModel[flagKey(model)] = flags;
        set({ flags, flagsByModel });
        void get().persistPrefs();
      },
      setTab: (tab) => set({ tab }),
      setHydrated: () => set({ hydrated: true }),
      applyPrefs: (prefs) => {
        const fromServer = migrateWatchDirs(prefs.watchDirs);
        const watchDirs = fromServer.length ? fromServer : migrateWatchDirs(get().watchDirs);
        const flagsByModel = { ...get().flagsByModel, ...(prefs.flagsByModel ?? {}) };
        const model = get().models.find((m) => m.id === get().selectedId);
        const flags = flagsForModel(model, model ? flagsByModel[flagKey(model)] : get().flags);
        set({ watchDirs, flagsByModel, flags });
      },
      persistPrefs: async () => {
        if (!get().hydrated) return;
        const { watchDirs, flagsByModel } = get();
        try {
          await putPrefs({ watchDirs, flagsByModel });
        } catch {
          /* preview without a gateway still keeps localStorage */
        }
      },
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
            ...catalog([], [], get().selectedId),
          });
          return;
        }
        set({ scanning: true });
        try {
          const { models, errors } = await postScan(dirs);
          const next = catalog(models, [], get().selectedId);
          const selected = next.models.find((m) => m.id === next.selectedId);
          set({
            scanning: false,
            scanErrors: errors,
            flags: flagsForModel(selected, selected ? get().flagsByModel[flagKey(selected)] : undefined),
            ...next,
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
        const failed = { ...get().failed };
        delete failed[model.id];
        set({ loadingId: model.id, failed });
        try {
          await postLoad({
            engine: model.engine,
            model: loadTarget(model),
            args: flagArgs(model.engine, flags, ["host", "port"]),
          });
          const gateway = (await getHealth()).gateway;
          const listed = await listServed(gateway);
          const stillFailed = { ...get().failed };
          delete stillFailed[model.id];
          set({
            served: attachFlags(listed, get().served, { model, flags }),
            gateway,
            loadingId: null,
            failed: stillFailed,
            tab: opts?.stay ? get().tab : model.engine === "embed" ? "endpoint" : "playground",
          });
        } catch (err) {
          set({
            loadingId: null,
            failed: { ...get().failed, [model.id]: err instanceof Error ? err.message : "Serve failed" },
          });
          throw err;
        }
      },
      stopServe: async () => {
        const model = get().selected();
        if (!model) return;
        await postUnload(loadTarget(model));
        const gateway = (await getHealth()).gateway;
        const listed = await listServed(gateway);
        const failed = { ...get().failed };
        delete failed[model.id];
        set({ served: attachFlags(listed, get().served), gateway, failed });
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
        selectedId: s.selectedId,
        flagsByModel: s.flagsByModel,
        flags: s.flags,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<StudioState>;
        const persistedDirs = migrateWatchDirs(p.watchDirs);
        return {
          ...current,
          ...p,
          watchDirs: persistedDirs.length ? persistedDirs : current.watchDirs,
          flagsByModel: p.flagsByModel ?? current.flagsByModel,
          hydrated: false,
        };
      },
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        state.watchDirs = migrateWatchDirs(state.watchDirs);
        state.flagsByModel = state.flagsByModel ?? {};
        state.flags = mergeFlags(state.flags);
        const next = catalog([], [], state.selectedId);
        state.scanned = [];
        state.models = next.models;
        state.extraModels = [];
        state.selectedId = next.selectedId;
        state.scanning = false;
        state.scanErrors = [];
        state.loadingId = null;
        state.failed = {};
        state.hydrated = false;
      },
    },
  ),
);
