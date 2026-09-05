import { create } from "zustand";
import { persist } from "zustand/middleware";
import { defaultFlags, flagArgs, flagsForModel, mergeFlags, type EngineKind, type FlagValues } from "./flags";
import { applyEngineOverrides, flagKey, loadTarget, mergeCatalog, migrateWatchDirs, type ModelRec } from "./models";
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
  type ProgressSnapshot,
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
  engineByModel: Partial<Record<string, EngineKind>>;
  lockedByModel: Record<string, boolean>;
  served: ServedRuntime[];
  gateway: GatewayInfo;
  tab: StudioTab;
  dirDraft: string;
  scanning: boolean;
  scanErrors: ScanError[];
  loadingIds: string[];
  pinKeys: string[];
  failed: Record<string, string>;
  hydrated: boolean;
  progress: ProgressSnapshot | null;
  addWatchDir: (dir: string) => Promise<void>;
  removeWatchDir: (dir: string) => Promise<void>;
  setDirDraft: (value: string) => void;
  selectModel: (id: string) => void;
  setFlag: (key: string, value: string | number | boolean) => void;
  resetFlags: () => void;
  setEngineOverride: (engine: EngineKind | null) => void;
  setModelLocked: (locked: boolean) => void;
  setTab: (tab: StudioTab) => void;
  setHydrated: () => void;
  applyPrefs: (prefs: {
    watchDirs?: string[];
    flagsByModel?: Record<string, FlagValues>;
    engineByModel?: Partial<Record<string, EngineKind>>;
    lockedByModel?: Record<string, boolean>;
  }) => void;
  persistPrefs: () => Promise<void>;
  scanWatchDirs: () => Promise<void>;
  startServe: (id?: string) => Promise<void>;
  stopServe: (id?: string) => Promise<void>;
  reloadServe: () => Promise<void>;
  syncServed: () => Promise<void>;
  selected: () => ModelRec | undefined;
  isLoaded: (id?: string | null) => boolean;
  setProgress: (progress: ProgressSnapshot | null) => void;
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

function withLoading(ids: string[], id: string): string[] {
  return ids.includes(id) ? ids : [...ids, id];
}

function withoutLoading(ids: string[], id: string): string[] {
  return ids.filter((item) => item !== id);
}

/** Catalog ids that should stay orange + sorted: currently live, plus any extras (loading). */
export function pinCatalogIds(
  models: ModelRec[],
  served: ServedRuntime[],
  extra: (string | null | undefined)[] = [],
): string[] {
  const pins = new Set<string>();
  for (const model of models) {
    if (modelIsLive(served, model)) pins.add(model.id);
  }
  for (const id of extra) {
    if (id) pins.add(id);
  }
  return [...pins];
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
      engineByModel: {},
      lockedByModel: {},
      served: [],
      gateway: DEFAULT_GATEWAY,
      tab: "settings",
      dirDraft: "",
      scanning: false,
      scanErrors: [],
      loadingIds: [],
      pinKeys: [],
      failed: {},
      hydrated: false,
      progress: null,
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
        set({ selectedId: id, flags });
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
      setEngineOverride: (engine) => {
        const model = get().selected();
        if (!model) return;
        const key = flagKey(model);
        const engineByModel = { ...get().engineByModel };
        if (engine) engineByModel[key] = engine;
        else delete engineByModel[key];
        const models = applyEngineOverrides(get().models, engineByModel);
        const selected = models.find((m) => m.id === model.id) ?? model;
        const flags = flagsForModel(selected, get().flagsByModel[key]);
        set({ engineByModel, models, flags });
        void get().persistPrefs();
      },
      setModelLocked: (locked) => {
        const model = get().selected();
        if (!model) return;
        const key = flagKey(model);
        const lockedByModel = { ...get().lockedByModel };
        if (locked) lockedByModel[key] = true;
        else delete lockedByModel[key];
        set({ lockedByModel });
        void get().persistPrefs();
      },
      setTab: (tab) => set({ tab }),
      setHydrated: () => set({ hydrated: true }),
      applyPrefs: (prefs) => {
        const fromServer = migrateWatchDirs(prefs.watchDirs);
        const watchDirs = fromServer.length ? fromServer : migrateWatchDirs(get().watchDirs);
        const flagsByModel = { ...get().flagsByModel, ...(prefs.flagsByModel ?? {}) };
        const engineByModel = { ...get().engineByModel, ...(prefs.engineByModel ?? {}) };
        const lockedByModel = { ...get().lockedByModel, ...(prefs.lockedByModel ?? {}) };
        const models = applyEngineOverrides(get().models, engineByModel);
        const model = models.find((m) => m.id === get().selectedId);
        const flags = flagsForModel(model, model ? flagsByModel[flagKey(model)] : get().flags);
        set({ watchDirs, flagsByModel, engineByModel, lockedByModel, models, flags });
      },
      persistPrefs: async () => {
        if (!get().hydrated) return;
        const { watchDirs, flagsByModel, engineByModel, lockedByModel } = get();
        try {
          await putPrefs({ watchDirs, flagsByModel, engineByModel, lockedByModel });
        } catch {
          /* preview without a gateway still keeps localStorage */
        }
      },
      selected: () => get().models.find((m) => m.id === get().selectedId),
      isLoaded: (id) => {
        const model = get().models.find((m) => m.id === id) ?? (id ? { id, repo: id } : null);
        return modelIsLive(get().served, model);
      },
      setProgress: (progress) => set({ progress }),
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
          const applied = applyEngineOverrides(next.models, get().engineByModel);
          const selected = applied.find((m) => m.id === next.selectedId);
          set({
            scanning: false,
            scanErrors: errors,
            flags: flagsForModel(selected, selected ? get().flagsByModel[flagKey(selected)] : undefined),
            ...next,
            models: applied,
          });
        } catch (err) {
          set({
            scanning: false,
            scanErrors: [{ dir: dirs.join(", "), message: err instanceof Error ? err.message : "Scan failed" }],
          });
        }
      },
      startServe: async (id) => {
        const model = id ? get().models.find((m) => m.id === id) : get().selected();
        if (!model) return;
        if (get().loadingIds.includes(model.id)) return;
        const flags =
          !id || id === get().selectedId
            ? get().flags
            : flagsForModel(model, get().flagsByModel[flagKey(model)]);
        const failed = { ...get().failed };
        delete failed[model.id];
        // Pin every in-flight load, not just the latest click. Finishing a
        // small model must not drop a large one that is still loading.
        const loadingIds = withLoading(get().loadingIds, model.id);
        const pinKeys = pinCatalogIds(get().models, get().served, loadingIds);
        set({ loadingIds, failed, pinKeys });
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
          const remaining = withoutLoading(get().loadingIds, model.id);
          set({
            served: attachFlags(listed, get().served, { model, flags }),
            gateway,
            loadingIds: remaining,
            failed: stillFailed,
            pinKeys: pinCatalogIds(get().models, listed, remaining),
          });
        } catch (err) {
          const remaining = withoutLoading(get().loadingIds, model.id);
          set({
            loadingIds: remaining,
            failed: { ...get().failed, [model.id]: err instanceof Error ? err.message : "Serve failed" },
            pinKeys: pinCatalogIds(get().models, get().served, remaining),
          });
          throw err;
        }
      },
      stopServe: async (id) => {
        const model = id ? get().models.find((m) => m.id === id) : get().selected();
        if (!model) return;
        await postUnload(loadTarget(model));
        const gateway = (await getHealth()).gateway;
        const listed = await listServed(gateway);
        const failed = { ...get().failed };
        delete failed[model.id];
        const loadingIds = withoutLoading(get().loadingIds, model.id);
        set({
          served: attachFlags(listed, get().served),
          gateway,
          failed,
          loadingIds,
          pinKeys: pinCatalogIds(get().models, listed, loadingIds),
        });
      },
      reloadServe: async () => {
        await get().startServe();
      },
      syncServed: async () => {
        const { gateway } = await getHealth();
        const listed = await listServed(gateway);
        const loadingIds = get().loadingIds;
        set({
          served: attachFlags(listed, get().served),
          gateway,
          pinKeys: pinCatalogIds(get().models, listed, loadingIds),
        });
      },
    }),
    {
      name: "edge-studio",
      skipHydration: true,
      partialize: (s) => ({
        watchDirs: s.watchDirs,
        selectedId: s.selectedId,
        flagsByModel: s.flagsByModel,
        engineByModel: s.engineByModel,
        lockedByModel: s.lockedByModel,
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
          engineByModel: p.engineByModel ?? current.engineByModel,
          lockedByModel: p.lockedByModel ?? current.lockedByModel,
          hydrated: false,
        };
      },
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        state.watchDirs = migrateWatchDirs(state.watchDirs);
        state.flagsByModel = state.flagsByModel ?? {};
        state.engineByModel = state.engineByModel ?? {};
        state.lockedByModel = state.lockedByModel ?? {};
        state.flags = mergeFlags(state.flags);
        const next = catalog([], [], state.selectedId);
        state.scanned = [];
        state.models = next.models;
        state.extraModels = [];
        state.selectedId = next.selectedId;
        state.scanning = false;
        state.scanErrors = [];
        state.loadingIds = [];
        state.pinKeys = [];
        state.failed = {};
        state.hydrated = false;
      },
    },
  ),
);
