import { useEffect, useState } from "preact/hooks";
import {
  api,
  openStream,
  type Activation,
  type DatasetInfo,
  type EpochRecord,
  type LayerSpec,
  type LossName,
  type OptimizerName,
  type Prediction,
  type RunConfig,
  type RunSummary,
  type Snapshot,
  type StreamEvent,
  type StreamStatus,
} from "./api";

export type View = "data" | "network" | "train" | "probe";

export interface Draft {
  hidden: LayerSpec[];
  output_activation: Activation;
  loss: LossName;
  optimizer: OptimizerName;
  learning_rate: number;
  momentum: number;
  epochs: number;
  batch_size: number;
  validation_split: number;
  shuffle: boolean;
  early_stopping: boolean;
  patience: number;
  shuffle_before_split: boolean;
  normalize: boolean;
  pace: number;
}

export interface RunView {
  summary: RunSummary;
  history: EpochRecord[];
  /** Snapshots with a field, thinned to keep memory flat on long runs */
  frames: Snapshot[];
  latest: Snapshot | null;
}

export interface State {
  view: View;
  dataset: DatasetInfo | null;
  file: File | null;
  target: number | null;
  /** Feature indices (target removed) drawn on the chart */
  plane: [number, number];
  draft: Draft;
  runs: RunView[];
  active: string | null;
  /** Epoch shown on the chart while scrubbing, null follows live */
  scrub: number | null;
  stream: StreamStatus | "idle";
  probe: { inputs: number[]; result: Prediction | null; row: number | null } | null;
  busy: string | null;
  error: string | null;
}

const MAX_FRAMES = 400;

const defaultDraft: Draft = {
  hidden: [
    { size: 8, activation: "tanh" },
    { size: 8, activation: "tanh" },
  ],
  output_activation: "sigmoid",
  loss: "bce",
  optimizer: "adam",
  learning_rate: 0.01,
  momentum: 0.9,
  epochs: 200,
  batch_size: 32,
  validation_split: 0.2,
  shuffle: true,
  early_stopping: true,
  patience: 20,
  shuffle_before_split: true,
  normalize: true,
  pace: 30,
};

let state: State = {
  view: "data",
  dataset: null,
  file: null,
  target: null,
  plane: [0, 1],
  draft: defaultDraft,
  runs: [],
  active: null,
  scrub: null,
  stream: "idle",
  probe: null,
  busy: null,
  error: null,
};

const listeners = new Set<() => void>();
let queued = false;

export function get(): State {
  return state;
}

/** Shallow merge, listeners run once per frame however many sets happen */
export function set(patch: Partial<State> | ((s: State) => Partial<State>)): void {
  const p = typeof patch === "function" ? patch(state) : patch;
  state = { ...state, ...p };
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => {
    queued = false;
    listeners.forEach((l) => l());
  });
}

export function useStore(): State {
  const [, bump] = useState(0);
  useEffect(() => {
    const l = () => bump((n) => n + 1);
    listeners.add(l);
    return () => void listeners.delete(l);
  }, []);
  return state;
}

export function setDraft(patch: Partial<Draft>): void {
  set((s) => ({ draft: { ...s.draft, ...patch } }));
}

export function activeRun(s: State = state): RunView | null {
  return s.runs.find((r) => r.summary.id === s.active) ?? null;
}

export function isLive(r: RunView | null): boolean {
  return !!r && (r.summary.state === "running" || r.summary.state === "paused");
}

/** Features (all columns but the target) of the loaded file */
export function featureColumns(s: State = state) {
  return s.dataset ? s.dataset.columns.filter((c) => c.index !== s.target) : [];
}

function fail(e: unknown) {
  set({ error: e instanceof Error ? e.message : String(e), busy: null });
}

function sensibleDraft(d: DatasetInfo, target: number): Partial<Draft> {
  const col = d.columns[target];
  if (!col) return {};
  return col.binary
    ? { output_activation: "sigmoid", loss: "bce" }
    : { output_activation: "linear", loss: "mse" };
}

function pickTarget(d: DatasetInfo): number {
  // Last 0/1 column, else the last column, matches how most files are laid out
  const usable = d.columns.filter((c) => !c.constant);
  const binary = [...usable].reverse().find((c) => c.binary);
  return (binary ?? usable[usable.length - 1] ?? d.columns[d.columns.length - 1]!).index;
}

function adopt(d: DatasetInfo, file: File | null) {
  const target = pickTarget(d);
  set((s) => ({
    dataset: d,
    file,
    target,
    plane: [0, Math.min(1, d.columns.length - 2)] as [number, number],
    draft: { ...s.draft, ...sensibleDraft(d, target) },
    busy: null,
    error: null,
  }));
}

export async function loadFile(file: File) {
  set({ busy: `Reading ${file.name}`, error: null });
  try {
    adopt(await api.upload(file), file);
  } catch (e) {
    fail(e);
  }
}

export async function loadExample() {
  set({ busy: "Loading example data", error: null });
  try {
    adopt(await api.example(), null);
  } catch (e) {
    fail(e);
  }
}

export async function setHeader(hasHeader: boolean) {
  const s = state;
  if (!s.dataset || !s.file) return;
  set({ busy: "Reading again", error: null });
  try {
    adopt(await api.reparse(s.file, s.dataset.name, hasHeader), s.file);
  } catch (e) {
    fail(e);
  }
}

export function setTarget(target: number) {
  const s = state;
  if (!s.dataset) return;
  set({
    target,
    plane: [0, Math.min(1, s.dataset.columns.length - 2)],
    draft: { ...s.draft, ...sensibleDraft(s.dataset, target) },
  });
}

let closeStream: (() => void) | null = null;

function patchRun(id: string, f: (r: RunView) => RunView) {
  set((s) => ({ runs: s.runs.map((r) => (r.summary.id === id ? f(r) : r)) }));
}

function keepFrame(frames: Snapshot[], snap: Snapshot): Snapshot[] {
  if (!snap.field) return frames;
  const next = frames.length && frames[frames.length - 1]!.epoch === snap.epoch ? frames.slice(0, -1) : frames.slice();
  next.push(snap);
  if (next.length <= MAX_FRAMES) return next;
  // Drop every second older frame, the newest quarter stays dense
  const keep = Math.floor(MAX_FRAMES / 4);
  const old = next.slice(0, -keep).filter((_, i) => i % 2 === 0);
  return [...old, ...next.slice(-keep)];
}

function onEvent(id: string, e: StreamEvent) {
  switch (e.type) {
    case "hello":
      patchRun(id, (r) => ({
        ...r,
        summary: e.summary,
        history: e.history,
        latest: e.snapshot ?? r.latest,
        frames: e.snapshot ? keepFrame(r.frames, e.snapshot) : r.frames,
      }));
      break;
    case "epoch": {
      const { type: _t, ...rec } = e;
      patchRun(id, (r) =>
        r.history.length && r.history[r.history.length - 1]!.epoch >= rec.epoch
          ? r
          : { ...r, history: [...r.history, rec] },
      );
      break;
    }
    case "snapshot": {
      const { type: _t, ...snap } = e;
      patchRun(id, (r) => ({ ...r, latest: snap, frames: keepFrame(r.frames, snap) }));
      break;
    }
    case "state":
      patchRun(id, (r) => ({ ...r, summary: { ...r.summary, state: e.state, reason: e.reason ?? r.summary.reason } }));
      break;
    case "lagged":
      break;
  }
}

function watch(id: string) {
  closeStream?.();
  closeStream = openStream(
    id,
    (e) => onEvent(id, e),
    (st) => set({ stream: st }),
  );
}

export function configFromDraft(s: State = state): RunConfig | null {
  if (!s.dataset || s.target === null) return null;
  const d = s.draft;
  return {
    dataset_id: s.dataset.id,
    target: s.target,
    normalize: d.normalize,
    hidden: d.hidden,
    output_activation: d.output_activation,
    loss: d.loss,
    optimizer: d.optimizer,
    learning_rate: d.learning_rate,
    momentum: d.momentum,
    epochs: d.epochs,
    batch_size: d.batch_size,
    validation_split: d.validation_split,
    shuffle: d.shuffle,
    early_stopping_patience: d.early_stopping ? d.patience : null,
    shuffle_before_split: d.shuffle_before_split,
    plane: s.plane,
    pace: d.pace,
    label: "",
  };
}

export async function startRun() {
  const config = configFromDraft();
  if (!config) return;
  set({ busy: "Building the network", error: null });
  try {
    const summary = await api.start(config);
    set((s) => ({
      runs: [...s.runs, { summary, history: [], frames: [], latest: null }],
      active: summary.id,
      scrub: null,
      view: "train",
      busy: null,
      probe: null,
    }));
    watch(summary.id);
  } catch (e) {
    fail(e);
  }
}

export function selectRun(id: string) {
  set({ active: id, scrub: null, probe: null });
  watch(id);
}

export async function control(cmd: "pause" | "resume" | "step" | "stop") {
  const r = activeRun();
  if (!r) return;
  try {
    await api.control(r.summary.id, cmd);
  } catch (e) {
    fail(e);
  }
}

export async function setPlane(plane: [number, number]) {
  set({ plane });
  const r = activeRun();
  if (!r) return;
  try {
    const snap = await api.plane(r.summary.id, plane[0], plane[1]);
    // Older frames show another plane, they no longer replay
    patchRun(r.summary.id, (v) => ({ ...v, latest: snap, frames: snap.field ? [snap] : [] }));
    set({ scrub: null });
  } catch (e) {
    fail(e);
  }
}

let probeSeq = 0;
export async function probe(inputs: number[], row: number | null = null) {
  const r = activeRun();
  if (!r) return;
  const seq = ++probeSeq;
  set((s) => ({ probe: { inputs, row, result: s.probe?.result ?? null } }));
  try {
    const result = await api.predict(r.summary.id, inputs);
    if (seq === probeSeq) set({ probe: { inputs, row, result } });
  } catch (e) {
    fail(e);
  }
}

/** Pick up runs a reload left behind */
export async function restore() {
  try {
    const runs = await api.runs();
    if (!runs.length) return;
    const last = runs[runs.length - 1]!.summary;
    const dataset = await api.dataset(last.config.dataset_id).catch(() => null);
    set((s) => ({
      runs: runs.map((r) => ({ summary: r.summary, history: r.history, frames: [], latest: null })),
      dataset: s.dataset ?? dataset,
      target: s.dataset ? s.target : last.config.target,
      plane: last.config.plane ?? s.plane,
    }));
    selectRun(last.id);
  } catch {
    // a fresh server simply has nothing to restore
  }
}
