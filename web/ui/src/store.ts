import { useEffect, useState } from "preact/hooks";
import {
  api,
  openStream,
  type Activation,
  type ColumnInfo,
  type DatasetInfo,
  type Encoding,
  type Kind,
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
  /** Column kinds set by hand, by column index */
  kinds: Record<number, Kind>;
  /** Which value of a two-valued target counts as 1, null = the server's pick */
  positive: string | null;
  /** What the network inputs would be with the current choices, from the server */
  encoding: Encoding | null;
  /** Column indices of the two numbers drawn on the chart */
  plane: [number, number];
  draft: Draft;
  runs: RunView[];
  active: string | null;
  /** Epoch shown on the chart while scrubbing, null follows live */
  scrub: number | null;
  stream: StreamStatus | "idle";
  probe: { inputs: (number | string)[]; result: Prediction | null; row: number | null } | null;
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
  kinds: {},
  positive: null,
  encoding: null,
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

export function kindOf(c: ColumnInfo, s: State = state): Kind {
  const k = s.kinds[c.index];
  return k && c.allowed.includes(k) ? k : c.kind;
}

/** Columns that feed the network: everything but the target and ignored ones */
export function featureColumns(s: State = state) {
  return s.dataset ? s.dataset.columns.filter((c) => c.index !== s.target && kindOf(c, s) !== "ignore") : [];
}

/** Feature columns used as numbers, the only ones the chart can put on an axis */
export function numericColumns(s: State = state) {
  return featureColumns(s).filter((c) => kindOf(c, s) === "numeric");
}

function fail(e: unknown) {
  set({ error: e instanceof Error ? e.message : String(e), busy: null });
}

function isClassTarget(c: ColumnInfo | undefined): boolean {
  return !!c && (c.binary || (c.distinct === 2 && c.min === null));
}

function sensibleDraft(d: DatasetInfo, target: number): Partial<Draft> {
  const head = isClassTarget(d.columns[target])
    ? { output_activation: "sigmoid" as const, loss: "bce" as const }
    : { output_activation: "linear" as const, loss: "mse" as const };
  // The crate steps the optimizer after every row, so a big file needs a small rate
  // and a wide one a wider net; small files keep the playground defaults
  const big = d.rows > 2000;
  return {
    ...head,
    learning_rate: big ? 0.001 : 0.01,
    patience: big ? 6 : 20,
    hidden: big
      ? [
          { size: 32, activation: "relu" },
          { size: 16, activation: "relu" },
        ]
      : [
          { size: 8, activation: "tanh" },
          { size: 8, activation: "tanh" },
        ],
  };
}

function pickTarget(d: DatasetInfo): number {
  // Last two-valued column, else the last number column, matches how most files are laid out
  const usable = d.columns.filter((c) => !c.constant);
  const binary = [...usable].reverse().find(isClassTarget);
  const numeric = [...usable].reverse().find((c) => c.min !== null);
  return (binary ?? numeric ?? d.columns[d.columns.length - 1]!).index;
}

/** Absolute Pearson correlation over the sample rows, cells that do not read as numbers skipped */
function correlation(xs: number[], ys: number[]): number {
  let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]!;
    const y = ys[i]!;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
  }
  const vx = n * sxx - sx * sx;
  const vy = n * syy - sy * sy;
  return n > 2 && vx > 0 && vy > 0 ? Math.abs((n * sxy - sx * sy) / Math.sqrt(vx * vy)) : 0;
}

/** The two number columns that move most with the target go on the axes */
function defaultPlane(s: State): [number, number] {
  const d = s.dataset;
  const nums = numericColumns(s);
  const t = d && s.target !== null ? d.columns[s.target] : undefined;
  if (!d || !t || nums.length < 2) return [nums[0]?.index ?? 0, nums[1]?.index ?? nums[0]?.index ?? 0];
  const first = t.categories[0]?.[0];
  const ys = d.sample.map((r) => (t.min !== null ? Number(r[t.index]) : r[t.index] === first ? 1 : 0));
  const ranked = nums
    .map((c) => {
      // a column that is one value in most rows (capital gain: 0) makes a flat axis
      const share = (c.categories[0]?.[1] ?? 0) / Math.max(1, d.rows - c.missing);
      return { c, r: correlation(d.sample.map((row) => Number(row[c.index])), ys) * (share > 0.7 ? 0.25 : 1) };
    })
    .sort((a, b) => b.r - a.r)
    .map(({ c }) => c.index);
  return [ranked[0]!, ranked[1]!];
}

let encTimer = 0;
let encSeq = 0;
/** Ask the server what the current column choices encode to, a moment after the last change */
function refreshEncoding() {
  window.clearTimeout(encTimer);
  encTimer = window.setTimeout(async () => {
    const s = state;
    if (!s.dataset || s.target === null) return;
    const seq = ++encSeq;
    try {
      const encoding = await api.encoding(s.dataset.id, { target: s.target, kinds: s.kinds, target_positive: s.positive });
      if (seq === encSeq) set({ encoding });
    } catch (e) {
      fail(e);
    }
  }, 120);
}

function adopt(d: DatasetInfo, file: File | null) {
  const target = pickTarget(d);
  set((s) => {
    const next = { ...s, dataset: d, target, kinds: {}, positive: null, encoding: null };
    return {
      dataset: d,
      file,
      target,
      kinds: {},
      positive: null,
      encoding: null,
      plane: defaultPlane(next),
      draft: { ...s.draft, ...sensibleDraft(d, target) },
      busy: null,
      error: null,
    };
  });
  refreshEncoding();
}

export function setKind(index: number, kind: Kind) {
  set((s) => {
    const kinds = { ...s.kinds, [index]: kind };
    // the axes stay unless one of them stopped being a number or category column
    const next = { ...s, kinds };
    const ok = (i: number) => featureColumns(next).some((c) => c.index === i && ["numeric", "category", "binary"].includes(kindOf(c, next)));
    const plane = ok(s.plane[0]) && ok(s.plane[1]) ? s.plane : defaultPlane(next);
    return { kinds, plane };
  });
  refreshEncoding();
}

export function setPositive(positive: string) {
  set({ positive });
  refreshEncoding();
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
    positive: null,
    plane: defaultPlane({ ...s, target }),
    // a new target only changes the output head, layers the user set stay
    draft: (({ output_activation, loss }) => ({ ...s.draft, output_activation: output_activation ?? s.draft.output_activation, loss: loss ?? s.draft.loss }))(
      sensibleDraft(s.dataset, target),
    ),
  });
  refreshEncoding();
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
    plane: planePositions(s.encoding?.inputs ?? [], s.plane),
    kinds: s.kinds,
    target_positive: s.positive,
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

/** Column indices to positions in a run's input list, null when either is not an input */
export function planePositions(inputs: { index: number }[], plane: [number, number]): [number, number] | null {
  const x = inputs.findIndex((i) => i.index === plane[0]);
  const y = inputs.findIndex((i) => i.index === plane[1]);
  return x >= 0 && y >= 0 && x !== y ? [x, y] : null;
}

export async function setPlane(plane: [number, number]) {
  set({ plane });
  const r = activeRun();
  if (!r) return;
  const pos = planePositions(r.summary.inputs, plane);
  if (!pos) return;
  try {
    const snap = await api.plane(r.summary.id, pos[0], pos[1]);
    // Older frames show another plane, they no longer replay
    patchRun(r.summary.id, (v) => ({ ...v, latest: snap, frames: snap.field ? [snap] : [] }));
    set({ scrub: null });
  } catch (e) {
    fail(e);
  }
}

let probeSeq = 0;
export async function probe(inputs: (number | string)[], row: number | null = null) {
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
      kinds: s.dataset ? s.kinds : Object.fromEntries(Object.entries(last.config.kinds).map(([k, v]) => [+k, v])),
      positive: s.dataset ? s.positive : last.config.target_positive,
      plane: last.config.plane
        ? ([last.inputs[last.config.plane[0]]?.index ?? 0, last.inputs[last.config.plane[1]]?.index ?? 0] as [number, number])
        : s.plane,
    }));
    selectRun(last.id);
    refreshEncoding();
  } catch {
    // a fresh server simply has nothing to restore
  }
}
