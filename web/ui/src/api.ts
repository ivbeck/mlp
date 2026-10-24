// Wire types mirror web/src/{dataset,run}.rs, keep them in step

export type Kind = "numeric" | "binary" | "category" | "date" | "ignore";

export interface ColumnInfo {
  index: number;
  name: string;
  kind: Kind;
  allowed: Kind[];
  reason: string | null;
  missing: number;
  distinct: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  constant: boolean;
  categories: [string, number][];
  binary: boolean;
}

export interface DatasetInfo {
  id: string;
  name: string;
  has_header: boolean;
  rows: number;
  columns: ColumnInfo[];
  /** First rows as text cells */
  sample: string[][];
  size_bytes: number;
  is_sample: boolean;
}

export interface InputInfo {
  index: number;
  name: string;
  kind: Kind;
  options: string[];
  min: number | null;
  max: number | null;
  default: number | string;
  width: number;
}

export type TargetSpec = { kind: "binary"; one: string; zero: string } | { kind: "numeric" };

export interface Encoding {
  names: string[];
  inputs: InputInfo[];
  target: TargetSpec | null;
  error: string | null;
}

export interface LayerSpec {
  size: number;
  activation: Activation;
}

export type Activation = "relu" | "leaky_relu" | "sigmoid" | "tanh" | "linear";
export type LossName = "mse" | "mae" | "bce" | "huber";
export type OptimizerName = "sgd" | "momentum" | "adam" | "rmsprop";

export interface RunConfig {
  dataset_id: string;
  target: number;
  normalize: boolean;
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
  early_stopping_patience: number | null;
  shuffle_before_split: boolean;
  plane: [number, number] | null;
  kinds: Record<number, Kind>;
  target_positive: string | null;
  pace: number;
  label: string;
}

export interface EpochRecord {
  epoch: number;
  train_loss: number;
  val_loss: number | null;
  train_acc: number | null;
  val_acc: number | null;
  best: boolean;
  since_best: number;
  ms: number;
}

export interface Field {
  x: number;
  y: number;
  /** Raw range of a number axis, null on a category axis */
  x_range: [number, number] | null;
  y_range: [number, number] | null;
  /** One band per value on a category axis, null on a number axis */
  x_labels: string[] | null;
  y_labels: string[] | null;
  nx: number;
  ny: number;
  values: number[];
}

export interface Snapshot {
  epoch: number;
  weights: number[][] | null;
  biases: number[][] | null;
  field: Field | null;
}

export type RunState = "running" | "paused" | "finished" | "early_stopped" | "stopped" | "failed";

export interface RunSummary {
  id: string;
  config: RunConfig;
  architecture: number[];
  parameters: number;
  dataset_name: string;
  features: string[];
  target_name: string;
  binary_target: boolean;
  train_rows: number;
  val_rows: number;
  state: RunState;
  reason: string | null;
  notes: { level: "info" | "warn"; text: string }[];
  inputs: InputInfo[];
  target: TargetSpec;
  started_at_ms: number;
}

export interface Prediction {
  input: number[];
  names: string[];
  layers: number[][];
  output: number | null;
}

export type StreamEvent =
  | { type: "hello"; summary: RunSummary; history: EpochRecord[]; snapshot: Snapshot | null }
  | ({ type: "epoch" } & EpochRecord)
  | ({ type: "snapshot" } & Snapshot)
  | { type: "state"; state: RunState; reason: string | null }
  | { type: "lagged" };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new Error("The console server is not reachable. Is mlp-console still running?");
  }
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error((body && typeof body.error === "string" && body.error) || `${res.status} ${res.statusText}`);
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  upload(file: File, hasHeader?: boolean): Promise<DatasetInfo> {
    const q = new URLSearchParams({ name: file.name });
    if (hasHeader !== undefined) q.set("has_header", String(hasHeader));
    return call(`/api/datasets?${q}`, { method: "POST", body: file });
  },
  reparse(file: Blob, name: string, hasHeader: boolean): Promise<DatasetInfo> {
    const q = new URLSearchParams({ name, has_header: String(hasHeader) });
    return call(`/api/datasets?${q}`, { method: "POST", body: file });
  },
  encoding: (id: string, body: { target: number; kinds: Record<number, Kind>; target_positive: string | null }): Promise<Encoding> =>
    call(`/api/datasets/${id}/encoding`, json(body)),
  dataset: (id: string): Promise<DatasetInfo> => call(`/api/datasets/${id}`),
  example: (): Promise<DatasetInfo> => call("/api/datasets/example", { method: "POST" }),
  start: (config: RunConfig): Promise<RunSummary> => call("/api/runs", json(config)),
  runs: (): Promise<{ summary: RunSummary; history: EpochRecord[] }[]> => call("/api/runs"),
  control: (id: string, cmd: "pause" | "resume" | "step" | "stop"): Promise<void> =>
    call(`/api/runs/${id}/control`, json({ cmd })),
  predict: (id: string, inputs: (number | string)[]): Promise<Prediction> =>
    call(`/api/runs/${id}/predict`, json({ inputs })),
  plane: (id: string, x: number, y: number): Promise<Snapshot> =>
    call(`/api/runs/${id}/plane`, json({ x, y })),
};

export type StreamStatus = "connecting" | "live" | "retrying" | "closed";

/** Websocket for one run, reconnects with backoff until the run is over or close() */
export function openStream(
  id: string,
  onEvent: (e: StreamEvent) => void,
  onStatus: (s: StreamStatus) => void,
): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let over = false;
  let delay = 400;
  let timer = 0;

  const connect = () => {
    onStatus(delay > 400 ? "retrying" : "connecting");
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/api/runs/${id}/ws`);
    ws.onopen = () => {
      delay = 400;
      onStatus("live");
    };
    ws.onmessage = (m) => {
      const e = JSON.parse(m.data as string) as StreamEvent;
      if (e.type === "state" && ["finished", "early_stopped", "stopped", "failed"].includes(e.state)) over = true;
      if (e.type === "hello" && ["finished", "early_stopped", "stopped", "failed"].includes(e.summary.state)) over = true;
      onEvent(e);
    };
    ws.onclose = () => {
      if (closed || over) {
        onStatus("closed");
        return;
      }
      onStatus("retrying");
      timer = window.setTimeout(connect, delay);
      delay = Math.min(delay * 2, 8000);
    };
  };
  connect();
  return () => {
    closed = true;
    window.clearTimeout(timer);
    ws?.close();
  };
}
