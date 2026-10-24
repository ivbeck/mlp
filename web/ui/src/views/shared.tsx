import type { ColumnInfo, DatasetInfo, RunSummary } from "../api";
import type { Station } from "../charts/Surface";
import { ACTIVATIONS, label } from "../util";

export function stationsFor(d: DatasetInfo, xCol: number, yCol: number, target: number): Station[] {
  return d.sample.map((row) => ({ x: row[xCol]!, y: row[yCol]!, v: row[target]! }));
}

/** Columns that feed the network of a run, in input order */
export function runFeatures(d: DatasetInfo | null, run: RunSummary): ColumnInfo[] {
  return d && d.id === run.config.dataset_id ? d.columns.filter((c) => c.index !== run.config.target) : [];
}

export function PlaneSelect({
  feats,
  plane,
  onChange,
  disabled,
}: {
  feats: { name: string }[];
  plane: [number, number];
  onChange: (p: [number, number]) => void;
  disabled?: boolean;
}) {
  if (feats.length < 3) return null;
  const pick = (axis: 0 | 1, v: number) => {
    const next: [number, number] = [...plane];
    next[axis] = v;
    // picking the feature already on the other axis swaps them
    if (next[0] === next[1]) next[1 - axis] = plane[axis];
    onChange(next);
  };
  return (
    <div class="plane">
      {(["x", "y"] as const).map((ax, i) => (
        <label>
          {ax}
          <select
            disabled={disabled}
            value={String(plane[i])}
            onChange={(e) => pick(i as 0 | 1, +(e.currentTarget as HTMLSelectElement).value)}
          >
            {feats.map((f, n) => (
              <option value={n}>{f.name}</option>
            ))}
          </select>
        </label>
      ))}
    </div>
  );
}

export function archLine(run: RunSummary): string {
  const c = run.config;
  return [
    String(run.architecture[0]),
    ...c.hidden.map((h) => `${h.size} ${label(ACTIVATIONS, h.activation)}`),
    `1 ${label(ACTIVATIONS, c.output_activation)}`,
  ].join(" → ");
}

export const STATE_TEXT: Record<RunSummary["state"], string> = {
  running: "Training",
  paused: "Paused",
  finished: "Finished",
  early_stopped: "Stopped early",
  stopped: "Stopped",
  failed: "Failed",
};

/** What the marks on the surface mean, next to the chart it explains */
export function ChartKey({
  binary,
  stations,
  vRange,
  targetName,
}: {
  binary: boolean;
  stations: { v: number }[];
  vRange: [number, number];
  targetName: string;
}) {
  const ones = stations.filter((s) => s.v >= 0.5).length;
  return (
    <section class="chart-key" aria-label="Chart key">
      <div class="lab">Key</div>
      {binary ? (
        <>
          <div class="key-item">
            <svg viewBox="0 0 24 14" aria-hidden="true"><circle cx="12" cy="7" r="4.2" class="k-zero" /></svg>
            {targetName} = 0 · {stations.length - ones} rows
          </div>
          <div class="key-item">
            <svg viewBox="0 0 24 14" aria-hidden="true"><circle cx="12" cy="7" r="4.2" class="k-one" /></svg>
            {targetName} = 1 · {ones} rows
          </div>
          <div class="key-item">
            <svg viewBox="0 0 24 14" aria-hidden="true">
              <path d="M0 7H12" class="k-front-warm" />
              <path d="M12 7H24" class="k-front-cold" />
              <path d="M2 7A4 4 0 0 0 10 7Z" class="k-semi" />
              <path d="M14 7L22 7L18 1Z" class="k-tri" />
            </svg>
            front at p = 0.5, triangles face the 0 side
          </div>
        </>
      ) : (
        <div class="key-item ramp">
          <span>{vRange[0].toPrecision(3)}</span>
          <i />
          <span>{vRange[1].toPrecision(3)}</span>
          <span class="muted">{targetName} per row</span>
        </div>
      )}
      <div class="key-item">
        <svg viewBox="0 0 24 14" aria-hidden="true"><path d="M0 9C8 3 16 11 24 5" class="k-iso" /></svg>
        isolines, labelled with the prediction
      </div>
    </section>
  );
}
