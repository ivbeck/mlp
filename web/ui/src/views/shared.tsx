import type { DatasetInfo, Kind, RunSummary, TargetSpec } from "../api";
import type { Station } from "../charts/Surface";
import { ACTIVATIONS, label } from "../util";

const MISSING = new Set(["", "?", "NA", "N/A", "na", "n/a", "nan", "NaN", "NAN", "null", "NULL", "None", "none", "-"]);

export function isMissing(cell: string | undefined): boolean {
  return cell === undefined || MISSING.has(cell);
}

/** A target cell as the number the network learns: 0/1 for two values, else the number */
export function targetValue(cell: string | undefined, spec: TargetSpec | null): number {
  if (isMissing(cell)) return NaN;
  if (spec?.kind === "binary") return cell === spec.one ? 1 : 0;
  return Number(cell);
}

/** Sample rows as chart stations, rows with a missing cell left out */
export function stationsFor(d: DatasetInfo, xCol: number, yCol: number, target: number, spec: TargetSpec | null): Station[] {
  const out: Station[] = [];
  for (const row of d.sample) {
    const x = isMissing(row[xCol]) ? NaN : Number(row[xCol]);
    const y = isMissing(row[yCol]) ? NaN : Number(row[yCol]);
    const v = targetValue(row[target], spec);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(v)) out.push({ x, y, v });
  }
  return out;
}

export function targetRange(d: DatasetInfo, target: number, spec: TargetSpec | null): [number, number] {
  const c = d.columns[target];
  if (spec?.kind === "binary" || !c || c.min === null || c.max === null) return [0, 1];
  return [c.min, c.max];
}

export const KIND_TEXT: Record<Kind, string> = {
  numeric: "Number",
  binary: "Two values",
  category: "Category",
  date: "Date",
  ignore: "Ignore",
};

export function PlaneSelect({
  feats,
  plane,
  onChange,
  disabled,
}: {
  /** Number columns that can go on an axis, by column index */
  feats: { index: number; name: string }[];
  plane: [number, number];
  onChange: (p: [number, number]) => void;
  disabled?: boolean;
}) {
  if (feats.length < 3) return null;
  const pick = (axis: 0 | 1, v: number) => {
    const next: [number, number] = [...plane];
    next[axis] = v;
    // picking the column already on the other axis swaps them
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
            {feats.map((f) => (
              <option value={f.index}>{f.name}</option>
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
  spec,
  stations,
  vRange,
  targetName,
}: {
  spec: TargetSpec | null;
  stations: { v: number }[];
  vRange: [number, number];
  targetName: string;
}) {
  const ones = stations.filter((s) => s.v >= 0.5).length;
  const one = spec?.kind === "binary" ? spec.one : "1";
  const zero = spec?.kind === "binary" ? spec.zero : "0";
  return (
    <section class="chart-key" aria-label="Chart key">
      <div class="lab">Key</div>
      {spec?.kind === "binary" ? (
        <>
          <div class="key-item">
            <svg viewBox="0 0 24 14" aria-hidden="true"><circle cx="12" cy="7" r="4.2" class="k-zero" /></svg>
            {targetName} {zero} · {stations.length - ones} rows
          </div>
          <div class="key-item">
            <svg viewBox="0 0 24 14" aria-hidden="true"><circle cx="12" cy="7" r="4.2" class="k-one" /></svg>
            {targetName} {one} · {ones} rows
          </div>
          <div class="key-item">
            <svg viewBox="0 0 24 14" aria-hidden="true">
              <path d="M0 7H12" class="k-front-warm" />
              <path d="M12 7H24" class="k-front-cold" />
              <path d="M2 7A4 4 0 0 0 10 7Z" class="k-semi" />
              <path d="M14 7L22 7L18 1Z" class="k-tri" />
            </svg>
            front at p = 0.5, triangles face the {zero} side
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
