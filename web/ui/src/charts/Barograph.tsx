import { useState } from "preact/hooks";
import type { EpochRecord } from "../api";
import { niceLevels } from "./contour";
import { fmt, useWidth } from "../util";

interface Props {
  history: EpochRecord[];
  /** Other runs with the same loss, drawn faint for comparison */
  others: { id: string; history: EpochRecord[] }[];
  epochs: number;
  height: number;
  /** Epochs that have a stored field, the scrubber snaps to them */
  frames: number[];
  scrub: number | null;
  onScrub: (epoch: number | null) => void;
  lossName: string;
}

const M = { l: 48, r: 14, t: 18, b: 26 };

export function Barograph(p: Props) {
  const [wrap, width] = useWidth<HTMLDivElement>();
  const [log, setLog] = useState(false);
  const [drag, setDrag] = useState(false);
  const W = Math.max(width - M.l - M.r, 10);
  const H = Math.max(p.height - M.t - M.b, 10);

  const last = p.history[p.history.length - 1];
  const xMax = Math.max(p.epochs, last?.epoch ?? 1, 2);
  const all = [...p.history, ...p.others.flatMap((o) => o.history)];
  const vals = all.flatMap((r) => [r.train_loss, r.val_loss ?? NaN]).filter((v) => Number.isFinite(v) && (!log || v > 0));
  let lo = vals.length ? Math.min(...vals) : 0;
  let hi = vals.length ? Math.max(...vals) : 1;
  if (!log) lo = Math.min(0, lo);
  if (hi <= lo) hi = lo + 1;
  const f = (v: number) => (log ? Math.log10(v) : v);
  const [flo, fhi] = [f(lo), f(hi)];

  const x = (e: number) => M.l + ((e - 1) / (xMax - 1)) * W;
  const y = (v: number) => M.t + H - ((f(v) - flo) / (fhi - flo || 1)) * H;

  // At most two points per pixel column, long runs stay cheap to redraw
  const line = (h: EpochRecord[], pick: (r: EpochRecord) => number | null) => {
    const step = Math.max(1, Math.floor(h.length / (W * 2)));
    let d = "";
    for (let i = 0; i < h.length; i += step) {
      const r = h[i]!;
      const v = pick(r);
      if (v === null || !Number.isFinite(v) || (log && v <= 0)) continue;
      d += `${d ? "L" : "M"}${x(r.epoch).toFixed(1)} ${y(v).toFixed(1)}`;
    }
    return d;
  };

  const yt = log
    ? niceLevels(flo, fhi, 4).map((e) => 10 ** e)
    : niceLevels(lo, hi, 4);
  const xt = niceLevels(1, xMax, 8).filter((e) => e >= 1);

  const shown = p.scrub ?? last?.epoch ?? null;
  const rec = shown !== null ? p.history.find((r) => r.epoch === shown) ?? last : null;
  const best = p.history.reduce<EpochRecord | null>(
    (b, r) => ((r.val_loss ?? r.train_loss) < (b ? b.val_loss ?? b.train_loss : Infinity) ? r : b),
    null,
  );

  const epochAt = (clientX: number, el: Element) => {
    const r = el.getBoundingClientRect();
    const e = 1 + ((clientX - r.left - M.l) / W) * (xMax - 1);
    if (!p.frames.length) return null;
    let nearest = p.frames[0]!;
    for (const fe of p.frames) if (Math.abs(fe - e) < Math.abs(nearest - e)) nearest = fe;
    return nearest;
  };
  const latestFrame = p.frames[p.frames.length - 1];
  const scrubTo = (e: number | null) => p.onScrub(e === null || e === latestFrame ? null : e);

  return (
    <div class="barograph" ref={wrap}>
      <div class="baro-head">
        <h2>Barograph</h2>
        <span class="key"><i class="k-train" />train {fmt(rec?.train_loss)}</span>
        {rec?.val_loss !== null && rec?.val_loss !== undefined && (
          <span class="key"><i class="k-val" />val {fmt(rec.val_loss)}</span>
        )}
        {p.others.length > 0 && <span class="key"><i class="k-other" />other runs, val</span>}
        <span class="spacer" />
        <span class="meta">{p.lossName} · epoch {shown ?? 0} of {p.epochs}</span>
        <div class="seg" role="group" aria-label="Loss scale">
          <button class={log ? "" : "on"} aria-pressed={!log} onClick={() => setLog(false)}>Linear</button>
          <button class={log ? "on" : ""} aria-pressed={log} onClick={() => setLog(true)}>Log</button>
        </div>
        {p.scrub !== null && (
          <button class="link" onClick={() => p.onScrub(null)}>Back to live</button>
        )}
      </div>
      <svg
        width={width}
        height={p.height}
        class={drag ? "dragging" : ""}
        role="slider"
        tabIndex={0}
        aria-label="Epoch shown on the chart"
        aria-valuemin={1}
        aria-valuemax={xMax}
        aria-valuenow={shown ?? 0}
        onPointerDown={(e) => {
          (e.currentTarget as Element).setPointerCapture(e.pointerId);
          setDrag(true);
          scrubTo(epochAt(e.clientX, e.currentTarget as Element));
        }}
        onPointerMove={(e) => drag && scrubTo(epochAt(e.clientX, e.currentTarget as Element))}
        onPointerUp={() => setDrag(false)}
        onKeyDown={(e) => {
          if (!p.frames.length || shown === null) return;
          const i = p.frames.findIndex((fe) => fe >= shown);
          if (e.key === "ArrowLeft") scrubTo(p.frames[Math.max(0, i - 1)] ?? null);
          else if (e.key === "ArrowRight") scrubTo(p.frames[Math.min(p.frames.length - 1, i + 1)] ?? null);
          else if (e.key === "End") scrubTo(null);
          else return;
          e.preventDefault();
        }}
      >
        <g class="grid fine">
          {Array.from({ length: Math.min(60, Math.floor(W / 14)) }, (_, i) => (
            <line x1={M.l + (i * W) / Math.min(60, Math.floor(W / 14))} x2={M.l + (i * W) / Math.min(60, Math.floor(W / 14))} y1={M.t} y2={M.t + H} />
          ))}
        </g>
        <g class="grid">
          {yt.map((t) => <line x1={M.l} x2={M.l + W} y1={y(t)} y2={y(t)} />)}
        </g>
        <g class="ticks">
          {yt.map((t) => <text x={M.l - 8} y={y(t) + 4} text-anchor="end">{fmt(t, 2)}</text>)}
          {xt.map((t) => <text x={x(t)} y={M.t + H + 17} text-anchor="middle">{t}</text>)}
        </g>
        {p.others.map((o) => <path class="trace other" d={line(o.history, (r) => r.val_loss ?? r.train_loss)} />)}
        <path class="trace train" d={line(p.history, (r) => r.train_loss)} />
        <path class="trace val" d={line(p.history, (r) => r.val_loss)} />
        {best && (best.val_loss ?? best.train_loss) !== undefined && (
          <circle class="best" cx={x(best.epoch)} cy={y(best.val_loss ?? best.train_loss)} r={4.5}>
            <title>Best epoch {best.epoch}</title>
          </circle>
        )}
        {shown !== null && (
          <g class={`scrubber${p.scrub !== null ? " held" : ""}`} transform={`translate(${x(shown)} 0)`}>
            <line y1={M.t} y2={M.t + H} />
            <path d={`M-6 ${M.t - 10}L6 ${M.t - 10}L0 ${M.t - 1}Z`} />
          </g>
        )}
        <rect class="frame" x={M.l} y={M.t} width={W} height={H} />
      </svg>
    </div>
  );
}
