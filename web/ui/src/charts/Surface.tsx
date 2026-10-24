import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { Field } from "../api";
import { Bands } from "./Bands";
import { contour, length, niceLevels, sample, walk, type Pt } from "./contour";
import { COLD, PAPER, WARM, mix, pad, readoutHint, stationColour, useTween } from "./paper";
import { fmt, fmtP, useWidth } from "../util";

/** A data row on the chart. On a category axis the coordinate is the band index. */
export interface Station {
  x: number;
  y: number;
  v: number;
}

export interface SurfaceProps {
  field: Field | null;
  stations: Station[];
  binary: boolean;
  /** Raw ranges used when there is no field yet */
  xRange: [number, number];
  yRange: [number, number];
  xName: string;
  yName: string;
  height: number;
  /** Value range of the target, colours regression stations */
  vRange: [number, number];
  /** Bands of a category axis when there is no field yet */
  xLabels?: string[] | null;
  yLabels?: string[] | null;
  probe?: [number | string, number | string] | null;
  onProbe?: (x: number | string, y: number | string) => void;
  label: string;
}

/** Two number axes draw contours, any category axis draws bands */
export function Surface(p: SurfaceProps) {
  const banded = p.field ? !!(p.field.x_labels || p.field.y_labels) : !!(p.xLabels || p.yLabels);
  return banded ? <Bands {...p} /> : <Contours {...p} />;
}

const M = { l: 48, r: 14, t: 14, b: 36 };
// Tried in this order, a level is kept only when it runs clear of the ones already kept,
// so a sharp, converged field still reads as a few spaced lines instead of a stack
const BINARY_LEVELS = [0.5, 0.1, 0.9, 0.25, 0.75, 0.02, 0.98, 0.01, 0.99];
// 18 px on a desktop plot, shrinking on a phone so the outer levels survive there too
const gapFor = (w: number) => Math.max(7, Math.min(18, w / 40));
/** Chaikin corner cutting, takes the marching squares stair steps off a line */
function smooth(l: Pt[], rounds = 2): Pt[] {
  let cur = l;
  for (let r = 0; r < rounds && cur.length > 2; r++) {
    const closed = cur[0]![0] === cur[cur.length - 1]![0] && cur[0]![1] === cur[cur.length - 1]![1];
    const next: Pt[] = closed ? [] : [cur[0]!];
    for (let i = 0; i < cur.length - 1; i++) {
      const a = cur[i]!;
      const b = cur[i + 1]!;
      next.push([a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25], [a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75]);
    }
    if (closed) next.push(next[0]!);
    else next.push(cur[cur.length - 1]!);
    cur = next;
  }
  return cur;
}

function Contours(p: SurfaceProps) {
  const [wrap, width] = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const values = useTween(p.field);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const field = p.field;
  const lastProbe = useRef(0);

  const W = Math.max(width - M.l - M.r, 10);
  const H = Math.max(p.height - M.t - M.b, 10);
  const [x0, x1] = field?.x_range ?? pad(p.xRange);
  const [y0, y1] = field?.y_range ?? pad(p.yRange);
  const res = field?.nx ?? 2;
  const px = (x: number) => M.l + ((x - x0) / (x1 - x0)) * W;
  const py = (y: number) => M.t + H - ((y - y0) / (y1 - y0)) * H;
  const gpx = (g: Pt): Pt => [M.l + (g[0] / (res - 1)) * W, M.t + H - (g[1] / (res - 1)) * H];
  const toRaw = (cx: number, cy: number): [number, number] => [
    x0 + ((cx - M.l) / W) * (x1 - x0),
    y0 + ((M.t + H - cy) / H) * (y1 - y0),
  ];
  const at = (x: number, y: number) =>
    values ? sample(values, res, res, ((x - x0) / (x1 - x0)) * (res - 1), ((y - y0) / (y1 - y0)) * (res - 1)) : null;

  let lo = 0;
  let hi = 1;
  if (values && !p.binary) {
    lo = Infinity;
    hi = -Infinity;
    for (const v of values) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  const levels = useMemo(
    () => (p.binary ? BINARY_LEVELS : niceLevels(lo, hi)),
    // regression levels follow the field, rounded so they do not flicker every frame
    [p.binary, p.binary ? 0 : +lo.toPrecision(2), p.binary ? 0 : +hi.toPrecision(2)],
  );
  const mid = p.binary ? 0.5 : null;

  // Tint: a res x res image scaled up smooth, warm above the boundary, cold below
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(W * dpr);
    c.height = Math.round(H * dpr);
    const g = c.getContext("2d");
    if (!g) return;
    g.clearRect(0, 0, c.width, c.height);
    if (!values) return;
    const img = new ImageData(res, res);
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const v = values[j * res + i]!;
        const k = p.binary ? (v - 0.5) * 2 : hi > lo ? ((v - lo) / (hi - lo)) * 2 - 1 : 0;
        const rgb = mix(PAPER, k >= 0 ? WARM : COLD, Math.min(1, Math.abs(k)) * 0.11);
        const o = ((res - 1 - j) * res + i) * 4;
        img.data[o] = rgb[0]!;
        img.data[o + 1] = rgb[1]!;
        img.data[o + 2] = rgb[2]!;
        img.data[o + 3] = 255;
      }
    }
    const small = new OffscreenCanvas(res, res);
    small.getContext("2d")?.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";
    // half a cell inset, cell centres sit on the grid points
    const cw = c.width / (res - 1);
    const ch = c.height / (res - 1);
    g.drawImage(small, -cw / 2, -ch / 2, c.width + cw, c.height + ch);
  }, [values, W, H, res, p.binary, lo, hi]);

  const lines = useMemo(() => {
    if (!values) return [];
    const kept: { level: number; paths: Pt[][]; probe: Pt[] }[] = [];
    const gap = gapFor(W);
    const order = p.binary ? levels : [...levels].sort((a, b) => Math.abs(a - (lo + hi) / 2) - Math.abs(b - (lo + hi) / 2));
    for (const level of order) {
      const paths = contour(values, res, level)
        .map((l) => smooth(l.map(gpx)))
        .filter((l) => length(l) > 8);
      if (!paths.length) continue;
      const probe = paths.flatMap((l) => walk(l, 10).map((w) => w.p));
      // dropped when a third of it sits closer than the gap to a kept line
      const close = probe.filter((q) =>
        kept.some((k) => k.probe.some((r) => Math.abs(r[0] - q[0]) < gap && Math.hypot(r[0] - q[0], r[1] - q[1]) < gap)),
      ).length;
      if (kept.length && close > probe.length / 3) continue;
      kept.push({ level, paths, probe });
    }
    return kept.sort((a, b) => a.level - b.level).map(({ level, paths }) => ({ level, paths }));
  }, [values, levels, W, H]);

  // Pressure centres, marked like H and L: among the cells near the field's max (min) the one
  // furthest from any station, kept inside the frame and clear of the readout box
  const centres = useMemo(() => {
    const f = p.field;
    if (!f) return [];
    const vals = f.values;
    let hiV = -Infinity;
    let loV = Infinity;
    for (const v of vals) {
      hiV = Math.max(hiV, v);
      loV = Math.min(loV, v);
    }
    const spread = hiV - loV;
    if (!(spread > (p.binary ? 0.5 : 0))) return [];
    const pts = p.stations.map((st) => [px(st.x), py(st.y)] as Pt);
    const clampTo = (q: Pt): Pt => {
      let x = Math.min(Math.max(q[0], M.l + 58), M.l + W - 58);
      let y = Math.min(Math.max(q[1], M.t + 34), M.t + H - 12);
      if (x > M.l + W - 330 && y < M.t + 58) y = M.t + 58;
      return [x, y];
    };
    const pick = (warm: boolean, avoid: Pt | null) => {
      let best: Pt | null = null;
      let bestScore = -1;
      let bestV = 0;
      for (let i = 0; i < vals.length; i++) {
        const v = vals[i]!;
        if (warm ? v < hiV - spread * 0.04 : v > loV + spread * 0.04) continue;
        const raw = gpx([i % res, Math.floor(i / res)]);
        const at = clampTo(raw);
        // only cells that already sit inside the safe box, a clamped one would print its
        // value somewhere the field has another; and the two numerals stay well apart
        if (Math.hypot(at[0] - raw[0], at[1] - raw[1]) > 2) continue;
        if (avoid && Math.hypot(avoid[0] - at[0], avoid[1] - at[1]) < 150) continue;
        let near = Infinity;
        for (const q of pts) near = Math.min(near, Math.hypot(q[0] - at[0], (q[1] - at[1] + 8) * 1.4));
        if (near > bestScore) {
          bestScore = near;
          best = at;
          bestV = v;
        }
      }
      return best ? { at: best, v: bestV, warm } : null;
    };
    const hiC = pick(true, null);
    const loC = pick(false, hiC?.at ?? null);
    return [hiC, loC].filter((c): c is { at: Pt; v: number; warm: boolean } => c !== null);
  }, [p.field, p.stations, W, H]);

  const xt = niceLevels(x0, x1, 7);
  const yt = niceLevels(y0, y1, 5);
  const d = (l: Pt[]) => "M" + l.map((q) => `${q[0].toFixed(1)} ${q[1].toFixed(1)}`).join("L");

  const front = mid === null ? null : lines.find((l) => l.level === mid);
  const symbols: preact.JSX.Element[] = [];
  if (front && values) {
    let n = 0;
    for (const path of front.paths) {
      if (length(path) < 30) continue;
      for (const { p: q, t } of walk(path, 42)) {
        const nx = -t[1];
        const ny = t[0];
        const [rx, ry] = toRaw(q[0] + nx * 7, q[1] + ny * 7);
        const warmSide = (at(rx, ry) ?? 0.5) > 0.5 ? 1 : -1;
        const wx = nx * warmSide;
        const wy = ny * warmSide;
        const ang = (Math.atan2(t[1], t[0]) * 180) / Math.PI;
        if (n++ % 2 === 0) {
          // cold triangle, pointing into the class 0 side
          const pts = [
            [q[0] - t[0] * 6, q[1] - t[1] * 6],
            [q[0] + t[0] * 6, q[1] + t[1] * 6],
            [q[0] - wx * 9, q[1] - wy * 9],
          ];
          symbols.push(<path class="sym cold" d={`M${pts.map((v) => v.join(" ")).join("L")}Z`} />);
        } else {
          // warm semicircle bulging into the class 1 side
          // local +y of the rotated frame is the normal; sweep 0 bulges towards +y
          symbols.push(
            <path
              class="sym warm"
              transform={`translate(${q[0]} ${q[1]}) rotate(${ang})`}
              d={`M-6 0A6 6 0 0 ${warmSide > 0 ? 0 : 1} 6 0Z`}
            />,
          );
        }
      }
    }
  }

  // One boxed numeral per level on its longest line, nudged along it until it clears the others
  const placed: Pt[] = centres.map((c) => [c.at[0], c.at[1] - 8] as Pt);
  const labels = lines
    .filter((l) => l.paths.length)
    .sort((a, b) => (a.level === mid ? -1 : b.level === mid ? 1 : 0))
    .map((l) => {
      const best = l.paths.reduce((a, b) => (length(b) > length(a) ? b : a));
      const len = length(best);
      if (len < 60) return null;
      let spot: Pt | null = null;
      for (const f of [0.5, 0.35, 0.65, 0.2, 0.8, 0.42, 0.58, 0.28, 0.72, 0.12, 0.88]) {
        const at = walk(best, len, len * f)[0]?.p;
        if (!at) continue;
        if (at[0] < M.l + 24 || at[0] > M.l + W - 24 || at[1] < M.t + 12 || at[1] > M.t + H - 12) continue;
        if (placed.every((q) => Math.abs(q[0] - at[0]) > 48 || Math.abs(q[1] - at[1]) > 24)) {
          spot = at;
          break;
        }
      }
      if (!spot) return null;
      placed.push(spot);
      const text = fmt(l.level, 2);
      const w = text.length * 6.6 + 10;
      return (
        <g class={l.level === mid ? "lbl front" : "lbl"} transform={`translate(${spot[0]} ${spot[1]})`}>
          <rect x={-w / 2} y={-9} width={w} height={18} />
          <text y={4}>{text}</text>
        </g>
      );
    });


  const move = (e: PointerEvent) => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const cx = e.clientX - r.left;
    const cy = e.clientY - r.top;
    if (cx < M.l || cx > M.l + W || cy < M.t || cy > M.t + H) {
      setHover(null);
      return;
    }
    const raw = toRaw(cx, cy);
    setHover(raw);
    if (p.onProbe && e.buttons === 1 && performance.now() - lastProbe.current > 60) {
      lastProbe.current = performance.now();
      p.onProbe(raw[0], raw[1]);
    }
  };
  const probeAt = p.probe ? ([Number(p.probe[0]), Number(p.probe[1])] as const) : null;
  const down = (e: PointerEvent) => {
    if (!p.onProbe) return;
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const raw = toRaw(e.clientX - r.left, e.clientY - r.top);
    lastProbe.current = performance.now();
    p.onProbe(raw[0], raw[1]);
  };

  const hv = hover ? at(hover[0], hover[1]) : null;

  return (
    <div class={`surface${p.onProbe ? " probing" : ""}`} ref={wrap} style={{ height: p.height }}>
      <canvas
        ref={canvas}
        style={{ left: M.l, top: M.t, width: W, height: H }}
        aria-hidden="true"
      />
      <svg width={width} height={p.height} role="img" aria-label={p.label} onPointerMove={move} onPointerLeave={() => setHover(null)} onPointerDown={down}>
        <g class="grid">
          {xt.map((t) => <line x1={px(t)} x2={px(t)} y1={M.t} y2={M.t + H} />)}
          {yt.map((t) => <line x1={M.l} x2={M.l + W} y1={py(t)} y2={py(t)} />)}
        </g>
        <g class="ticks">
          {xt.map((t) => <text x={px(t)} y={M.t + H + 16} text-anchor="middle">{fmt(t, 2)}</text>)}
          {yt.map((t) => <text x={M.l - 8} y={py(t) + 4} text-anchor="end">{fmt(t, 2)}</text>)}
        </g>
        <text class="axis" x={M.l + W} y={M.t + H + 31} text-anchor="end">{p.xName} →</text>
        <text class="axis" x={M.l + 8} y={M.t + 16}>↑ {p.yName}</text>
        {lines.map((l) =>
          l.level === mid ? null : l.paths.map((path) => <path class="iso" d={d(path)} />),
        )}
        {front?.paths.map((path) => (
          <>
            <path class="front-warm" d={d(path)} />
            <path class="front-cold" d={d(path)} />
          </>
        ))}
        {symbols}
        {centres.map((c) => (
          <text class={c.warm ? "centre warm" : "centre cold"} x={c.at[0]} y={c.at[1]} text-anchor="middle">
            {p.binary ? c.v.toFixed(2) : fmt(c.v, 3)}
          </text>
        ))}
        <g class="stations">
          {p.stations.map((s) =>
            p.binary ? (
              <circle cx={px(s.x)} cy={py(s.y)} r={4.2} class={s.v >= 0.5 ? "one" : "zero"} />
            ) : (
              <circle cx={px(s.x)} cy={py(s.y)} r={4.2} class="val" fill={stationColour(s.v, p.vRange)} />
            ),
          )}
        </g>
        {labels}
        {probeAt && Number.isFinite(probeAt[0]) && Number.isFinite(probeAt[1]) && (
          <g class="probe-mark" transform={`translate(${px(probeAt[0])} ${py(probeAt[1])})`}>
            <circle r={11} />
            <line x1={-17} x2={-6} />
            <line x1={6} x2={17} />
            <line y1={-17} y2={-6} />
            <line y1={6} y2={17} />
          </g>
        )}
        <rect class="frame" x={M.l} y={M.t} width={W} height={H} />
      </svg>
      <div class="readout" aria-live="off">
        {hover ? (
          <>
            <span>{p.xName} <b>{fmt(hover[0], 3)}</b></span>
            <span>{p.yName} <b>{fmt(hover[1], 3)}</b></span>
            {hv !== null && <span>{p.binary ? "p" : "ŷ"} <b>{p.binary ? fmtP(hv) : fmt(hv, 3)}</b></span>}
          </>
        ) : (
          <span class="hint">{readoutHint(!!p.onProbe)}</span>
        )}
      </div>
    </div>
  );
}
