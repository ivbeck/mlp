import { useEffect, useRef, useState } from "preact/hooks";
import { niceLevels, sample } from "./contour";
import { COLD, PAPER, WARM, mix, pad, readoutHint, stationColour, useTween } from "./paper";
import type { SurfaceProps } from "./Surface";
import { fmt, fmtP, useWidth } from "../util";

// A category axis has no "between", so instead of contours the chart is cut into one band
// per value. Inside a band the number axis still runs smooth, and the 0.5 front shows up
// as a short tick where the prediction crosses it. Two category axes make a table of cells.

const GAP = 1.5;
const LABEL_MAX = 18;

const short = (s: string) => (s.length > LABEL_MAX ? `${s.slice(0, LABEL_MAX - 1)}…` : s);
const longest = (l: string[]) => Math.max(...l.map((s) => Math.min(s.length, LABEL_MAX)));
/** Spread the stations of one band across it, same spot every render. The R2 sequence
 * fills two axes evenly, a single golden step for both would line them up diagonally. */
const jitterX = (k: number) => (((0.5 + k * 0.7548776662) % 1) - 0.5) * 0.62;
const jitterY = (k: number) => (((0.5 + k * 0.569840291) % 1) - 0.5) * 0.62;

/** Band index of a cell on a category axis, missing and unlisted values in their own bands */
export function bandOf(cell: string | undefined, labels: string[]): number {
  const missing = cell === undefined || ["", "?", "NA", "N/A", "na", "n/a", "nan", "NaN", "null", "NULL", "None", "none", "-"].includes(cell);
  const i = labels.indexOf(missing ? "(missing)" : (cell as string));
  return i >= 0 ? i : labels.indexOf("(other)");
}

export function Bands(p: SurfaceProps) {
  const [wrap, width] = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const eased = useTween(p.field);
  const [hover, setHover] = useState<[number | string, number | string] | null>(null);
  const lastProbe = useRef(0);
  const f = p.field;

  const xl = f ? f.x_labels : (p.xLabels ?? null);
  const yl = f ? f.y_labels : (p.yLabels ?? null);
  const xr = f?.x_range ?? pad(p.xRange);
  const yr = f?.y_range ?? pad(p.yRange);
  const nx = f?.nx ?? xl?.length ?? 2;
  const ny = f?.ny ?? yl?.length ?? 2;
  // right after an axis change the eased field is still the old grid for one render
  const values = eased && eased.length === nx * ny ? eased : null;

  const left = yl ? Math.min(150, Math.max(60, longest(yl) * 6.6 + 18)) : 52;
  const plotW = Math.max(width - left - 14, 10);
  const rotate = !!xl && xl.length * longest(xl) * 6.4 > plotW * 0.92;
  const bottom = xl ? (rotate ? Math.min(110, longest(xl) * 4.2 + 40) : 44) : 36;
  // the y axis name sits above the frame, inside it would cover a band
  const M = { l: left, r: 14, t: 30, b: bottom };
  const W = plotW;
  const H = Math.max(p.height - M.t - M.b, 10);
  const bw = xl ? W / xl.length : 0;
  const bh = yl ? H / yl.length : 0;

  const px = (v: number) => (xl ? M.l + (v + 0.5) * bw : M.l + ((v - xr[0]) / (xr[1] - xr[0])) * W);
  const py = (v: number) => (yl ? M.t + H - (v + 0.5) * bh : M.t + H - ((v - yr[0]) / (yr[1] - yr[0])) * H);
  const clampI = (i: number, n: number) => Math.min(n - 1, Math.max(0, i));
  const toRaw = (cx: number, cy: number): [number | string, number | string] => [
    xl ? xl[clampI(Math.floor((cx - M.l) / bw), xl.length)]! : xr[0] + ((cx - M.l) / W) * (xr[1] - xr[0]),
    yl ? yl[clampI(Math.floor((M.t + H - cy) / bh), yl.length)]! : yr[0] + ((M.t + H - cy) / H) * (yr[1] - yr[0]),
  ];
  /** Grid position of a raw value: band index, or fractional step along a number axis */
  const gx = (v: number | string) => (xl ? Math.max(0, xl.indexOf(String(v))) : ((Number(v) - xr[0]) / (xr[1] - xr[0])) * (nx - 1));
  const gy = (v: number | string) => (yl ? Math.max(0, yl.indexOf(String(v))) : ((Number(v) - yr[0]) / (yr[1] - yr[0])) * (ny - 1));

  let lo = 0;
  let hi = 1;
  if (values && !p.binary) {
    lo = Math.min(...values);
    hi = Math.max(...values);
  }
  const tint = (v: number) => {
    const k = p.binary ? (v - 0.5) * 2 : hi > lo ? ((v - lo) / (hi - lo)) * 2 - 1 : 0;
    return mix(PAPER, k >= 0 ? WARM : COLD, Math.min(1, Math.abs(k)) * 0.42);
  };

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(W * dpr);
    c.height = Math.round(H * dpr);
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!values) return;
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = "high";
    const at = (i: number, j: number) => values[j * nx + i]!;
    const strip = (w: number, h: number, fill: (k: number) => number[]) => {
      const img = new ImageData(w, h);
      for (let k = 0; k < w * h; k++) {
        const rgb = fill(k);
        img.data.set([rgb[0]!, rgb[1]!, rgb[2]!, 255], k * 4);
      }
      const o = new OffscreenCanvas(w, h);
      o.getContext("2d")?.putImageData(img, 0, 0);
      return o;
    };
    if (xl && yl) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          g.fillStyle = `rgb(${tint(at(i, j)).join(",")})`;
          g.fillRect(i * bw + GAP, H - (j + 1) * bh + GAP, bw - 2 * GAP, bh - 2 * GAP);
        }
      }
    } else if (xl) {
      // one vertical strip per band, the number axis smooth inside it
      const ch = H / (ny - 1);
      for (let i = 0; i < nx; i++) {
        const o = strip(1, ny, (k) => tint(at(i, ny - 1 - k)));
        g.drawImage(o, i * bw + GAP, -ch / 2, bw - 2 * GAP, H + ch);
      }
    } else if (yl) {
      const cw = W / (nx - 1);
      for (let j = 0; j < ny; j++) {
        const o = strip(nx, 1, (k) => tint(at(k, j)));
        g.drawImage(o, -cw / 2, H - (j + 1) * bh + GAP, W + cw, bh - 2 * GAP);
      }
    }
  }, [values, W, H, nx, ny, p.binary, lo, hi, xl?.join(), yl?.join()]);

  // Where a band's prediction crosses 0.5 along its number axis: one front tick each
  const ticks: preact.JSX.Element[] = [];
  if (values && p.binary && (xl ? !yl : yl)) {
    const along = xl ? ny : nx;
    const bands = xl ? nx : ny;
    for (let b = 0; b < bands; b++) {
      const v = (s: number) => (xl ? values[s * nx + b]! : values[b * nx + s]!);
      for (let s = 0; s < along - 1; s++) {
        const a = v(s) - 0.5;
        const c = v(s + 1) - 0.5;
        if (a === 0 || a * c >= 0) continue;
        const t = s + a / (a - c);
        const warmUp = c > 0; // warm on the side the number axis grows towards
        const half = (xl ? bw : bh) * 0.4;
        if (xl) {
          const y = py(yr[0] + (t / (along - 1)) * (yr[1] - yr[0]));
          const x = px(b);
          const dir = warmUp ? -1 : 1; // screen y of the warm side
          ticks.push(
            <g class="band-front">
              <line class="front-warm" x1={x - half} x2={x} y1={y} y2={y} />
              <line class="front-cold" x1={x} x2={x + half} y1={y} y2={y} />
              {bw >= 26 && (
                <>
                  <path class="sym warm" d={`M${x - half * 0.55 - 5} ${y}A5 5 0 0 ${dir < 0 ? 1 : 0} ${x - half * 0.55 + 5} ${y}Z`} />
                  <path class="sym cold" d={`M${x + half * 0.55 - 5} ${y}L${x + half * 0.55 + 5} ${y}L${x + half * 0.55} ${y - dir * 8}Z`} />
                </>
              )}
            </g>,
          );
        } else {
          const x = px(xr[0] + (t / (along - 1)) * (xr[1] - xr[0]));
          const y = py(b);
          const dir = warmUp ? 1 : -1; // screen x of the warm side
          ticks.push(
            <g class="band-front">
              <line class="front-warm" x1={x} x2={x} y1={y - half} y2={y} />
              <line class="front-cold" x1={x} x2={x} y1={y} y2={y + half} />
              {bh >= 26 && (
                <>
                  <path class="sym warm" d={`M${x} ${y - half * 0.55 - 5}A5 5 0 0 ${dir > 0 ? 1 : 0} ${x} ${y - half * 0.55 + 5}Z`} />
                  <path class="sym cold" d={`M${x} ${y + half * 0.55 - 5}L${x} ${y + half * 0.55 + 5}L${x - dir * 8} ${y + half * 0.55}Z`} />
                </>
              )}
            </g>,
          );
        }
      }
    }
  }

  // Two category axes: every cell carries its value when there is room
  const cellText: preact.JSX.Element[] = [];
  if (values && xl && yl && bw >= 36 && bh >= 20) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const v = values[j * nx + i]!;
        cellText.push(
          <text class="cell-val" x={px(i)} y={py(j) + 4} text-anchor="middle">
            {p.binary ? (v < 0.005 ? "0" : v > 0.995 ? "1" : v.toFixed(2).replace(/^0/, "")) : fmt(v, 2)}
          </text>,
        );
      }
    }
  }

  const xt = xl ? [] : niceLevels(xr[0], xr[1], 7);
  const yt = yl ? [] : niceLevels(yr[0], yr[1], 5);

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
  const down = (e: PointerEvent) => {
    if (!p.onProbe) return;
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const raw = toRaw(e.clientX - r.left, e.clientY - r.top);
    lastProbe.current = performance.now();
    p.onProbe(raw[0], raw[1]);
  };
  const hv = hover && values ? sample(values, nx, ny, gx(hover[0]), gy(hover[1])) : null;
  const show = (v: number | string) => (typeof v === "number" ? fmt(v, 3) : v);

  const probe = p.probe
    ? ([xl ? bandOf(String(p.probe[0]), xl) : Number(p.probe[0]), yl ? bandOf(String(p.probe[1]), yl) : Number(p.probe[1])] as const)
    : null;
  const probeOk = probe && Number.isFinite(probe[0]) && Number.isFinite(probe[1]) && (!xl || probe[0] >= 0) && (!yl || probe[1] >= 0);

  return (
    <div class={`surface${p.onProbe ? " probing" : ""}`} ref={wrap} style={{ height: p.height }}>
      <canvas ref={canvas} style={{ left: M.l, top: M.t, width: W, height: H }} aria-hidden="true" />
      <svg width={width} height={p.height} role="img" aria-label={p.label} onPointerMove={move} onPointerLeave={() => setHover(null)} onPointerDown={down}>
        <g class="grid">
          {xt.map((t) => <line x1={px(t)} x2={px(t)} y1={M.t} y2={M.t + H} />)}
          {yt.map((t) => <line x1={M.l} x2={M.l + W} y1={py(t)} y2={py(t)} />)}
        </g>
        <g class="ticks">
          {xt.map((t) => <text x={px(t)} y={M.t + H + 16} text-anchor="middle">{fmt(t, 2)}</text>)}
          {yt.map((t) => <text x={M.l - 8} y={py(t) + 4} text-anchor="end">{fmt(t, 2)}</text>)}
          {xl?.map((l, i) =>
            rotate ? (
              <text x={px(i)} y={M.t + H + 12} text-anchor="end" transform={`rotate(-40 ${px(i)} ${M.t + H + 12})`}>
                {short(l)}
                <title>{l}</title>
              </text>
            ) : (
              <text x={px(i)} y={M.t + H + 16} text-anchor="middle">
                {short(l)}
                <title>{l}</title>
              </text>
            ),
          )}
          {yl?.map((l, j) => (
            <text x={M.l - 8} y={py(j) + 4} text-anchor="end">
              {short(l)}
              <title>{l}</title>
            </text>
          ))}
        </g>
        <text class="axis" x={M.l + W} y={M.t + H + M.b - 6} text-anchor="end">{p.xName} →</text>
        <text class="axis" x={M.l} y={M.t - 9}>↑ {p.yName}</text>
        {ticks}
        <g class="stations">
          {p.stations.map((s, k) => {
            const cx = px(xl ? s.x + jitterX(k) : s.x);
            const cy = py(yl ? s.y + jitterY(k) : s.y);
            return p.binary ? (
              <circle cx={cx} cy={cy} r={3.4} class={s.v >= 0.5 ? "one" : "zero"} />
            ) : (
              <circle cx={cx} cy={cy} r={3.4} class="val" fill={stationColour(s.v, p.vRange)} />
            );
          })}
        </g>
        {cellText}
        {probeOk && probe && (
          <g class="probe-mark" transform={`translate(${px(probe[0])} ${py(probe[1])})`}>
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
            <span>{p.xName} <b>{show(hover[0])}</b></span>
            <span>{p.yName} <b>{show(hover[1])}</b></span>
            {hv !== null && <span>{p.binary ? "p" : "ŷ"} <b>{p.binary ? fmtP(hv) : fmt(hv, 3)}</b></span>}
          </>
        ) : (
          <span class="hint">{readoutHint(!!p.onProbe)}</span>
        )}
      </div>
    </div>
  );
}
