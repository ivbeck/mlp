// Shared pieces of the chart paper: palette for the field tint, the eased field, padding
import { useEffect, useRef, useState } from "preact/hooks";
import type { Field } from "../api";

export const COLD = [36, 89, 168];
export const WARM = [210, 54, 43];
export const PAPER = [233, 238, 242];
const TWEEN_MS = 160;

export const easeOut = (t: number) => 1 - Math.pow(2, -10 * t);

export function mix(a: number[], b: number[], k: number) {
  return a.map((v, i) => Math.round(v + (b[i]! - v) * k));
}

export function pad([lo, hi]: [number, number]): [number, number] {
  const span = hi > lo ? hi - lo : 1;
  return [lo - span * 0.08, hi + span * 0.08];
}

/** Regression stations: cold below the middle of the target range, warm above */
export function stationColour(v: number, [lo, hi]: [number, number]): string {
  const k = hi > lo ? (v - lo) / (hi - lo) : 0.5;
  const c = mix(k < 0.5 ? COLD : PAPER, k < 0.5 ? PAPER : WARM, k < 0.5 ? k * 2 : (k - 0.5) * 2);
  return `rgb(${c.join(",")})`;
}

/** The field shown on screen, eased towards each new one so fronts move instead of jump */
export function useTween(field: Field | null): Float64Array | null {
  const [shown, setShown] = useState<Float64Array | null>(null);
  const from = useRef<Float64Array | null>(null);
  const plane = useRef("");

  useEffect(() => {
    if (!field) {
      from.current = null;
      setShown(null);
      return;
    }
    const target = Float64Array.from(field.values);
    const id = `${field.x}:${field.y}:${field.nx}:${field.ny}`;
    const start = from.current;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!start || start.length !== target.length || plane.current !== id || reduce) {
      plane.current = id;
      from.current = target;
      setShown(target);
      return;
    }
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const k = easeOut(Math.min(1, (now - t0) / TWEEN_MS));
      const cur = new Float64Array(target.length);
      for (let i = 0; i < cur.length; i++) cur[i] = start[i]! + (target[i]! - start[i]!) * k;
      from.current = cur;
      setShown(cur);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [field]);

  return shown;
}

export function readoutHint(probing: boolean): string {
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  if (coarse) return probing ? "Tap or drag on the chart to probe a point" : "Tap the chart to read the field";
  return probing ? "Click or drag on the chart to probe a point" : "Point at the chart to read the field";
}
