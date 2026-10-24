import { useLayoutEffect, useRef, useState } from "preact/hooks";

/** Short number for charts and tables: fixed decimals, exponent for tiny or huge values */
export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
  const a = Math.abs(v);
  if (a !== 0 && (a < 10 ** -digits || a >= 1e6)) return v.toExponential(2).replace("-", "−");
  const s = v.toFixed(digits);
  // trim trailing zeros on tick style numbers only
  return (digits <= 2 ? String(+s) : s).replace("-", "−");
}

/** A probability: three decimals, the ends said in words instead of 0.000 or an exponent */
export function fmtP(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (v < 0.001) return "< 0.001";
  if (v > 0.999) return "> 0.999";
  return v.toFixed(3);
}

export function fmtInt(v: number): string {
  return v.toLocaleString("en-US");
}

/** Element width, kept current with a ResizeObserver */
export function useWidth<T extends HTMLElement>(): [preact.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.clientWidth);
    const ro = new ResizeObserver(([e]) => e && setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export const ACTIVATIONS = [
  ["relu", "ReLU"],
  ["leaky_relu", "Leaky ReLU"],
  ["tanh", "Tanh"],
  ["sigmoid", "Sigmoid"],
  ["linear", "Linear"],
] as const;

export const LOSSES = [
  ["bce", "Binary cross-entropy"],
  ["mse", "Mean squared error"],
  ["mae", "Mean absolute error"],
  ["huber", "Huber"],
] as const;

export const OPTIMIZERS = [
  ["adam", "Adam"],
  ["sgd", "SGD"],
  ["momentum", "SGD + momentum"],
  ["rmsprop", "RMSprop"],
] as const;

export const label = (list: readonly (readonly [string, string])[], key: string) =>
  list.find(([k]) => k === key)?.[1] ?? key;
