// Marching squares over a row major res x res grid. Output polylines are in grid units
// (0..res-1 on both axes, y growing with the row index), joined end to end so a label or
// a front symbol can walk along them.

export type Pt = [number, number];

function key(p: Pt): string {
  return `${p[0].toFixed(4)},${p[1].toFixed(4)}`;
}

export function contour(values: ArrayLike<number>, res: number, level: number): Pt[][] {
  const segs: [Pt, Pt][] = [];
  const v = (i: number, j: number) => values[j * res + i]!;
  const lerp = (a: number, b: number) => (a === b ? 0.5 : (level - a) / (b - a));

  for (let j = 0; j < res - 1; j++) {
    for (let i = 0; i < res - 1; i++) {
      const a = v(i, j); // bottom left
      const b = v(i + 1, j); // bottom right
      const c = v(i + 1, j + 1); // top right
      const d = v(i, j + 1); // top left
      const code = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
      if (code === 0 || code === 15) continue;
      const bottom: Pt = [i + lerp(a, b), j];
      const right: Pt = [i + 1, j + lerp(b, c)];
      const top: Pt = [i + lerp(d, c), j + 1];
      const left: Pt = [i, j + lerp(a, d)];
      const centre = (a + b + c + d) / 4 > level;
      switch (code) {
        case 1: case 14: segs.push([left, bottom]); break;
        case 2: case 13: segs.push([bottom, right]); break;
        case 3: case 12: segs.push([left, right]); break;
        case 4: case 11: segs.push([right, top]); break;
        case 6: case 9: segs.push([bottom, top]); break;
        case 7: case 8: segs.push([left, top]); break;
        // saddles, the cell centre decides which corners connect
        case 5:
          if (centre) { segs.push([left, top]); segs.push([bottom, right]); }
          else { segs.push([left, bottom]); segs.push([right, top]); }
          break;
        case 10:
          if (centre) { segs.push([left, bottom]); segs.push([right, top]); }
          else { segs.push([left, top]); segs.push([bottom, right]); }
          break;
      }
    }
  }
  return join(segs);
}

function join(segs: [Pt, Pt][]): Pt[][] {
  const byEnd = new Map<string, number[]>();
  segs.forEach((s, n) => {
    for (const p of s) {
      const k = key(p);
      const list = byEnd.get(k);
      if (list) list.push(n);
      else byEnd.set(k, [n]);
    }
  });
  const used = new Uint8Array(segs.length);
  const lines: Pt[][] = [];

  const extend = (line: Pt[], atEnd: boolean) => {
    for (;;) {
      const tip = atEnd ? line[line.length - 1]! : line[0]!;
      const next = (byEnd.get(key(tip)) ?? []).find((n) => !used[n]);
      if (next === undefined) return;
      used[next] = 1;
      const [p, q] = segs[next]!;
      const other = key(p) === key(tip) ? q : p;
      if (atEnd) line.push(other);
      else line.unshift(other);
    }
  };

  segs.forEach((s, n) => {
    if (used[n]) return;
    used[n] = 1;
    const line: Pt[] = [s[0], s[1]];
    extend(line, true);
    extend(line, false);
    lines.push(line);
  });
  return lines;
}

export function length(line: Pt[]): number {
  let l = 0;
  for (let i = 1; i < line.length; i++) {
    l += Math.hypot(line[i]![0] - line[i - 1]![0], line[i]![1] - line[i - 1]![1]);
  }
  return l;
}

/** Points every `step` along a polyline with the unit tangent there */
export function walk(line: Pt[], step: number, offset = step / 2): { p: Pt; t: Pt }[] {
  const out: { p: Pt; t: Pt }[] = [];
  let want = offset;
  let at = 0;
  for (let i = 1; i < line.length; i++) {
    const a = line[i - 1]!;
    const b = line[i]!;
    const seg = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (seg === 0) continue;
    while (want <= at + seg) {
      const f = (want - at) / seg;
      out.push({
        p: [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f],
        t: [(b[0] - a[0]) / seg, (b[1] - a[1]) / seg],
      });
      want += step;
    }
    at += seg;
  }
  return out;
}

/** Bilinear sample of an nx by ny grid at a fractional grid position */
export function sample(values: ArrayLike<number>, nx: number, ny: number, gx: number, gy: number): number {
  const x = Math.min(Math.max(gx, 0), nx - 1);
  const y = Math.min(Math.max(gy, 0), ny - 1);
  const i = Math.max(0, Math.min(Math.floor(x), nx - 2));
  const j = Math.max(0, Math.min(Math.floor(y), ny - 2));
  const fx = nx > 1 ? x - i : 0;
  const fy = ny > 1 ? y - j : 0;
  const v = (a: number, b: number) => values[Math.min(b, ny - 1) * nx + Math.min(a, nx - 1)]!;
  return (
    v(i, j) * (1 - fx) * (1 - fy) +
    v(i + 1, j) * fx * (1 - fy) +
    v(i, j + 1) * (1 - fx) * fy +
    v(i + 1, j + 1) * fx * fy
  );
}

/** Round-ish contour levels across [lo, hi] for regression fields */
export function niceLevels(lo: number, hi: number, count = 6): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v < hi; v += step) out.push(+v.toPrecision(10));
  return out;
}
