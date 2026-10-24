import { useEffect } from "preact/hooks";
import { Surface } from "../charts/Surface";
import { activeRun, probe, setPlane, type State } from "../store";
import { ACTIVATIONS, fmt, fmtP, label, useWidth } from "../util";
import { ChartKey, KIND_TEXT, PlaneSelect, axisInputs, fieldAxes, stationsFor, targetRange, targetValue } from "./shared";

export function ProbeView({ s }: { s: State }) {
  const run = activeRun(s)!;
  const sum = run.summary;
  const c = sum.config;
  const defaults = sum.inputs.map((i) => i.default);
  const inputs = s.probe?.inputs ?? defaults;
  const result = s.probe?.result ?? null;
  const field = run.latest?.field ?? null;
  const xc = field ? sum.inputs[field.x] : undefined;
  const yc = field ? sum.inputs[field.y] : undefined;
  const sameData = !!s.dataset && s.dataset.id === c.dataset_id;
  const vRange: [number, number] = sameData ? targetRange(s.dataset!, c.target, sum.target) : [0, 1];
  const rows = sameData ? s.dataset!.sample.slice(0, 200) : [];
  const nums = axisInputs(sum.inputs);
  const axes = field ? fieldAxes(field, sum.inputs) : null;
  const stations = axes && sameData ? stationsFor(s.dataset!, axes[0], axes[1], c.target, sum.target) : [];
  const num = (v: number | string | undefined) => (typeof v === "number" ? v : Number(v));
  const out = result?.output ?? null;

  useEffect(() => {
    if (!s.probe) void probe(defaults.slice());
  }, [sum.id]);

  const setInput = (i: number, v: number | string) => {
    const next = inputs.slice();
    next[i] = v;
    void probe(next);
  };

  return (
    <>
      <section class="work">
        <div class="ttl">
          <h1>
            Probe · <b>run {sum.id.slice(1)}, epoch {run.latest?.epoch ?? 0}</b>
          </h1>
          <span class="meta">
            {sum.state === "running"
              ? "the run is still training, the field keeps moving"
              : window.matchMedia("(pointer: coarse)").matches
                ? "tap or drag on the chart"
                : "click or drag on the chart"}
          </span>
          <PlaneSelect feats={nums} plane={xc && yc ? [xc.index, yc.index] : s.plane} onChange={(p) => void setPlane(p)} />
        </div>
        {field && (
          <Surface
            field={field}
            stations={stations}
            binary={sum.binary_target}
            xRange={field.x_range ?? [0, 1]}
            yRange={field.y_range ?? [0, 1]}
            vRange={vRange}
            xName={xc?.name ?? "x"}
            yName={yc?.name ?? "y"}
            height={420}
            probe={[
              field.x_labels ? String(inputs[field.x] ?? "") : num(inputs[field.x]),
              field.y_labels ? String(inputs[field.y] ?? "") : num(inputs[field.y]),
            ]}
            onProbe={(x, y) => {
              const next = inputs.slice();
              next[field.x] = x;
              next[field.y] = y;
              void probe(next);
            }}
            label="Prediction field, click to probe a point"
          />
        )}
        <Activations
          layers={result?.layers ?? null}
          sizes={sum.architecture}
          names={[...c.hidden.map((h) => h.activation), c.output_activation]}
          outRange={vRange}
          mid={sum.binary_target ? 0.5 : null}
        />
      </section>
      <aside class="side">
        <section class="verdict">
          <div class="ep">
            <span class="ep-name">{sum.target.kind === "binary" ? `p(${sum.target_name} = ${sum.target.one})` : sum.target_name}</span>
            {sum.binary_target ? fmtP(out) : fmt(out, 4)}
          </div>
          {sum.target.kind === "binary" && out !== null && (
            <>
              <div class="meter" aria-hidden="true">
                <i style={{ left: `${Math.min(1, Math.max(0, out)) * 100}%` }} />
              </div>
              <p class="net">Predicts {sum.target_name} = {out >= 0.5 ? sum.target.one : sum.target.zero}</p>
            </>
          )}
          {s.probe?.row !== null && s.probe?.row !== undefined && (
            <p class="muted">
              Row {s.probe.row}: actual {sum.target_name} = {rows[s.probe.row]?.[c.target] ?? "–"}
            </p>
          )}
        </section>
        {rows.length > 0 && (
          <label class="field">
            Fill from a data row
            <select
              value={String(s.probe?.row ?? "")}
              onChange={(e) => {
                const v = (e.currentTarget as HTMLSelectElement).value;
                if (v === "") return;
                const row = rows[+v]!;
                void probe(sum.inputs.map((inp) => row[inp.index] ?? ""), +v);
              }}
            >
              <option value="">Choose a row</option>
              {rows.map((r, i) => (
                <option value={i}>
                  Row {i} · {sum.target_name} {sum.target.kind === "binary" ? r[c.target] : fmt(targetValue(r[c.target], sum.target), 3)}
                </option>
              ))}
            </select>
          </label>
        )}
        <section>
          <div class="lab">Inputs, in file units</div>
          <div class="inputs">
            {sum.inputs.map((inp, i) => {
              const v = inputs[i];
              return (
                <label class="field">
                  <span class="in-name">
                    {inp.name}
                    <span class="muted">
                      {inp.kind === "numeric" && inp.min !== null && inp.max !== null
                        ? `${fmt(inp.min, 2)} to ${fmt(inp.max, 2)}`
                        : inp.kind === "date"
                          ? "YYYY-MM-DD"
                          : `${KIND_TEXT[inp.kind].toLowerCase()}, ${inp.options.length} values`}
                    </span>
                  </span>
                  {inp.kind === "numeric" ? (
                    <input
                      type="number"
                      step="any"
                      value={Number.isFinite(num(v)) ? +num(v).toPrecision(6) : ""}
                      onChange={(e) => setInput(i, +(e.currentTarget as HTMLInputElement).value)}
                    />
                  ) : inp.kind === "date" ? (
                    <input type="text" inputMode="numeric" value={String(v ?? "")} onChange={(e) => setInput(i, (e.currentTarget as HTMLInputElement).value)} />
                  ) : (
                    <select value={String(v ?? "")} onChange={(e) => setInput(i, (e.currentTarget as HTMLSelectElement).value)}>
                      {!inp.options.includes(String(v ?? "")) && <option value={String(v ?? "")}>{String(v ?? "") || "(empty)"}</option>}
                      {inp.options.map((o) => (
                        <option value={o}>{o}</option>
                      ))}
                    </select>
                  )}
                </label>
              );
            })}
          </div>
          <button class="link" onClick={() => void probe(defaults.slice())}>
            Reset to means and most common values
          </button>
        </section>
        {field && (
          <ChartKey
            banded={!!(field.x_labels || field.y_labels)}
            spec={sum.target}
            stations={stations}
            vRange={vRange}
            targetName={sum.target_name}
          />
        )}
        <p class="note">Hidden layer bars are scaled per layer, the output sits on the target's scale. Hidden layers use {[...new Set(c.hidden.map((h) => label(ACTIVATIONS, h.activation)))].join(", ") || "no activation"}; bars show each neuron's output for this input.</p>
      </aside>
    </>
  );
}

function Activations({
  layers,
  sizes,
  names,
  outRange,
  mid,
}: {
  layers: number[][] | null;
  sizes: number[];
  names: string[];
  outRange: [number, number];
  mid: number | null;
}) {
  const [wrap, width] = useWidth<HTMLDivElement>();
  if (!layers) return <div class="acts" ref={wrap} />;
  return (
    <div class="acts" ref={wrap}>
      <h2>Layer outputs</h2>
      {layers.map((a, l) => {
        if (l === layers.length - 1) {
          // The output on the target's own scale, never scaled to its own max
          const v = a[0] ?? 0;
          const [lo, hi] = outRange;
          const span = hi > lo ? hi - lo : 1;
          const k = (x: number) => Math.min(1, Math.max(0, (x - lo) / span));
          const w = Math.max(width - 170, 10);
          const warm = mid !== null ? v >= mid : k(v) >= 0.5;
          return (
            <div class="act-row">
              <div class="act-name">
                Output
                <span class="muted">1 · {label(ACTIVATIONS, names[l] ?? "")}</span>
              </div>
              <svg width={w} height={44} role="img" aria-label={`Output ${fmt(v, 4)} on a scale from ${fmt(lo, 3)} to ${fmt(hi, 3)}`}>
                <line class="zero" x1={0} x2={w} y1={22} y2={22} />
                {mid !== null && <line class="midline" x1={k(mid) * w} x2={k(mid) * w} y1={12} y2={32} />}
                <rect
                  class={warm ? "warm" : "cold"}
                  x={Math.min(k(v), k(mid ?? lo)) * w}
                  y={17}
                  width={Math.max(Math.abs(k(v) - k(mid ?? lo)) * w, 1.5)}
                  height={10}
                />
                <circle class="out-mark" cx={k(v) * w} cy={22} r={6} />
                <text class="scale" x={0} y={42}>{fmt(lo, 2)}</text>
                <text class="scale" x={w} y={42} text-anchor="end">{fmt(hi, 2)}</text>
              </svg>
            </div>
          );
        }
        const max = Math.max(1e-9, ...a.map(Math.abs));
        const shownN = Math.min(a.length, Math.floor((width - 170) / 7));
        const bw = Math.max(3, Math.min(22, (width - 170) / Math.max(shownN, 1) - 2));
        return (
          <div class="act-row">
            <div class="act-name">
              {l === layers.length - 1 ? "Output" : `Hidden ${l + 1}`}
              <span class="muted">
                {sizes[l + 1]} · {label(ACTIVATIONS, names[l] ?? "")}
              </span>
            </div>
            <svg width={Math.max(width - 170, 10)} height={44} role="img" aria-label={`Outputs of layer ${l + 1}: ${a.slice(0, 16).map((v) => fmt(v, 2)).join(", ")}`}>
              <line class="zero" x1={0} x2={width} y1={22} y2={22} />
              {a.slice(0, shownN).map((v, i) => {
                const h = (Math.abs(v) / max) * 20;
                return (
                  <rect class={v >= 0 ? "warm" : "cold"} x={i * (bw + 2)} y={v >= 0 ? 22 - h : 22} width={bw} height={Math.max(h, 0.8)}>
                    <title>
                      neuron {i + 1}: {fmt(v, 4)}
                    </title>
                  </rect>
                );
              })}
            </svg>
          </div>
        );
      })}
    </div>
  );
}
