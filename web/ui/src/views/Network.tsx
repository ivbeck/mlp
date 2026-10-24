import type { Activation, LayerSpec } from "../api";
import { activeRun, featureColumns, isLive, setDraft, startRun, type State } from "../store";
import { ACTIVATIONS, LOSSES, OPTIMIZERS, fmtInt, useWidth } from "../util";
import { DraftSettings } from "./Settings";

const MAX_DRAWN = 12;

export function NetworkView({ s }: { s: State }) {
  const d = s.draft;
  const cols = featureColumns(s).length;
  const inputs = s.encoding?.names.length ?? cols;
  const sizes = [inputs, ...d.hidden.map((h) => h.size), 1];
  const params = sizes.slice(1).reduce((n, out, i) => n + sizes[i]! * out + out, 0);
  const run = activeRun(s);
  const sameShape = run && run.summary.architecture.join() === sizes.join() && run.latest?.weights;
  const target = s.dataset && s.target !== null ? s.dataset.columns[s.target] : undefined;

  const setLayer = (i: number, patch: Partial<LayerSpec>) =>
    setDraft({ hidden: d.hidden.map((h, n) => (n === i ? { ...h, ...patch } : h)) });
  const move = (i: number, by: number) => {
    const next = d.hidden.slice();
    const [l] = next.splice(i, 1);
    next.splice(i + by, 0, l!);
    setDraft({ hidden: next });
  };

  const hints: string[] = [];
  if (d.loss === "bce" && target && !target.binary) hints.push("Binary cross-entropy expects 0/1 targets; this target is not.");
  if (d.loss === "bce" && d.output_activation !== "sigmoid") hints.push("Binary cross-entropy wants outputs in (0, 1). Use a sigmoid output.");
  if (target && !target.binary && d.output_activation === "sigmoid") hints.push("A sigmoid output can only reach 0 to 1, the target goes beyond that.");
  if (d.hidden.length === 0) hints.push("No hidden layer: this is a single neuron, it can only draw a straight boundary.");

  return (
    <>
      <section class="work">
        <div class="ttl">
          <h1>
            Network · <b>{fmtInt(params)} parameters</b>
          </h1>
          <span class="meta">
            {sameShape ? `edges show the weights of run ${run.summary.id.slice(1)} at epoch ${run.latest?.epoch}` : "untrained, edges show connections only"}
          </span>
        </div>
        <Diagram sizes={sizes} hidden={d.hidden} out={d.output_activation} weights={sameShape ? run.latest!.weights! : null} />
        <div class="layers" role="list" aria-label="Layers">
          <div class="layer fixed" role="listitem">
            <span class="ln">In</span>
            <span class="lname">Input</span>
            <span class="lsize">{inputs}</span>
            <span class="muted">
              {inputs === cols ? "one per feature column, set by the data" : `from ${cols} columns, categories one-hot (set on the Data tab)`}
            </span>
          </div>
          {d.hidden.map((h, i) => (
            <div class="layer" role="listitem">
              <span class="ln">{i + 1}</span>
              <span class="lname">Hidden</span>
              <label class="inline">
                <span class="visually-hidden">Neurons in hidden layer {i + 1}</span>
                <input
                  type="number"
                  min={1}
                  max={4096}
                  value={h.size}
                  onInput={(e) => {
                    const v = Math.round(+(e.currentTarget as HTMLInputElement).value);
                    if (v >= 1) setLayer(i, { size: Math.min(v, 4096) });
                  }}
                />
              </label>
              <label class="inline">
                <span class="visually-hidden">Activation of hidden layer {i + 1}</span>
                <select value={h.activation} onChange={(e) => setLayer(i, { activation: (e.currentTarget as HTMLSelectElement).value as Activation })}>
                  {ACTIVATIONS.map(([k, n]) => (
                    <option value={k}>{n}</option>
                  ))}
                </select>
              </label>
              <span class="spacer" />
              <button class="icon" disabled={i === 0} aria-label={`Move layer ${i + 1} up`} onClick={() => move(i, -1)}>
                <svg viewBox="0 0 16 16"><path d="M4 10l4-4 4 4" /></svg>
              </button>
              <button class="icon" disabled={i === d.hidden.length - 1} aria-label={`Move layer ${i + 1} down`} onClick={() => move(i, 1)}>
                <svg viewBox="0 0 16 16"><path d="M4 6l4 4 4-4" /></svg>
              </button>
              <button class="icon" aria-label={`Remove layer ${i + 1}`} onClick={() => setDraft({ hidden: d.hidden.filter((_, n) => n !== i) })}>
                <svg viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" /></svg>
              </button>
            </div>
          ))}
          <button
            class="add"
            onClick={() =>
              setDraft({ hidden: [...d.hidden, { size: d.hidden[d.hidden.length - 1]?.size ?? 8, activation: d.hidden[d.hidden.length - 1]?.activation ?? "relu" }] })
            }
          >
            Add hidden layer
          </button>
          <div class="layer fixed" role="listitem">
            <span class="ln">Out</span>
            <span class="lname">Output</span>
            <span class="lsize">1</span>
            <label class="inline">
              <span class="visually-hidden">Output activation</span>
              <select value={d.output_activation} onChange={(e) => setDraft({ output_activation: (e.currentTarget as HTMLSelectElement).value as Activation })}>
                {ACTIVATIONS.map(([k, n]) => (
                  <option value={k}>{n}</option>
                ))}
              </select>
            </label>
            <span class="muted">the crate trains a single output</span>
          </div>
        </div>
        {hints.length > 0 && (
          <ul class="hints">
            {hints.map((h) => (
              <li>{h}</li>
            ))}
          </ul>
        )}
      </section>
      <aside class="side">
        <DraftSettings s={s} />
        <div class="actions">
          <button class="primary" disabled={!s.dataset || s.target === null || !!s.busy} onClick={() => void startRun()}>
            {run && isLive(run) ? "Start another run" : "Train"}
          </button>
          <p class="note">
            {LOSSES.find(([k]) => k === d.loss)?.[1]}, {OPTIMIZERS.find(([k]) => k === d.optimizer)?.[1]} at {d.learning_rate}, {d.epochs} epochs.
          </p>
        </div>
      </aside>
    </>
  );
}

function Diagram({
  sizes,
  hidden,
  out,
  weights,
}: {
  sizes: number[];
  hidden: LayerSpec[];
  out: Activation;
  weights: number[][] | null;
}) {
  const [wrap, width] = useWidth<HTMLDivElement>();
  const H = 300;
  const top = 26;
  const bottom = 46;
  const cols = sizes.length;
  const xs = (i: number) => 60 + (i * (width - 120)) / Math.max(cols - 1, 1);
  // Big layers show their first and last neurons with a gap between
  const shown = sizes.map((n) => (n <= MAX_DRAWN ? [...Array(n).keys()] : [...[...Array(MAX_DRAWN - 2).keys()], n - 2, n - 1]));
  const ys = (layer: number, k: number) => {
    const count = shown[layer]!.length;
    const gap = Math.min(24, (H - top - bottom) / Math.max(count, 1));
    return top + (H - top - bottom) / 2 + (k - (count - 1) / 2) * gap;
  };
  let maxAbs = 0;
  weights?.forEach((l) => l.forEach((w) => (maxAbs = Math.max(maxAbs, Math.abs(w)))));

  const edges: preact.JSX.Element[] = [];
  for (let l = 0; l < cols - 1; l++) {
    const outN = sizes[l + 1]!;
    shown[l]!.forEach((i, a) =>
      shown[l + 1]!.forEach((j, b) => {
        const w = weights?.[l]?.[i * outN + j];
        const k = w !== undefined && maxAbs > 0 ? Math.abs(w) / maxAbs : 0;
        edges.push(
          <line
            class={w === undefined ? "edge" : w >= 0 ? "edge warm" : "edge cold"}
            x1={xs(l)}
            y1={ys(l, a)}
            x2={xs(l + 1)}
            y2={ys(l + 1, b)}
            stroke-width={w === undefined ? 0.6 : 0.4 + k * 2.6}
            stroke-opacity={w === undefined ? 0.5 : 0.12 + k * 0.85}
          />,
        );
      }),
    );
  }

  const name = (l: number) =>
    l === 0 ? "Input" : l === cols - 1 ? ACTIVATIONS.find(([k]) => k === out)?.[1] ?? out : ACTIVATIONS.find(([k]) => k === hidden[l - 1]?.activation)?.[1];

  return (
    <div class="diagram" ref={wrap}>
      {width > 0 && (
        <svg width={width} height={H} role="img" aria-label={`Network ${sizes.join(", ")}`}>
          {edges}
          {sizes.map((n, l) => (
            <g>
              {shown[l]!.map((_, k) => (
                <circle class={l === 0 ? "node in" : "node"} cx={xs(l)} cy={ys(l, k)} r={5} />
              ))}
              {n > MAX_DRAWN && (
                <text class="more" x={xs(l) + 10} y={ys(l, MAX_DRAWN - 2) - 8}>
                  +{n - MAX_DRAWN}
                </text>
              )}
              <text class="col-n" x={xs(l)} y={H - 24} text-anchor="middle">
                {n}
              </text>
              <text class="col-a" x={xs(l)} y={H - 8} text-anchor="middle">
                {name(l)}
              </text>
            </g>
          ))}
        </svg>
      )}
      {weights && (
        <div class="key-row">
          <span class="key"><i class="k-warm" />positive weight</span>
          <span class="key"><i class="k-cold" />negative weight</span>
          <span class="muted">line weight follows |w|</span>
        </div>
      )}
    </div>
  );
}
