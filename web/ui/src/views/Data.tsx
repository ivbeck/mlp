import { useState } from "preact/hooks";
import type { ColumnInfo, Kind } from "../api";
import { Surface } from "../charts/Surface";
import {
  featureColumns,
  kindOf,
  loadExample,
  loadFile,
  setDraft,
  setHeader,
  setKind,
  setPlane,
  setPositive,
  setTarget,
  type State,
  type View,
} from "../store";
import { bytes, fmtInt, fmtShort } from "../util";
import { KIND_TEXT, PlaneSelect, bandLabels, stationsFor, targetRange } from "./shared";

function summary(c: ColumnInfo, rows: number): string {
  if (c.min !== null && c.max !== null && c.categories.length > 2 && c.distinct > 12) {
    return `${fmtShort(c.min)} to ${fmtShort(c.max)}, mean ${fmtShort(c.mean)}`;
  }
  const shown = c.categories.slice(0, 3).map(([v, n]) => `${v} ${Math.round((n / rows) * 100)}%`);
  const rest = c.distinct - shown.length;
  return shown.join(", ") + (rest > 0 ? `, ${rest} more` : "");
}

export function DataView({ s, go }: { s: State; go: (v: View) => void }) {
  if (!s.dataset) return <Empty busy={!!s.busy} />;
  const d = s.dataset;
  const axisCols = featureColumns(s)
    .filter((c) => ["numeric", "category", "binary"].includes(kindOf(c, s)))
    .map((c) => ({ index: c.index, name: c.name, cat: kindOf(c, s) !== "numeric" }));
  const target = s.target !== null ? d.columns[s.target] : undefined;
  const spec = s.encoding?.target ?? null;
  const xc = d.columns[s.plane[0]];
  const yc = d.columns[s.plane[1]];
  const canPlot = !!(xc && yc && target && xc !== yc && axisCols.some((a) => a.index === xc.index) && axisCols.some((a) => a.index === yc.index));
  const xLabels = xc ? bandLabels(xc, kindOf(xc, s)) : null;
  const yLabels = yc ? bandLabels(yc, kindOf(yc, s)) : null;
  const enc = s.encoding;
  const wide = enc?.inputs.filter((i) => i.width > 1) ?? [];

  return (
    <>
      <section class="work">
        <div class="ttl">
          <h1>
            Stations · <b>{d.name}</b>
          </h1>
          <span class="meta">
            {d.rows > d.sample.length ? `first ${fmtInt(d.sample.length)} of ${fmtInt(d.rows)} rows plotted` : `${fmtInt(d.rows)} rows plotted`}
          </span>
          <PlaneSelect feats={axisCols} plane={s.plane} onChange={(p) => void setPlane(p)} />
        </div>
        {canPlot ? (
          <Surface
            field={null}
            stations={stationsFor(d, { col: xc.index, labels: xLabels }, { col: yc.index, labels: yLabels }, target.index, spec)}
            xLabels={xLabels}
            yLabels={yLabels}
            binary={spec?.kind === "binary"}
            xRange={[xc.min ?? 0, xc.max ?? 1]}
            yRange={[yc.min ?? 0, yc.max ?? 1]}
            vRange={targetRange(d, target.index, spec)}
            xName={xc.name}
            yName={yc.name}
            height={420}
            label={`Scatter of ${xc.name} against ${yc.name}, coloured by ${target.name}`}
          />
        ) : (
          <div class="empty">
            <p>The chart needs two columns that are numbers or categories. Training works without it.</p>
          </div>
        )}
        <div class="table-wrap">
          <table class="cols">
            <caption>Columns · pick the target, set how each column is read</caption>
            <thead>
              <tr>
                <th scope="col">Target</th>
                <th scope="col">Name</th>
                <th scope="col">Used as</th>
                <th scope="col" class="num">Missing</th>
                <th scope="col" class="num">Values</th>
                <th scope="col">Summary</th>
              </tr>
            </thead>
            <tbody>
              {d.columns.map((c) => {
                const isTarget = c.index === s.target;
                const kind = kindOf(c, s);
                return (
                  <tr class={`${isTarget ? "sel" : ""}${!isTarget && kind === "ignore" ? " off" : ""}`}>
                    <td>
                      <input
                        type="radio"
                        name="target"
                        checked={isTarget}
                        aria-label={`Use ${c.name} as target`}
                        onChange={() => setTarget(c.index)}
                      />
                    </td>
                    <td class="name">{c.name}</td>
                    <td>
                      {isTarget ? (
                        <span class="flag warm">target</span>
                      ) : (
                        <select
                          class="kind"
                          aria-label={`How ${c.name} is read`}
                          value={kind}
                          onChange={(e) => setKind(c.index, (e.currentTarget as HTMLSelectElement).value as Kind)}
                        >
                          {c.allowed.map((k) => (
                            <option value={k}>{KIND_TEXT[k]}</option>
                          ))}
                        </select>
                      )}
                    </td>
                    <td class="num">{c.missing ? fmtInt(c.missing) : <span class="muted">none</span>}</td>
                    <td class="num">{fmtInt(c.distinct)}</td>
                    <td class="sum">
                      {summary(c, d.rows - c.missing)}
                      {c.reason && !isTarget && kind === "ignore" && <span class="muted"> · {c.reason}</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
      <aside class="side">
        <section>
          <div class="lab">File</div>
          <dl class="facts">
            <dt>Rows</dt>
            <dd>{fmtInt(d.rows)}</dd>
            <dt>Columns</dt>
            <dd>{d.columns.length}</dd>
            <dt>Size</dt>
            <dd>{bytes(d.size_bytes)}</dd>
          </dl>
          <label class="check">
            <input
              type="checkbox"
              checked={d.has_header}
              disabled={d.is_sample || !s.file}
              onChange={(e) => void setHeader((e.currentTarget as HTMLInputElement).checked)}
            />
            First row is a header
          </label>
          <FilePick label="Choose another file" />
        </section>
        <section>
          <div class="lab">Target</div>
          <label class="field">
            Column the network predicts
            <select value={String(s.target ?? "")} onChange={(e) => setTarget(+(e.currentTarget as HTMLSelectElement).value)}>
              {d.columns.map((c) => (
                <option value={c.index}>{c.name}</option>
              ))}
            </select>
          </label>
          {spec?.kind === "binary" && target && (
            <label class="field">
              Counts as class 1
              <select value={spec.one} onChange={(e) => setPositive((e.currentTarget as HTMLSelectElement).value)}>
                {[spec.one, spec.zero].sort().map((v) => (
                  <option value={v}>{v}</option>
                ))}
              </select>
            </label>
          )}
          {target?.constant && <p class="note warn">{target.name} never changes, there is nothing to learn from it.</p>}
          {enc?.error ? (
            <p class="note warn">{enc.error}</p>
          ) : (
            spec && (
              <p class="note">
                {spec.kind === "binary"
                  ? `Two classes: ${target?.name} = ${spec.one} against ${spec.zero}. Sigmoid output, cross-entropy loss, accuracy at 0.5.`
                  : "Numbers, so this trains as regression with a linear output."}
              </p>
            )
          )}
        </section>
        {enc && !enc.error && (
          <section>
            <div class="lab">Network inputs</div>
            <p class="net">
              {enc.inputs.length} columns → {enc.names.length} inputs
            </p>
            {wide.length > 0 && (
              <p class="note">
                Categories become one 0/1 input per value (top 30 per column, the rest share one), missing values get their own.{" "}
                {wide.map((i) => `${i.name} ${i.width}`).join(", ")}.
              </p>
            )}
          </section>
        )}
        <section>
          <div class="lab">Preparation</div>
          <label class="check">
            <input
              type="checkbox"
              checked={s.draft.normalize}
              onChange={(e) => setDraft({ normalize: (e.currentTarget as HTMLInputElement).checked })}
            />
            Normalize inputs to mean 0, std 1
          </label>
          <label class="check">
            <input
              type="checkbox"
              checked={s.draft.shuffle_before_split}
              onChange={(e) => setDraft({ shuffle_before_split: (e.currentTarget as HTMLInputElement).checked })}
            />
            Shuffle rows before the validation split
          </label>
          <p class="note">
            Without it the last {Math.round(s.draft.validation_split * 100)}% of rows in file order become validation, as in the CLI.
          </p>
        </section>
        <div class="actions">
          <button class="primary" disabled={s.target === null || !enc || !!enc.error} onClick={() => go("network")}>
            Shape the network
          </button>
        </div>
      </aside>
    </>
  );
}

function FilePick({ label, primary }: { label: string; primary?: boolean }) {
  return (
    <label class={primary ? "button primary" : "button"}>
      {label}
      <input
        type="file"
        accept=".csv,text/csv,text/plain"
        class="visually-hidden"
        onChange={(e) => {
          const f = (e.currentTarget as HTMLInputElement).files?.[0];
          if (f) void loadFile(f);
        }}
      />
    </label>
  );
}

function Empty({ busy }: { busy: boolean }) {
  const [over, setOver] = useState(false);
  return (
    <>
      <section class="work">
        <div class="ttl">
          <h1>Load a dataset</h1>
        </div>
        <div
          class={`drop${over ? " over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const f = e.dataTransfer?.files[0];
            if (f) void loadFile(f);
          }}
        >
          <svg class="drop-art" viewBox="0 0 240 120" aria-hidden="true">
            <path d="M10 92C50 90 70 40 120 44S190 96 230 70" />
            <path d="M10 70C46 66 72 22 120 24S192 74 230 46" />
            <path class="f" d="M10 110C56 112 76 62 120 64S186 112 230 96" />
            <circle cx="62" cy="40" r="4" />
            <circle cx="94" cy="66" r="4" class="o" />
            <circle cx="150" cy="30" r="4" />
            <circle cx="178" cy="84" r="4" class="o" />
          </svg>
          <p class="drop-lead">Drop a CSV here</p>
          <p class="note">Numbers, categories, yes/no columns and ISO dates, one row per sample, an optional header row. Pick the target and how each column is read after loading. The file stays on this machine.</p>
          <div class="row">
            <FilePick label="Choose a file" primary />
            <button disabled={busy} onClick={() => void loadExample()}>
              Use example data
            </button>
          </div>
        </div>
      </section>
      <aside class="side">
        <section>
          <div class="lab">How it goes</div>
          <ol class="steps">
            <li><b>Data.</b> Load a file, choose the target column.</li>
            <li><b>Network.</b> Layers, activations, loss, optimizer.</li>
            <li><b>Train.</b> Watch the field form over two features; pause, step, scrub back.</li>
            <li><b>Probe.</b> Click any point and read every layer.</li>
          </ol>
        </section>
        <section>
          <div class="lab">Example data</div>
          <p class="note">example_data.csv from the repository: 20 rows, two features, a 0/1 target.</p>
        </section>
      </aside>
    </>
  );
}
