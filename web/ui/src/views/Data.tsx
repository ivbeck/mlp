import { useState } from "preact/hooks";
import { Surface } from "../charts/Surface";
import { featureColumns, loadExample, loadFile, setDraft, setHeader, setPlane, setTarget, type State, type View } from "../store";
import { bytes, fmt, fmtInt } from "../util";
import { PlaneSelect, stationsFor } from "./shared";

export function DataView({ s, go }: { s: State; go: (v: View) => void }) {
  if (!s.dataset) return <Empty busy={!!s.busy} />;
  const d = s.dataset;
  const feats = featureColumns(s);
  const target = s.target !== null ? d.columns[s.target] : undefined;
  const xc = feats[s.plane[0]];
  const yc = feats[s.plane[1]] ?? xc;
  const constant = feats.filter((c) => c.constant);

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
          <PlaneSelect feats={feats} plane={s.plane} onChange={(p) => void setPlane(p)} />
        </div>
        {xc && yc && target ? (
          <Surface
            field={null}
            stations={stationsFor(d, xc.index, yc.index, target.index)}
            binary={target.binary}
            xRange={[xc.min, xc.max]}
            yRange={[yc.min, yc.max]}
            vRange={[target.min, target.max]}
            xName={xc.name}
            yName={yc.name}
            height={420}
            label={`Scatter of ${xc.name} against ${yc.name}, coloured by ${target.name}`}
          />
        ) : (
          <p class="note">One feature only, there is no plane to draw. Training still works.</p>
        )}
        <div class="table-wrap">
          <table class="cols">
            <caption>Columns · pick the target the network should predict</caption>
            <thead>
              <tr>
                <th scope="col">Target</th>
                <th scope="col" class="num">#</th>
                <th scope="col">Name</th>
                <th scope="col" class="num">Min</th>
                <th scope="col" class="num">Max</th>
                <th scope="col" class="num">Mean</th>
                <th scope="col" class="num">Std</th>
                <th scope="col">Values</th>
              </tr>
            </thead>
            <tbody>
              {d.columns.map((c) => (
                <tr class={c.index === s.target ? "sel" : ""} onClick={() => setTarget(c.index)}>
                  <td>
                    <input
                      type="radio"
                      name="target"
                      checked={c.index === s.target}
                      aria-label={`Use ${c.name} as target`}
                      onChange={() => setTarget(c.index)}
                    />
                  </td>
                  <td class="num">{c.index}</td>
                  <td class="name">{c.name}</td>
                  <td class="num">{fmt(c.min, 3)}</td>
                  <td class="num">{fmt(c.max, 3)}</td>
                  <td class="num">{fmt(c.mean, 3)}</td>
                  <td class="num">{fmt(c.std, 3)}</td>
                  <td>
                    {c.constant ? (
                      <span class="flag warm">constant</span>
                    ) : c.binary ? (
                      <span class="flag">0 / 1</span>
                    ) : c.distinct.length ? (
                      <span class="muted">{c.distinct.length} distinct</span>
                    ) : (
                      <span class="muted">continuous</span>
                    )}
                  </td>
                </tr>
              ))}
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
          {target?.constant && <p class="note warn">{target.name} never changes, there is nothing to learn from it.</p>}
          {target && !target.constant && (
            <p class="note">
              {target.binary
                ? "Values are 0 and 1, so this trains as binary classification: sigmoid output, cross-entropy loss, accuracy at 0.5."
                : target.distinct.length > 2
                  ? `${target.distinct.length} distinct values. The crate trains one output neuron, so this runs as regression; it can not learn ${target.distinct.length} classes with softmax.`
                  : "Continuous values, so this trains as regression with a linear output."}
            </p>
          )}
        </section>
        <section>
          <div class="lab">Preparation</div>
          <label class="check">
            <input
              type="checkbox"
              checked={s.draft.normalize}
              onChange={(e) => setDraft({ normalize: (e.currentTarget as HTMLInputElement).checked })}
            />
            Normalize features to mean 0, std 1
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
          {constant.length > 0 && s.draft.normalize && (
            <p class="note warn">
              {constant.map((c) => c.name).join(", ")} {constant.length > 1 ? "are" : "is"} constant. The crate would divide by zero there; the console leaves {constant.length > 1 ? "them" : "it"} unscaled.
            </p>
          )}
        </section>
        <div class="actions">
          <button class="primary" disabled={s.target === null} onClick={() => go("network")}>
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
          <p class="note">Numbers only, one row per sample, an optional header row. Pick the target column after loading. The file stays on this machine.</p>
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
