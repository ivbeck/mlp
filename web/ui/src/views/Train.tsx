import { Barograph } from "../charts/Barograph";
import { Surface } from "../charts/Surface";
import { activeRun, control, isLive, selectRun, set, setPlane, startRun, type State, type View } from "../store";
import { LOSSES, OPTIMIZERS, fmt, fmtInt, label } from "../util";
import { DraftSettings } from "./Settings";
import { ChartKey, PlaneSelect, STATE_TEXT, archLine, stationsFor, targetRange } from "./shared";

const LOSS_SHORT: Record<string, string> = { bce: "BCE", mse: "MSE", mae: "MAE", huber: "Huber" };

export function TrainView({ s, go }: { s: State; go: (v: View) => void }) {
  const run = activeRun(s);
  if (!run) {
    return (
      <>
        <section class="work">
          <div class="ttl">
            <h1>No run yet</h1>
          </div>
          <div class="empty">
            <p>Train a network to see its field form over two features, epoch by epoch.</p>
            <button onClick={() => go("network")}>Check the network first</button>
          </div>
        </section>
        <aside class="side">
          <DraftSettings s={s} />
          <div class="actions">
            <button class="primary" disabled={!s.dataset || s.target === null} onClick={() => void startRun()}>
              Train
            </button>
          </div>
        </aside>
      </>
    );
  }

  const sum = run.summary;
  const c = sum.config;
  const live = isLive(run);
  const last = run.history[run.history.length - 1];
  const frame = s.scrub !== null ? run.frames.find((f) => f.epoch === s.scrub) ?? run.latest : run.latest;
  const field = frame?.field ?? null;
  const shownEpoch = s.scrub ?? last?.epoch ?? 0;
  const rec = run.history.find((r) => r.epoch === shownEpoch) ?? last;
  const nums = sum.inputs.filter((i) => i.kind === "numeric");
  const xc = field ? sum.inputs[field.x] : undefined;
  const yc = field ? sum.inputs[field.y] : undefined;
  const sameData = !!s.dataset && s.dataset.id === c.dataset_id;
  const plane: [number, number] = xc && yc ? [xc.index, yc.index] : s.plane;
  const vRange: [number, number] = sameData ? targetRange(s.dataset!, c.target, sum.target) : [0, 1];
  const others = s.runs.filter((r) => r !== run && r.summary.config.loss === c.loss && r.summary.config.dataset_id === c.dataset_id);
  const progress = last ? last.epoch / c.epochs : 0;
  const stations = field && xc && yc && sameData ? stationsFor(s.dataset!, xc.index, yc.index, c.target, sum.target) : [];
  const controls = live ? (
    <div class="btns">
      {sum.state === "running" ? (
        <button class="primary" onClick={() => void control("pause")}>Pause</button>
      ) : (
        <button class="primary" onClick={() => void control("resume")}>Resume</button>
      )}
      <button onClick={() => void control("step")}>Step</button>
      <button class="danger" onClick={() => void control("stop")}>Stop</button>
    </div>
  ) : (
    <div class="btns two">
      <button class="primary" disabled={!s.dataset || !!s.busy} onClick={() => void startRun()}>Train again</button>
      <button onClick={() => go("probe")} disabled={!run.history.length}>Probe</button>
    </div>
  );

  return (
    <>
      <section class="work">
        <div class="run-strip" aria-label="Run controls">
          <span class="rs-ep">
            <b>{last?.epoch ?? 0}</b> / {c.epochs}
            <span class={`state-${sum.state}`}> · {STATE_TEXT[sum.state]}</span>
          </span>
          {controls}
        </div>
        <div class="ttl">
          <h1>
            Surface analysis · <b>epoch {shownEpoch}</b>
          </h1>
          <span class="meta">
            {sum.target.kind === "binary" ? `p(${sum.target_name} = ${sum.target.one})` : `predicted ${sum.target_name}`}
            {xc && yc && ` over ${xc.name}, ${yc.name}`}
            {sum.inputs.length > 2 && " · other numbers at their mean, categories at their most common value"}
          </span>
          <PlaneSelect feats={nums} plane={plane} onChange={(p) => void setPlane(p)} />
        </div>
        {field ? (
          <Surface
            field={field}
            stations={stations}
            binary={sum.binary_target}
            xRange={field.x_range}
            yRange={field.y_range}
            vRange={vRange}
            xName={xc?.name ?? "x"}
            yName={yc?.name ?? "y"}
            height={470}
            label={`Prediction field at epoch ${shownEpoch}`}
          />
        ) : (
          <div class="empty tall">
            <p>
              {nums.length < 2
                ? "The chart needs two inputs used as numbers. The barograph below still tracks the run."
                : "Waiting for the first field from the server."}
            </p>
          </div>
        )}
        <Barograph
          history={run.history}
          others={others.map((o) => ({ id: o.summary.id, history: o.history }))}
          epochs={c.epochs}
          height={180}
          frames={run.frames.map((f) => f.epoch)}
          scrub={s.scrub}
          onScrub={(scrub) => set({ scrub })}
          lossName={label(LOSSES, c.loss)}
        />
      </section>
      <aside class="side">
        <section>
          <div class="ep side-counter">
            {last?.epoch ?? 0}
            <small> / {c.epochs}</small>
          </div>
          <div class="bar side-counter" role="progressbar" aria-valuemin={0} aria-valuemax={c.epochs} aria-valuenow={last?.epoch ?? 0}>
            <i style={{ width: `${progress * 100}%` }} />
          </div>
          <p class={`state-line state-${sum.state}`}>
            Run {sum.id.slice(1)} · {STATE_TEXT[sum.state]}
            {sum.reason && <span class={sum.state === "failed" ? "warn" : ""}>: {sum.reason}</span>}
          </p>
        </section>
        <section class="losses">
          <div>
            <span class="lab">Train loss</span>
            <strong>{fmt(rec?.train_loss)}</strong>
            {rec?.train_acc !== null && rec?.train_acc !== undefined && <span class="acc">accuracy {fmt(rec.train_acc * 100, 1)} %</span>}
          </div>
          <div class="v">
            <span class="lab">Val loss</span>
            <strong>{rec?.val_loss !== null && rec?.val_loss !== undefined ? fmt(rec.val_loss) : "no split"}</strong>
            {rec?.val_acc !== null && rec?.val_acc !== undefined && <span class="acc">accuracy {fmt(rec.val_acc * 100, 1)} %</span>}
          </div>
        </section>
        <div class="side-controls">{controls}</div>
        {live && <p class="note kbd-hint">Space pauses and resumes, the period key steps one epoch.</p>}
        <section>
          <div class="lab">Network</div>
          <p class="net">{archLine(sum)}</p>
          <p class="muted">
            {fmtInt(sum.parameters)} parameters · {label(LOSSES, c.loss)} · {label(OPTIMIZERS, c.optimizer)} at {c.learning_rate} · batch {c.batch_size}
          </p>
          <p class="muted">
            {fmtInt(sum.train_rows)} training rows, {fmtInt(sum.val_rows)} validation rows{c.normalize ? ", normalized" : ""}
          </p>
          {sum.notes.map((n) => (
            <p class={n.level === "warn" ? "note warn" : "note"}>{n.text}</p>
          ))}
        </section>
        {field && <ChartKey spec={sum.target} stations={stations} vRange={vRange} targetName={sum.target_name} />}
        {s.runs.length > 1 && (
          <section>
            <div class="lab">Runs</div>
            <ul class="runs">
              {[...s.runs].reverse().map((r) => {
                const l = r.history[r.history.length - 1];
                return (
                  <li>
                    <button class={r === run ? "on" : ""} aria-current={r === run} onClick={() => selectRun(r.summary.id)}>
                      <span class="rn">Run {r.summary.id.slice(1)}</span>
                      <span class="rd">{r.summary.dataset_name} · {r.summary.architecture.join("-")} · {label(OPTIMIZERS, r.summary.config.optimizer)} {r.summary.config.learning_rate}</span>
                      <span class="rv">
                        {fmt(l?.val_loss ?? l?.train_loss)}
                        <span class="muted"> {LOSS_SHORT[r.summary.config.loss]}</span>
                      </span>
                      <span class={`rs state-${r.summary.state}`}>{STATE_TEXT[r.summary.state]}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        )}
        <DraftSettings s={s} compact />
      </aside>
    </>
  );
}
