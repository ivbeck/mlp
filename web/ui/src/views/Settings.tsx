import type { LossName, OptimizerName } from "../api";
import { setDraft, type State } from "../store";
import { LOSSES, OPTIMIZERS } from "../util";

function num(e: Event): number {
  return +(e.currentTarget as HTMLInputElement).value;
}

/** The next run's training settings. Every CLI option that is not about layers lives here. */
export function DraftSettings({ s, compact, disabled }: { s: State; compact?: boolean; disabled?: boolean }) {
  const d = s.draft;
  return (
    <section>
      <div class="lab">{compact ? "Next run" : "Training"}</div>
      <fieldset class="form" disabled={disabled}>
        <label class="field wide">
          Loss
          <select value={d.loss} onChange={(e) => setDraft({ loss: (e.currentTarget as HTMLSelectElement).value as LossName })}>
            {LOSSES.map(([k, n]) => (
              <option value={k}>{n}</option>
            ))}
          </select>
        </label>
        <label class="field">
          Optimizer
          <select value={d.optimizer} onChange={(e) => setDraft({ optimizer: (e.currentTarget as HTMLSelectElement).value as OptimizerName })}>
            {OPTIMIZERS.map(([k, n]) => (
              <option value={k}>{n}</option>
            ))}
          </select>
        </label>
        <label class="field">
          Learning rate
          <input type="number" step="any" min={0} value={d.learning_rate} onChange={(e) => num(e) > 0 && setDraft({ learning_rate: num(e) })} />
        </label>
        {d.optimizer === "momentum" ? (
          <label class="field">
            Momentum
            <input type="number" step={0.05} min={0} max={0.999} value={d.momentum} onChange={(e) => setDraft({ momentum: Math.min(0.999, Math.max(0, num(e))) })} />
          </label>
        ) : (
          <label class="field">
            Batch size
            <input type="number" min={1} value={d.batch_size} onChange={(e) => num(e) >= 1 && setDraft({ batch_size: Math.round(num(e)) })} />
          </label>
        )}
        {d.optimizer === "momentum" && (
          <label class="field">
            Batch size
            <input type="number" min={1} value={d.batch_size} onChange={(e) => num(e) >= 1 && setDraft({ batch_size: Math.round(num(e)) })} />
          </label>
        )}
        <label class="field">
          Epochs
          <input type="number" min={1} max={100000} value={d.epochs} onChange={(e) => num(e) >= 1 && setDraft({ epochs: Math.round(num(e)) })} />
        </label>
        <label class="field">
          Validation split
          <select value={String(d.validation_split)} onChange={(e) => setDraft({ validation_split: num(e) })}>
            {[0, 0.1, 0.15, 0.2, 0.25, 0.3].map((v) => (
              <option value={v}>{v === 0 ? "none" : `${Math.round(v * 100)} %`}</option>
            ))}
          </select>
        </label>
        <label class="field">
          Speed
          <select value={String(d.pace)} onChange={(e) => setDraft({ pace: num(e) })}>
            <option value="5">5 epochs / s</option>
            <option value="30">30 epochs / s</option>
            <option value="120">120 epochs / s</option>
            <option value="0">as fast as possible</option>
          </select>
        </label>
        {!compact && (
          <>
            <label class="check wide">
              <input type="checkbox" checked={d.shuffle} onChange={(e) => setDraft({ shuffle: (e.currentTarget as HTMLInputElement).checked })} />
              Shuffle the training rows every epoch
            </label>
            <label class="check wide">
              <input
                type="checkbox"
                checked={d.early_stopping}
                disabled={d.validation_split === 0}
                onChange={(e) => setDraft({ early_stopping: (e.currentTarget as HTMLInputElement).checked })}
              />
              Stop early, patience
              <input
                class="tiny"
                type="number"
                min={1}
                value={d.patience}
                disabled={!d.early_stopping || d.validation_split === 0}
                aria-label="Patience in epochs"
                onChange={(e) => num(e) >= 1 && setDraft({ patience: Math.round(num(e)) })}
              />
              epochs
            </label>
          </>
        )}
      </fieldset>
      {!compact && (
        <p class="note">
          The crate updates the weights after every row, batch size only groups the loss it reports. Speed only spaces the epochs out so you can watch; the numbers are the same at any speed.
        </p>
      )}
    </section>
  );
}
