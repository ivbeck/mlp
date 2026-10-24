import "@fontsource/fira-sans/400.css";
import "@fontsource/fira-sans/500.css";
import "@fontsource/fira-sans/600.css";
import "@fontsource/fira-sans-condensed/500.css";
import "./style.css";

import { render } from "preact";
import { useEffect } from "preact/hooks";
import { activeRun, control, isLive, restore, set, useStore, type View } from "./store";
import { DataView } from "./views/Data";
import { NetworkView } from "./views/Network";
import { TrainView } from "./views/Train";
import { ProbeView } from "./views/Probe";
import { bytes, fmtInt } from "./util";

const TABS: { id: View; name: string }[] = [
  { id: "data", name: "Data" },
  { id: "network", name: "Network" },
  { id: "train", name: "Train" },
  { id: "probe", name: "Probe" },
];

function go(view: View) {
  set({ view });
  history.replaceState(null, "", `#${view}`);
}

function App() {
  const s = useStore();
  const run = activeRun(s);
  const enabled: Record<View, boolean> = {
    data: true,
    network: !!s.dataset,
    train: !!s.dataset || s.runs.length > 0,
    probe: !!run && run.history.length > 0,
  };
  const view = enabled[s.view] ? s.view : "data";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, select, textarea, button, [role=slider]") || view !== "train") return;
      const r = activeRun();
      if (!r || !isLive(r)) return;
      if (e.key === " ") {
        e.preventDefault();
        void control(r.summary.state === "running" ? "pause" : "resume");
      } else if (e.key === ".") {
        void control("step");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view]);

  const stream = !run
    ? null
    : s.stream === "live"
      ? isLive(run)
        ? { cls: "ok", text: "streaming" }
        : { cls: "idle", text: "run over" }
      : s.stream === "retrying"
        ? { cls: "bad", text: "reconnecting" }
        : s.stream === "connecting"
          ? { cls: "idle", text: "connecting" }
          : { cls: "idle", text: "run over" };

  return (
    <>
      <header class="top">
        <div class="brand">mlp console</div>
        <nav aria-label="Steps">
          {TABS.map((t) => (
            <button
              class={view === t.id ? "on" : ""}
              aria-current={view === t.id ? "page" : undefined}
              disabled={!enabled[t.id]}
              onClick={() => go(t.id)}
            >
              {t.name}
            </button>
          ))}
        </nav>
        <div class="file">
          {s.dataset?.is_sample && <span class="tag">sample data</span>}
          {s.dataset && (
            <span class="chip" title={`${bytes(s.dataset.size_bytes)}`}>
              {s.dataset.name} · {fmtInt(s.dataset.rows)} rows · {s.dataset.columns.length - 1} features
              {s.target !== null && <> → {s.dataset.columns[s.target]?.name}</>}
            </span>
          )}
          {stream && (
            <span class={`stream ${stream.cls}`}>
              <i />
              {stream.text}
            </span>
          )}
        </div>
      </header>
      {(s.error || s.busy) && (
        <div class={s.error ? "banner error" : "banner"} role={s.error ? "alert" : "status"}>
          <span>{s.error ?? `${s.busy}…`}</span>
          {s.error && (
            <button class="link" onClick={() => set({ error: null })}>
              Dismiss
            </button>
          )}
        </div>
      )}
      <main class={`view-${view}`}>
        {view === "data" && <DataView s={s} go={go} />}
        {view === "network" && <NetworkView s={s} />}
        {view === "train" && <TrainView s={s} go={go} />}
        {view === "probe" && <ProbeView s={s} />}
      </main>
    </>
  );
}

const start = location.hash.slice(1) as View;
if (TABS.some((t) => t.id === start)) set({ view: start });
void restore();
render(<App />, document.getElementById("app")!);
