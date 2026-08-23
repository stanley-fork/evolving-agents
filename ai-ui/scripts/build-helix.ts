/**
 * The bundle, as a page you can turn.
 *
 *   cd ai-ui && node scripts/build-helix.ts --out helix.html
 *
 * The geometry and every rule about what may be claimed live in
 * [helix.ts](../src/helix.ts) and [threads.ts](../src/threads.ts), both pure and
 * both tested. This file is the renderer and the chrome, and it is deliberately
 * the only part with no tests: it is the part where being wrong is visible.
 *
 * ## What moves, and what each motion means
 *
 * **The content drifts left, always.** `now` is the right edge and the clock is
 * genuinely running, so work recorded twenty-two minutes ago is twenty-three
 * minutes ago a minute later — and it moves. That drift means *time is passing*,
 * which is always true, and it is the only unconditional motion here. Because
 * the strands twist along the axis, the drift also turns the bundle: that
 * rotation is geometry, not animation, and it costs nothing.
 *
 * **Light travels a strand only where a step is open right now.** That means
 * *work is happening*, which is usually false — and when it is false, nothing on
 * any strand moves.
 *
 * **The bundle turns to bring something forward, and says why.** Turning is a
 * claim — *this deserves looking at* — so it carries its reason and the address
 * it read, and `assertJustified` throws rather than let one through without.
 *
 * ## Not three.js, and the reason has not changed
 *
 * Depth here is `cos θ`, and everything it buys — occlusion by draw order,
 * thickness, opacity, blur — is four lines of arithmetic. An engine buys real
 * perspective, a camera, and thousands of strands. That is worth buying **after**
 * the stopwatch says this arrangement helps, and it would spend the property that
 * makes any of it checkable: one self-contained file that opens from disk with no
 * server and no network.
 */
import { writeFileSync } from "node:fs";
import { HELIX_JS, attentionOf } from "../src/helix.ts";
import { threadsOf } from "../src/threads.ts";
import { traceOf } from "../src/trace.ts";
import { agentOfIntent } from "../src/server.ts";
import { DEMO_AT } from "../src/simulate.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";
import { cochleaProjectFlows } from "../src/cochlea-project.ts";
import { hemoFlows } from "../src/hemo-demo.ts";
import { memoryFlows } from "../src/memory-demo.ts";
import { dspFlows } from "../src/dsp-demo.ts";

const outIdx = process.argv.indexOf("--out");
const out = outIdx >= 0 ? process.argv[outIdx + 1]! : "helix.html";

const docsOf = (raw: Array<Record<string, unknown>>) =>
  raw.map((d) => ({
    id: d["id"] as string,
    title: d["title"] as string,
    state: d["state"] as string,
    trace: traceOf(d["steps"] as never, agentOfIntent),
  }));

const SCENES = [
  {
    id: "coclea-sr",
    label: "coclea-sr — truth is derivable",
    note:
      "Two chains, the same six agents, the same six steps — and thirty hours apart. One came " +
      "home. The other stops at AUDITOR, because a gate declared before the run measured " +
      "2.592e-4 against a tolerance of 1.0e-4. One step in this bundle is open right now.",
    world: threadsOf(docsOf([...cochleaFlows(DEMO_AT), ...cochleaProjectFlows(DEMO_AT)] as never)),
  },
  {
    id: "hemo-verified",
    label: "hemo-verified — truth is not derivable",
    note:
      "No closed form exists here, so the judge itself is measured: 0.9056 composite against a " +
      "kill threshold written down first, while six of its seven oracles are near a coin flip " +
      "alone. One strand frays and stops — A4, read on two machines, never compared under " +
      "conditions where disagreeing is defined.",
    world: threadsOf(docsOf(hemoFlows(DEMO_AT) as never)),
  },
  {
    id: "memory-lab",
    label: "memory lab — green and wrong",
    note:
      "Two strands index the same notes with the same agents, and both are green. One is wrong: " +
      "a step used nothing it was given, because its note claims 663 characters of a passage " +
      "that is 1,105. The bundle turns to it on its own, and says why.",
    world: threadsOf(docsOf(memoryFlows(DEMO_AT) as never)),
  },
  {
    id: "signal-lab",
    label: "signal lab — ran and carried nothing",
    note:
      "The same claim one level down, about a step rather than a result: a stage ran, settled, " +
      "reported, and carried nothing forward. Its output is a flatline and the prose about it " +
      "reads exactly like the prose about a clean band.",
    world: threadsOf(docsOf(dspFlows(DEMO_AT) as never)),
  },
];

const payload = SCENES.map((s) => ({
  id: s.id,
  label: s.label,
  note: s.note,
  world: s.world,
  attention: attentionOf(s.world),
}));

const CSS = String.raw`
:root{
  --blue:#0A84FF; --green:#30D158; --indigo:#5E5CE6; --orange:#FF9F0A;
  --pink:#FF375F; --purple:#BF5AF2; --red:#FF453A; --teal:#64D2FF; --yellow:#FFD60A;
  --bg:#000; --bg-2:#1C1C1E; --bg-3:#2C2C2E; --sep:#38383A;
  --ink:#F2F2F7; --dim:#98989F; --faint:#636366;
  --sans:ui-sans-serif,-apple-system,"SF Pro Text",system-ui,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Monaco,"Roboto Mono",monospace;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
  font:13.5px/1.55 var(--sans);-webkit-font-smoothing:antialiased;overflow:hidden}
.wrap{display:grid;grid-template-columns:1fr 340px;height:100%}
.stage{position:relative;overflow:hidden}
header{position:absolute;top:0;left:0;right:0;z-index:5;display:flex;gap:12px;align-items:center;
  padding:14px 20px 26px;background:linear-gradient(180deg,#000 52%,transparent);pointer-events:none}
header>*{pointer-events:auto}
header b{font-weight:600;letter-spacing:-.01em;white-space:nowrap}
header{flex-wrap:nowrap}
header .sk{font-size:10px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  color:var(--orange);background:rgba(255,159,10,.12);border:1px solid rgba(255,159,10,.28);
  border-radius:999px;padding:3px 10px}
header .build{font:10px var(--mono);color:var(--faint);background:var(--bg-2);
  border:1px solid var(--sep);border-radius:999px;padding:3px 9px;cursor:help}
header select{font:inherit;font-size:12px;background:var(--bg-2);color:var(--ink);
  border:1px solid var(--sep);border-radius:8px;padding:6px 10px}
header .right{margin-left:auto;display:flex;gap:8px;align-items:center}
header .lvl{font:11px var(--mono);color:var(--dim)}
svg{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
svg.grabbing{cursor:grabbing}

/* The axis the bundle is wound around. Faint: it is a construction line, not a
   thing that happened. */
.axis{stroke:#1A1A1C;stroke-width:1}
.tickline{stroke:#141416;stroke-width:1}
.axistext{font:10px var(--mono);fill:var(--faint)}
.nowline{stroke:var(--ink);stroke-width:1;opacity:.22}
.nowtext{font:10px var(--mono);fill:var(--dim)}

/* A strand. Width and opacity come from depth, in JS, because they are the
   picture's only cue for how far forward something is. */
.strand{fill:none;stroke-linecap:round;stroke-linejoin:round}
.glow{fill:none;stroke-linecap:round;stroke-linejoin:round}
.strand.ignored{stroke-dasharray:2 5}
.strand.open{stroke-dasharray:9 5}
.strand.pending{stroke-dasharray:2 6}
.gap{stroke:var(--faint);stroke-width:1;stroke-dasharray:1 4}
.stop{stroke-linecap:butt}
.hit{fill:none;stroke:transparent;cursor:pointer}

/* A body riding a strand: the agent holding the thought right now, drawn the way
   a polymerase is drawn on the strand it is reading. */
.body{stroke:#000;stroke-width:1.5}
.bodyring{fill:none;stroke-width:1.5}
.bodylabel{font:10px var(--mono);paint-order:stroke;stroke:#000;stroke-width:3px;stroke-linejoin:round}
.human .bodylabel{font-family:var(--sans);font-weight:600}

/* The system agent, attached. It rides whatever it was put on. */
.watcher{stroke:#000;stroke-width:1.5}
.watchring{fill:none;stroke:var(--teal);stroke-width:1;opacity:.8}
.watchlink{stroke:var(--teal);stroke-width:1;stroke-dasharray:2 3;opacity:.6}

.dot{font:10px var(--mono);fill:var(--dim)}
aside{background:var(--bg-2);border-left:1px solid var(--sep);padding:18px 20px;overflow:auto}
aside h2{margin:0 0 2px;font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint)}
aside h3{margin:9px 0 3px;font-size:16px;font-weight:600;letter-spacing:-.02em}
aside .sub{font:11px var(--mono);color:var(--dim);margin-bottom:12px}
aside .fld{display:grid;grid-template-columns:88px 1fr;gap:3px 10px;font-size:11.5px;margin:0 0 5px}
aside .fld .k{color:var(--faint)}
aside .at{display:block;font-family:var(--mono);font-size:10.5px;color:var(--teal);
  background:none;border:0;border-bottom:1px dotted rgba(100,210,255,.4);padding:0;margin-top:2px;
  cursor:pointer;text-align:left}
aside .fnd{border:1px solid var(--sep);border-radius:10px;padding:12px 14px;margin-top:12px;
  background:var(--bg-3)}
aside .fnd .vd{font-size:10px;letter-spacing:.07em;text-transform:uppercase;font-weight:700}
aside .fnd.ok .vd{color:var(--green)}
aside .fnd.problem .vd{color:var(--red)}
aside .fnd.unknown{background:repeating-linear-gradient(45deg,var(--bg-3) 0 6px,var(--bg-2) 6px 12px)}
aside .fnd.unknown .vd{color:var(--dim)}
aside .fnd .sy{margin:6px 0 0;font-size:12.5px;line-height:1.5}
aside .fnd .ct{margin-top:8px;font-size:10.5px;color:var(--faint)}
aside .sw{display:flex;gap:6px;margin:10px 0 12px}
aside .sw button{flex:1 1 0}
button{font:inherit;font-size:11.5px;font-weight:500;padding:6px 12px;color:var(--ink);
  background:var(--bg-3);border:1px solid var(--sep);border-radius:8px;cursor:pointer}
button:hover{background:#3A3A3C}
button[aria-selected="true"]{background:rgba(10,132,255,.18);border-color:rgba(10,132,255,.5);color:#9CCEFF}
button.on{background:rgba(100,210,255,.16);border-color:rgba(100,210,255,.5);color:#BFEBFF}

/* What the bundle turned to, and why. Always present, because a rotation the
   surface cannot justify is one it must not make. */
.att{margin-top:14px;border-top:1px solid var(--sep);padding-top:13px}
.att h4{margin:0 0 8px;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);
  font-weight:600}
.att .row{display:flex;gap:9px;align-items:flex-start;padding:5px 0;font-size:11.5px;line-height:1.45;
  cursor:pointer;border-radius:6px}
.att .row:hover{background:rgba(255,255,255,.04)}
.att .row .sw2{width:12px;height:12px;flex:0 0 12px;border-radius:3px;margin-top:2px}
.att .row.front{background:rgba(255,255,255,.06)}
.att .row .why{color:var(--dim);display:block;font-size:11px}
.att .row .kd{font:9px var(--mono);text-transform:uppercase;letter-spacing:.05em;color:var(--faint)}

.key{margin-top:14px;border-top:1px solid var(--sep);padding-top:13px}
.key h4{margin:0 0 7px;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);
  font-weight:600}
.key .row{display:flex;align-items:center;gap:9px;padding:3px 0;font-size:11.5px;line-height:1.4}
.key svg{width:44px;height:10px;flex:0 0 44px}
.note{margin-top:14px;font-size:11.5px;line-height:1.55;color:var(--dim)}
.bytes{font-family:var(--mono);font-size:10.5px;line-height:1.6;background:#000;border:1px solid var(--sep);
  border-radius:8px;padding:10px 11px;margin-top:9px;max-height:200px;overflow:auto;white-space:pre-wrap;
  word-break:break-word;color:#C7C7CC}
.tourbar{position:fixed;left:20px;bottom:20px;z-index:400;display:flex;align-items:center;gap:12px;
  background:var(--bg-2);border:1px solid var(--sep);border-radius:999px;
  box-shadow:0 8px 28px rgba(0,0,0,.6);padding:8px 16px 8px 8px;
  max-width:min(720px,calc(100vw - 400px))}
.tourbar button{border-radius:999px;background:var(--ink);color:#000;border-color:var(--ink);
  font-weight:650;padding:6px 16px}
.tourbar span{font-size:12px;line-height:1.45;color:var(--dim)}
`;

const JS = String.raw`
(() => {
  const S = window.__BUNDLE__;
  const G = window.__GEO__;
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a) => { const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]); return e; };
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const HUES = ['#0A84FF','#FF9F0A','#30D158','#BF5AF2','#FF375F','#64D2FF','#5E5CE6','#FFD60A'];
  const hueOf = (o) => HUES[o % HUES.length];

  let scene = S[0], sel = null, mode = 'read', watching = null;
  const svg = document.getElementById('stage');
  const panel = document.getElementById('panel');
  const pick = document.getElementById('scene');
  const lvl = document.getElementById('lvl');

  const M = { l: 96, r: 96, t: 96, b: 72 };
  let W = 0, H = 0, cy = 0, R = 0;

  /**
   * The window, and the drift.
   *
   * 'now' is the right edge. It advances with the wall clock, so recorded work
   * moves left on its own — not because anything is being animated, but because
   * time is passing, which is the one thing that is always true. 'driftedFor'
   * is how long this page has been open, in the same minutes the layout uses.
   */
  const opened = Date.now();
  const driftedFor = () => (Date.now() - opened) / 60000;
  let win = 0;                                  // minutes visible
  const nowT = () => scene.world.span + driftedFor();
  const t1 = () => nowT() + win * 0.06;         // a little room past now
  const t0 = () => t1() - win;
  /**
   * How wide the window opens, and why it is not 'all of it'.
   *
   * The bundle coils once every PITCH minutes, so a window has to be a few
   * pitches across for the coil to be a coil. Opening on the whole history put
   * forty-eight turns in a thousand pixels: every strand a vertical sliver, and
   * the shape an artefact of the sampling. Four pitches is enough to see the
   * winding and enough to hold a scope's recent work. 'All' is one button away
   * and it collapses the coil honestly — see TWIST_FLOOR_PX.
   */
  const PITCH = 90;

  /**
   * Frame the object, not a duration.
   *
   * The first version opened on a fixed four pitches, and two thirds of the
   * canvas was empty: the bundle exists where work happened, work happened
   * recently, and the frame did not know that. Nothing was wrong with the
   * strands — the *frame* was wrong, and the fix is not to add anything to the
   * picture but to put the picture in the frame.
   *
   * So the window is the extent of the most recent cluster of work, with a
   * margin, and 'now' stays pinned to the right edge because that rule is what
   * makes 'drag left to go back' mean anything. Older clusters are off to the
   * left, which is where they are; 'All' shows the whole history and lets the
   * coil collapse honestly.
   */
  const resetWin = () => {
    const n = nowT();
    const ends = scene.world.threads.map((t) => Math.max(...t.rests.map((r) => r.x1)));
    const latest = Math.max(...ends);
    // Threads that finished within the last tenth of the history are 'recent'.
    const recentCut = latest - Math.max(PITCH, scene.world.span * 0.1);
    const starts = scene.world.threads
      .filter((t, i) => ends[i] >= recentCut)
      .map((t) => Math.min(...t.rests.map((r) => r.x0)));
    const from = starts.length ? Math.min(...starts) : latest - PITCH;
    win = Math.max(PITCH * 0.6, Math.min(scene.world.span * 1.06, (n - from) * 1.18));
  };

  /**
   * Below this many pixels per turn, the coil is not drawn.
   *
   * Not a taste threshold: a turn narrower than this is finer than the marks
   * available to draw it, so what a reader sees is aliasing. The strands go
   * parallel — still depth-ordered, still turnable — and the panel says the coil
   * is not being drawn rather than drawing one that means nothing.
   */
  const TWIST_FLOOR_PX = 46;
  const pxPerTurn = () => (PITCH / win) * (W - M.l - M.r);
  const coiling = () => pxPerTurn() >= TWIST_FLOOR_PX;
  const twistNow = () => (coiling() ? (Math.PI * 2) / PITCH : 0);

  /** Rotation, and where it is heading. */
  let rot = 0, rotTarget = 0, rotWhy = null;
  /** Where the bodies are read. Null means 'wherever the attention is'. */
  let cursor = null;

  /**
   * Where the reading head sits when nobody has moved it.
   *
   * The middle of whatever the bundle turned to — so the strand that is in front
   * is also the one with a body on it, and the two mechanisms agree instead of
   * pointing at different things. Falling back to the last moment anything was
   * held, and only then to now.
   */
  const restingCursor = () => {
    const top = scene.attention[0];
    const th = top && scene.world.threads.find((t) => t.flowId === top.flowId);
    if (th) {
      const live = th.rests.find((r) => r.state === 'running');
      if (live) return (live.x0 + live.x1) / 2;
      const last = th.rests[th.rests.length - 1];
      if (last) return (last.x0 + last.x1) / 2;
    }
    let m = 0;
    for (const t of scene.world.threads)
      for (const r of t.rests) if (r.x1 > m) m = r.x1;
    return m || nowT();
  };

  const measure = () => {
    const r = svg.getBoundingClientRect();
    W = r.width; H = r.height;
    cy = M.t + (H - M.t - M.b) / 2;
    /**
     * The bundle's radius — how thick the rope is, not how tall the page is.
     *
     * It used to be half the available height, so the strands swept the whole
     * screen and read as waves rather than as a bundle. A rope is a *band*: wide
     * enough that a strand at the front is clearly in front of one at the back,
     * narrow enough that the whole thing is one object.
     */
    R = Math.max(48, Math.min(132, (H - M.t - M.b) / 2 - 26));
  };
  const xOf = (t) => M.l + ((t - t0()) / win) * (W - M.l - M.r);
  const tOf = (px) => t0() + ((px - M.l) / (W - M.l - M.r)) * win;

  /** Depth to ink. Everything the third dimension buys, in three lines. */
  const wOf = (z) => 1.0 + 2.4 * ((z + 1) / 2);
  const oOf = (z) => 0.14 + 0.86 * Math.pow((z + 1) / 2, 1.6);
  const yOf = (y) => cy + y * R;

  const pathOf = (samples) => samples
    .map((s, i) => (i ? 'L' : 'M') + xOf(s.t) + ' ' + yOf(s.y))
    .join(' ');

  const isLive = (s) => s.kind === 'rest' && s.state === 'running';
  const isPending = (s) => s.kind === 'rest' &&
    (s.state === 'pending' || s.state === 'waiting' || s.state === 'draft');
  const liveCount = () => scene.world.threads
    .reduce((n, t) => n + t.rests.filter((r) => r.state === 'running').length, 0);

  const clsOf = (s) => {
    if (s.kind === 'cross') return s.state;
    if (isLive(s)) return 'running';
    if (isPending(s)) return 'pending';
    return 'carried';
  };

  function draw() {
    measure();
    svg.replaceChildren();
    const now = nowT(), a = t0(), b = t1();

    // The axis the bundle is wound around, and the clock along it.
    svg.appendChild(el('line', { class: 'axis', x1: M.l, y1: cy, x2: W - M.r, y2: cy }));
    const ticks = 7;
    for (let i = 0; i <= ticks; i += 1) {
      const t = a + (win * i) / ticks, px = xOf(t);
      svg.appendChild(el('line', { class: 'tickline', x1: px, y1: cy - R - 18, x2: px, y2: cy + R + 18 }));
      const lab = el('text', { class: 'axistext', x: px, y: H - M.b + 30, 'text-anchor': 'middle' });
      const d = now - t;
      lab.textContent = d <= 0.4 ? 'now'
        : d < 90 ? Math.round(d) + 'm ago'
        : d < 2880 ? (d / 60).toFixed(1) + 'h ago'
        : (d / 1440).toFixed(1) + 'd ago';
      svg.appendChild(lab);
    }
    const nx = xOf(now);
    svg.appendChild(el('line', { class: 'nowline', x1: nx, y1: cy - R - 26, x2: nx, y2: cy + R + 26 }));
    const live = liveCount();
    const nl = el('text', { class: 'nowtext', x: nx - 8, y: cy - R - 32, 'text-anchor': 'end' });
    nl.textContent = live
      ? live + (live === 1 ? ' step open · now' : ' steps open · now')
      : 'now · nothing running';
    svg.appendChild(nl);

    // The strands, back to front. G.bundle sorts by depth so drawing in order
    // gives occlusion with no z-buffer and no thinking.
    const segs = G.bundle(scene.world, { now: now, rotation: rot, t0: a, t1: b, twist: twistNow() });

    /**
     * Frame the object on this axis too.
     *
     * A window a turn and a half wide catches the strands on whichever part of
     * the circle they happen to be riding, and with three of them in phase they
     * all sat in the top half — the object correctly drawn and floating in a
     * frame that was not about it. The horizontal fix was to frame the extent of
     * the content; this is the same fix on the other axis.
     *
     * The centre shifts, the radius does not: squeezing the bundle to fit would
     * make its thickness a function of which slice you are looking at, and
     * thickness is how a reader tells front from back.
     */
    let lo = 1, hi = -1;
    for (const sg of segs) for (const p of sg.samples) { if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y; }
    if (hi >= lo) cy -= ((lo + hi) / 2) * R;
    for (const s of segs) {
      const cls = clsOf(s);
      const hue = hueOf(s.ordinal);
      const dim = sel && sel.flowId !== s.flowId ? 0.22 : 1;
      const d = pathOf(s.samples);

      if (cls === 'unknown') {
        // Nothing recorded: no strand. You see the axis through the hole.
        svg.appendChild(el('path', { class: 'gap', d: d, opacity: oOf(s.z) * dim }));
        continue;
      }
      svg.appendChild(el('path', { class: 'glow', d: d, stroke: hue,
        'stroke-width': wOf(s.z) * 4.6, opacity: oOf(s.z) * 0.13 * dim }));
      const p = el('path', { class: 'strand ' + cls, d: d, stroke: hue,
        'stroke-width': wOf(s.z) * (sel && sel.flowId === s.flowId ? 1.6 : 1),
        opacity: oOf(s.z) * dim });
      if (cls === 'ignored') p.setAttribute('opacity', String(oOf(s.z) * 0.3 * dim));
      if (cls === 'pending') p.setAttribute('opacity', String(oOf(s.z) * 0.34 * dim));
      svg.appendChild(p);
      if (cls === 'running') p.classList.add('flow');

      const hit = el('path', { class: 'hit', d: d, 'stroke-width': Math.max(14, wOf(s.z) * 6) });
      hit.addEventListener('pointerdown', (ev) => { ev.stopPropagation(); select(s); });
      svg.appendChild(hit);
    }

    // Where a strand stops for good: it ran into something.
    for (const th of scene.world.threads) {
      if (th.delivered) continue;
      const last = th.crosses[th.crosses.length - 1];
      if (!last || last.x1 < a || last.x1 > b) continue;
      const ph = G.phase(th.ordinal, scene.world.threads.length, last.x1, now, rot, twistNow());
      svg.appendChild(el('line', { class: 'stop', x1: xOf(last.x1) + 3, y1: yOf(ph.y) - 9,
        x2: xOf(last.x1) + 3, y2: yOf(ph.y) + 9,
        stroke: last.state === 'open' ? '#98989F' : '#FF453A',
        'stroke-width': wOf(ph.z) * 1.4, opacity: oOf(ph.z) }));
    }

    /**
     * The bodies: who is holding what, at the moment you are looking at.
     *
     * 'cursor' defaults to the latest moment anything was recorded and follows
     * the pointer when you move it across the bundle. Not 'now': now runs past
     * the last record continuously — that is the drift — and at a moment after
     * everything finished, nobody is holding anything. True, and useless. The
     * cursor is the question 'who had this, then', which is the question a body
     * on a strand answers.
     *
     * A strand with nothing spanning the cursor gets no body. At that moment
     * nobody was holding it, and drawing one anyway would assert work that was
     * not being done.
     */
    const bodyT = cursor === null ? restingCursor() : cursor;
    const bodies = G.bodies(scene.world, bodyT, rot);
    svg.appendChild(el('line', { class: 'nowline', x1: xOf(bodyT), y1: cy - R - 8,
      x2: xOf(bodyT), y2: cy + R + 8, opacity: .12 }));
    for (const bo of bodies) {
      const hue = hueOf(bo.ordinal);
      const x = xOf(bo.t), y = yOf(bo.y);
      const dim = sel && sel.flowId !== bo.flowId ? 0.25 : 1;
      // A packet is smaller and hollow: it is a thing in transit, not somebody
      // working, and the two must not read alike.
      const isPkt = bo.kind === 'packet';
      const r = (isPkt ? 2.4 : 4) + (isPkt ? 1.6 : 3.2) * ((bo.z + 1) / 2);
      const g = el('g', { class: bo.kind === 'human' ? 'human' : 'agent', opacity: oOf(bo.z) * dim });
      g.appendChild(el('circle', { class: 'body', cx: x, cy: y, r: r,
        fill: isPkt ? 'none' : bo.kind === 'human' ? '#F2F2F7' : hue,
        stroke: isPkt ? hue : '#000', 'stroke-width': isPkt ? 1.6 : 1.5 }));
      if (bo.state === 'running')
        g.appendChild(el('circle', { class: 'bodyring pulse', cx: x, cy: y, r: r + 5, stroke: hue }));
      const lab = el('text', { class: 'bodylabel', x: x + r + 6, y: y + 3,
        fill: bo.kind === 'human' ? '#F2F2F7' : hue });
      lab.textContent = bo.name === '@human' ? 'you'
        : isPkt ? '→ ' + bo.name : bo.name;
      g.appendChild(lab);
      svg.appendChild(g);

      // The system agent, if it was attached to this strand.
      if (watching === bo.flowId) {
        svg.appendChild(el('line', { class: 'watchlink', x1: x, y1: y, x2: x, y2: y - 30 - r }));
        const wg = el('g', {});
        wg.appendChild(el('rect', { class: 'watcher', x: x - 7, y: y - 44 - r, width: 14, height: 14,
          rx: 3, fill: '#64D2FF' }));
        wg.appendChild(el('circle', { class: 'watchring pulse', cx: x, cy: y - 37 - r, r: 12 }));
        const wl = el('text', { class: 'bodylabel', x: x + 12, y: y - 33 - r, fill: '#64D2FF' });
        wl.textContent = 'INSPECTOR';
        wg.appendChild(wl);
        svg.appendChild(wg);
      }
    }

    lvl.textContent = fmtDur(win) + ' wide' + (coiling() ? '' : ' · flat');
    if (!sel) renderRest(); else renderPanel();
    animate();
  }

  const fmtDur = (min) => min < 1 ? Math.round(min * 60) + 's'
    : min < 90 ? Math.round(min) + 'm'
    : min < 2880 ? (min / 60).toFixed(1) + 'h' : (min / 1440).toFixed(1) + 'd';

  /**
   * The loop, and the two things it is allowed to do.
   *
   * It runs while the page is open, because the drift is unconditional — time
   * passes whether or not anything is happening, and the content has to move for
   * that to be true on screen. Inside it, the dash offset only advances on
   * strands that are actually open, so 'work is happening' stays a separate claim
   * from 'the clock is running'.
   */
  let raf = null, phase = 0, lastDraw = 0;
  const tick = (ts) => {
    // Cleared first, so every path below either schedules the next frame itself
    // or lets animate() do it. Without this the redraw branch returned with raf
    // still set, animate() saw a live loop that was not live, and the drift —
    // the one motion that is supposed to be unconditional — stopped after the
    // first frame.
    raf = null;
    // Ease the rotation toward whatever was asked for, the short way round.
    if (Math.abs(rotTarget - rot) > 0.001) {
      let d = rotTarget - rot;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      rot += d * 0.09;
      draw();
      raf = requestAnimationFrame(tick);
      return;
    }
    phase = (phase + 0.9) % 23;
    for (const p of svg.querySelectorAll('.strand.running'))
      p.setAttribute('stroke-dashoffset', String(-phase));
    // Redraw about four times a second so the drift is visible without
    // re-laying-out sixty times for a picture that moves a pixel a minute.
    if (ts - lastDraw > 240) { lastDraw = ts; draw(); return; }
    raf = requestAnimationFrame(tick);
  };
  const animate = () => { if (!raf) raf = requestAnimationFrame(tick); };
  // The dash pattern for a strand that is open right now. Set here rather than
  // in CSS so it cannot be applied to a strand that is not.
  const styleLive = () => {
    for (const p of svg.querySelectorAll('.strand.running'))
      p.setAttribute('stroke-dasharray', '14 9');
  };

  /**
   * Turn the bundle to a strand, and say why.
   *
   * assertJustified throws on a reason with no address, so a rotation the
   * surface cannot justify cannot be made at all.
   */
  function turnTo(flowId, why) {
    const th = scene.world.threads.find((t) => t.flowId === flowId);
    if (!th) return;
    if (why) G.justify(why);
    rotWhy = why || null;
    rotTarget = G.rotationFor(th.ordinal, scene.world.threads.length, nowT());
    animate();
  }

  const fld = (k, v, at) => '<div class="fld"><span class="k">' + esc(k) + '</span><span>' +
    esc(v) + (at ? '<button class="at" data-open="' + esc(at) + '">' + esc(at) + '</button>' : '') +
    '</span></div>';

  const finding = (f) => {
    if (f.verdict !== 'unknown' && !f.cites.length)
      throw new Error('a verdict without an address must be reported as unknown');
    return '<div class="fnd ' + f.verdict + '"><div class="vd">' +
      (f.verdict === 'unknown' ? 'unknown' : f.verdict === 'ok' ? 'no problem found' : 'problem') +
      '</div><p class="sy">' + esc(f.says) + '</p>' +
      (f.cites.length
        ? '<div class="ct">read: ' + f.cites.map((c) =>
            '<button class="at" data-open="' + esc(c) + '">' + esc(c) + '</button>').join('') + '</div>'
        : '<div class="ct">nothing to read — that is why this is unknown</div>') + '</div>';
  };

  const attHtml = () => {
    const front = scene.world.threads.find((t) => {
      const p = G.phase(t.ordinal, scene.world.threads.length, nowT(), nowT(), rot);
      return p.z > 0.86;
    });
    return '<div class="att"><h4>What is in front, and why</h4>' +
      scene.attention.map((a) =>
        '<div class="row' + (front && front.flowId === a.flowId ? ' front' : '') +
        '" data-turn="' + esc(a.flowId) + '">' +
        '<span class="sw2" style="background:' + hueOf(a.ordinal) + '"></span>' +
        '<span><b>' + esc(a.title) + '</b><span class="kd"> ' + esc(a.kind) + '</span>' +
        '<span class="why">' + esc(a.reason) +
        (a.at ? '' : ' — nothing recorded to point at') + '</span></span></div>').join('') +
      '</div>';
  };

  const keyHtml = () => {
    const live = liveCount();
    const sw = (cls, hue) => '<svg viewBox="0 0 44 10"><path class="strand ' + cls +
      '" d="M2 5 H42" stroke="' + (hue || '#0A84FF') + '" stroke-width="2.6"/></svg>';
    return '<div class="key"><h4>Depth — what is in front</h4>' +
      '<div class="row"><svg viewBox="0 0 44 10"><path class="strand" d="M2 5 H42" stroke="#0A84FF" stroke-width="3.4"/></svg>' +
      '<span>near — the bundle is turned to it</span></div>' +
      '<div class="row"><svg viewBox="0 0 44 10"><path class="strand" d="M2 5 H42" stroke="#0A84FF" stroke-width="1.1" opacity=".2"/></svg>' +
      '<span>behind — still there, still clickable</span></div>' +
      '<h4 style="margin-top:12px">Texture — what happened</h4>' +
      '<div class="row">' + sw('') + '<span><b>carried</b> — it moved and the next step used it</span></div>' +
      '<div class="row">' + sw('ignored') + '<span><b>carried nothing forward</b></span></div>' +
      '<div class="row">' + sw('open') + '<span><b>no verdict yet</b></span></div>' +
      '<div class="row">' + sw('pending') + '<span><b>not begun</b> — order known, time not</span></div>' +
      '<div class="row"><svg viewBox="0 0 44 10"><path class="gap" d="M2 5 H42"/></svg>' +
      '<span><b>unrecorded</b> — no strand is drawn</span></div>' +
      (coiling() ? '' :
        '<div class="row"><span><b>the coil is not drawn at this width</b> — one turn would be ' +
        'narrower than the marks drawing it, so what you would see is the sampling, not the shape. ' +
        'Zoom in and it winds.</span></div>') +
      '<h4 style="margin-top:12px">Motion</h4>' +
      '<div class="row"><span><b>everything drifts left</b> — the clock is running, which is always true</span></div>' +
      '<div class="row"><span><b>light travels a strand</b> — a step is open right now' +
      (live ? '' : '. <b>None is</b>, so no strand is moving') + '</span></div>' +
      '</div>';
  };

  const noteHtml = () => '<div class="note">' + esc(scene.note) + '</div>';

  function renderRest() {
    const live = liveCount();
    const steps = scene.world.threads.reduce((n, t) => n + t.rests.length, 0);
    panel.innerHTML = '<h2>This bundle</h2>' +
      '<h3>' + (live ? live + ' step' + (live === 1 ? '' : 's') + ' open right now' : 'nothing is running') + '</h3>' +
      '<div class="sub">' + scene.world.threads.length + ' strands · ' + steps + ' steps · ' +
      fmtDur(scene.world.span) + ' of history</div>' +
      '<div class="note" style="margin-top:0">Drag left or right to turn the bundle. Scroll to zoom. ' +
      'Click a strand, or a row below to bring it forward.</div>' +
      '<div class="sw"><button id="watch"' + (watching ? ' class="on"' : '') + '>' +
      (watching ? 'INSPECTOR is watching — detach' : 'Attach INSPECTOR to the front strand') + '</button></div>' +
      (watching ? watchHtml() : '') +
      attHtml() + keyHtml() + noteHtml();
    wire();
  }

  /**
   * What the system agent has to say about the strand it is riding.
   *
   * It is an agent like the others: one tool, read. Its finding goes through the
   * same guard as every other finding on this surface — a verdict with no
   * address is not renderable.
   */
  function watchHtml() {
    const a = scene.attention.find((x) => x.flowId === watching);
    if (!a) return '';
    const verdict = a.kind === 'ignored' || a.kind === 'blocked' ? 'problem'
      : a.kind === 'open' ? 'unknown' : a.kind === 'running' ? 'unknown' : 'ok';
    return finding({
      verdict: verdict,
      says: a.kind === 'running'
        ? 'I am on "' + a.title + '". ' + a.reason + ', so there is no observation yet — it has not finished.'
        : a.kind === 'settled'
          ? 'I am on "' + a.title + '". Nothing on it is open and nothing is flagged: ' + a.reason + '.'
          : 'I am on "' + a.title + '". ' + a.reason + '.',
      cites: a.at ? [a.at] : [],
    });
  }

  function select(seg) { sel = seg; mode = 'read'; turnTo(seg.flowId, null); draw(); }

  function renderPanel() {
    if (!sel) { renderRest(); return; }
    const isCross = sel.kind === 'cross';
    const nice = (n) => String(n).replace('@human', 'you');
    const head = '<h2>' + (isCross ? 'A handoff' : 'A step') + '</h2>' +
      '<h3>' + esc(isCross ? nice(sel.from) + ' → ' + nice(sel.to) : sel.from + ' · step ' + sel.index) + '</h3>' +
      '<div class="sub">' + esc(sel.title) + ' · ' + fmtDur(sel.t1 - sel.t0) + '</div>' +
      '<div class="sw"><button id="m-read"' + (mode === 'read' ? ' aria-selected="true"' : '') + '>Read it</button>' +
      '<button id="m-agent"' + (mode === 'agent' ? ' aria-selected="true"' : '') + '>Ask an agent</button></div>';

    let body;
    if (mode === 'read') {
      body = fld('state', sel.state) + fld('took', fmtDur(sel.t1 - sel.t0)) +
        (isCross && sel.because ? fld('because', sel.because) : '') +
        (sel.digest ? fld('observation', sel.digest, sel.source || undefined)
                    : fld('carried', 'nothing recorded — this is not a claim that nothing moved'));
    } else {
      const at = 'flow:' + sel.flowId + '#step-' + sel.index;
      const cites = sel.source ? [sel.source, at] : [at];
      let f;
      if (isCross && sel.from === '@human')
        f = { verdict: 'ok', cites: ['flow:' + sel.flowId],
              says: 'You asked for this. It is the one handoff on the strand that is not in question — everything after it is.' };
      else if (isCross && sel.to === '@human')
        f = { verdict: 'ok', cites: cites,
              says: 'It finished and came back to you. That is about delivery, not about whether the answer is right — a gate says that, and this does not.' };
      else if (isCross && sel.state === 'unknown')
        f = { verdict: 'unknown', cites: [],
              says: 'Nothing was recorded for this hop, so there is nothing to read. That is not a pass: it is the absence of evidence either way.' };
      else if (isCross && sel.state === 'open')
        f = { verdict: 'unknown', cites: cites,
              says: 'It reached ' + nice(sel.to) + ' and no verdict has been reached. ' + sel.because + '.' };
      else if (isCross && sel.state === 'ignored')
        f = { verdict: 'problem', cites: cites,
              says: 'It arrived and ' + sel.to + ' used none of it. ' + sel.because + '.' };
      else if (isCross && sel.state === 'blocked')
        f = { verdict: 'problem', cites: cites,
              says: 'It arrived, ' + sel.to + ' ran, and it did not pass. ' + sel.because + '.' };
      else if (isCross)
        f = { verdict: 'ok', cites: cites,
              says: nice(sel.from) + ' closed with an observation and ' + sel.to + ' read it.' };
      else if (isPending(sel))
        f = { verdict: 'unknown', cites: [],
              says: 'This step has not begun. Its place in the order is known and its time is not.' };
      else if (isLive(sel))
        f = { verdict: 'unknown', cites: cites,
              says: sel.from + ' has had this open for ' + fmtDur(sel.t1 - sel.t0) + ' and has not closed it. There is no observation yet, because it has not finished.' };
      else if (!sel.digest)
        f = { verdict: 'unknown', cites: [], says: 'This step recorded no observation, so there is nothing to read.' };
      else
        f = { verdict: (sel.state === 'failed' || sel.state === 'blocked') ? 'problem' : 'ok', cites: cites,
              says: sel.from + ' at step ' + sel.index + ' is ' + sel.state + ', took ' + fmtDur(sel.t1 - sel.t0) + ', and recorded ' + sel.digest + '.' };
      body = finding(f);
    }
    panel.innerHTML = head + body +
      '<div class="sw"><button id="back">Back to the bundle</button></div>' + attHtml() + keyHtml();
    wire();
  }

  const wire = () => {
    const r = panel.querySelector('#m-read'), a = panel.querySelector('#m-agent');
    if (r) r.onclick = () => { mode = 'read'; renderPanel(); };
    if (a) a.onclick = () => { mode = 'agent'; renderPanel(); };
    const back = panel.querySelector('#back');
    if (back) back.onclick = () => { sel = null; renderRest(); draw(); };
    const w = panel.querySelector('#watch');
    if (w) w.onclick = () => {
      if (watching) { watching = null; renderRest(); draw(); return; }
      const front = scene.attention[0];
      watching = front ? front.flowId : scene.world.threads[0].flowId;
      turnTo(watching, scene.attention.find((x) => x.flowId === watching));
      renderRest(); draw();
    };
    for (const row of panel.querySelectorAll('.att .row'))
      row.onclick = () => {
        const id = row.dataset.turn;
        turnTo(id, scene.attention.find((x) => x.flowId === id));
        sel = null; renderRest(); draw();
      };
    for (const b of panel.querySelectorAll('button.at'))
      b.onclick = () => {
        const at = b.dataset.open;
        const html = '<div class="bytes">' + esc(at) + '\n\n' +
          'This address is recorded on the segment above. This page has no filesystem, so the ' +
          'address is shown rather than the bytes — and no contents are invented for it.</div>';
        const box = panel.querySelector('.bytes');
        if (box) box.outerHTML = html; else panel.insertAdjacentHTML('beforeend', html);
        wire();
      };
  };

  // ---- turning and zooming -------------------------------------------------
  let drag = null;
  svg.addEventListener('pointerdown', (ev) => {
    drag = { x: ev.clientX, rot: rotTarget };
    svg.classList.add('grabbing');
    svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener('pointermove', (ev) => {
    if (!drag) {
      // Not dragging: move the reading head. The bodies are where you point.
      const px = ev.clientX - svg.getBoundingClientRect().left;
      if (px > M.l && px < W - M.r) { cursor = tOf(px); draw(); }
      return;
    }
    // Dragging turns the bundle. A full window's width is a full turn, so the
    // gesture is 'roll it', not 'scrub it'.
    rotTarget = drag.rot + ((ev.clientX - drag.x) / Math.max(200, W)) * Math.PI * 2;
    rot = rotTarget;
    rotWhy = null;
    draw();
  });
  const endDrag = (ev) => {
    if (drag && Math.abs(ev.clientX - drag.x) < 3) { sel = null; renderRest(); draw(); }
    drag = null; svg.classList.remove('grabbing');
  };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', () => { drag = null; svg.classList.remove('grabbing'); });
  svg.addEventListener('pointerleave', () => { cursor = null; draw(); });

  svg.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    win = Math.min(scene.world.span * 2.2, Math.max(1.5, win * Math.exp(ev.deltaY * 0.0016)));
    draw();
  }, { passive: false });

  document.getElementById('zin').onclick = () => { win = Math.max(1.5, win / 1.8); draw(); };
  document.getElementById('zout').onclick = () => {
    win = Math.min(scene.world.span * 2.2, win * 1.8); draw();
  };
  document.getElementById('znow').onclick = () => { resetWin(); sel = null; renderRest(); draw(); };

  pick.addEventListener('change', () => {
    scene = S.find((x) => x.id === pick.value) || S[0];
    sel = null; watching = null; resetWin();
    // Turn to whatever this scene says is worth looking at, and say why.
    const first = scene.attention[0];
    rot = rotTarget = 0;
    if (first) turnTo(first.flowId, first);
    renderRest(); draw();
  });
  window.addEventListener('resize', draw);

  resetWin();
  const first = scene.attention[0];
  if (first) { rot = G.rotationFor(first.ordinal, scene.world.threads.length, nowT()); rotTarget = rot; rotWhy = first; }
  draw();
  styleLive();
  animate();
})();
`;

const html = `<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ai-os — the bundle</title>
<style>${CSS}
@media (prefers-reduced-motion: no-preference){
  .pulse{animation:pulse 2.2s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:.35}50%{opacity:.9}}
}
</style>
<div class="wrap">
  <div class="stage">
    <header>
      <b>ai-os</b>
      <span class="sk" title="No core, no model, nothing stored. Every number comes from the projects' own artifacts.">Simulated</span>
      <span class="build" title="When this file was generated. If this has not changed, you are looking at a cached copy.">build ${new Date()
        .toISOString()
        .slice(5, 16)
        .replace("T", " ")}</span>
      <select id="scene">${payload
        .map((s) => `<option value="${s.id}">${s.label}</option>`)
        .join("")}</select>
      <span class="right">
        <span class="lvl" id="lvl"></span>
        <button id="zout" title="Zoom out — see more time">&minus;</button>
        <button id="zin" title="Zoom in — see more detail">+</button>
        <button id="znow" title="All of it, back to now">All</button>
      </span>
    </header>
    <svg id="stage" aria-label="A bundle of flows coiled around a time axis"></svg>
  </div>
  <aside id="panel"></aside>
</div>
<script>window.__BUNDLE__ = ${JSON.stringify(payload).replace(/</g, "\\u003c")};</script>
<script>window.__GEO__ = (() => {
${HELIX_JS}
  return { phase: phaseOf, rotationFor, bundle: bundleOf, bodies: bodiesAt,
           attention: attentionOf, justify: assertJustified };
})();</script>
<script>${JS}</script>
</html>`;

writeFileSync(out, html);
console.log(`wrote ${out} — ${(html.length / 1024).toFixed(0)} kB, self-contained`);
for (const s of payload) {
  const live = s.world.threads.reduce(
    (n, t) => n + t.rests.filter((r) => r.state === "running").length,
    0,
  );
  console.log(
    `  ${s.id}: ${s.world.threads.length} strand(s), ${s.world.lanes.length - 1} agent(s), ` +
      `${live} open now, front = ${s.attention[0]?.kind}`,
  );
}
