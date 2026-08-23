/**
 * The thread view, as a page you can touch.
 *
 *   cd ai-ui && node scripts/build-threads.ts --out threads.html
 *
 * ## Why this is SVG and not three.js
 *
 * The proposal that produced it came with a suggestion: vivid colour, and maybe
 * WebGL. The colour is right and it is here. The engine is the right *second*
 * move and the wrong first one, for a reason that has nothing to do with taste.
 *
 * The question worth answering first is **does standing on a rope help you say
 * what happened here** — and that is answerable in a day with paths and
 * transforms. If the answer is yes, an engine buys depth, a real camera, and a
 * thousand ropes instead of five, and it will have been paid for. If the answer
 * is no, an engine has made the failure expensive, beautiful, and much harder to
 * abandon.
 *
 * There is also a property this repository has already argued for and would lose
 * on the way: the demo is one self-contained file that opens from disk with no
 * server and no network. That is why anybody can check it. A 600 kB library and
 * a bundler is the first instalment of the bill `doc/08` says this pillar must
 * not run up before it has earned one.
 *
 * ## Colour, and how it can be loud without lying
 *
 * The desk's rule is that colour is state and evidence and nothing else. That
 * rule is not "be beige" — it is "one channel, one meaning". A dark ground buys
 * a second channel:
 *
 * - **Hue is identity.** Which thread this is. Five saturated colours, so you
 *   can follow one rope through eight lanes among four others.
 * - **Luminance and texture are state.** Carried is lit. Ignored goes dark from
 *   where it landed. Blocked ends against a bar. Open frays. Unrecorded is a
 *   *gap you see the background through*.
 *
 * Two orthogonal channels, each meaning exactly one thing, and the legend says
 * both. On the beige desk a saturated palette would have been noise; here it is
 * the information.
 *
 * ## What this is not
 *
 * A sketch. It does not replace the desk, it is not linked from the site, and it
 * has not been measured against anything. `doc/04`'s stopwatch is still unrun,
 * and it is the only thing that could say whether either surface is worth
 * having.
 */
import { writeFileSync } from "node:fs";
import { threadsOf, contentions, type ThreadWorld } from "../src/threads.ts";
import { traceOf } from "../src/trace.ts";
import { agentOfIntent } from "../src/server.ts";
import { DEMO_AT } from "../src/simulate.ts";
import { cochleaFlows } from "../src/cochlea-demo.ts";
import { cochleaProjectFlows } from "../src/cochlea-project.ts";
import { hemoFlows } from "../src/hemo-demo.ts";
import { memoryFlows } from "../src/memory-demo.ts";
import { dspFlows } from "../src/dsp-demo.ts";

const outIdx = process.argv.indexOf("--out");
const out = outIdx >= 0 ? process.argv[outIdx + 1]! : "threads.html";

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
      "Two chains, the same six agents, the same six steps — and thirty hours apart, which the " +
      "desk could not have told you. One came home. The other stops at AUDITOR, because a gate " +
      "declared before the run measured 2.592e-4 against a tolerance of 1.0e-4. The three threads " +
      "nearest now are the project being built, and one of their steps is still open.",
    world: threadsOf(
      docsOf([...cochleaFlows(DEMO_AT), ...cochleaProjectFlows(DEMO_AT)] as never),
    ),
  },
  {
    id: "hemo-verified",
    label: "hemo-verified — truth is not derivable",
    note:
      "No closed form exists here, so the judge itself is measured: 0.9056 composite against a " +
      "kill threshold written down first. The frayed rope is A4 — read on two machines, never " +
      "compared under conditions where disagreeing is defined.",
    world: threadsOf(docsOf(hemoFlows(DEMO_AT) as never)),
  },
  /**
   * The two scopes that carry a falsification the projects do not.
   *
   * Both are about a flow that reports cleanly and is wrong, which is the thing
   * this system exists to make visible — and each shows it as a *texture* the
   * thread view already has, where the desk needed a paragraph.
   */
  {
    id: "memory-lab",
    label: "memory lab — green and wrong",
    note:
      "Two threads index the same notes with the same agents, and both are green. One is wrong: " +
      "a step used nothing it was given, because its note claims 663 characters of a passage that " +
      "is 1,105 — so following its range lands on different words. The rope goes dark from where " +
      "that landed.",
    world: threadsOf(docsOf(memoryFlows(DEMO_AT) as never)),
  },
  {
    id: "signal-lab",
    label: "signal lab — ran and carried nothing",
    note:
      "The same claim one level down, about a step rather than a result: a stage ran, settled, " +
      "reported, and carried nothing forward. Its output is a flatline, and the prose about it " +
      "reads exactly like the prose about a clean band.",
    world: threadsOf(docsOf(dspFlows(DEMO_AT) as never)),
  },
];

const payload = SCENES.map((s) => ({
  id: s.id,
  label: s.label,
  note: s.note,
  world: s.world,
  contentions: contentions(s.world),
}));

const CSS = String.raw`
:root{
  /* Apple's dark system colours, as published for dark backgrounds. Used as
     identity — which thread — never as state. */
  --blue:#0A84FF; --green:#30D158; --indigo:#5E5CE6; --orange:#FF9F0A;
  --pink:#FF375F; --purple:#BF5AF2; --red:#FF453A; --teal:#64D2FF; --yellow:#FFD60A;
  --bg:#000000; --bg-2:#1C1C1E; --bg-3:#2C2C2E;
  --lane:#2C2C2E; --sep:#38383A;
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
  padding:14px 20px 22px;background:linear-gradient(180deg,#000 55%,transparent);
  pointer-events:none}
header>*{pointer-events:auto}
header b{font-weight:600;letter-spacing:-.01em}
header .sk{font-size:10px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  color:var(--orange);background:rgba(255,159,10,.12);border:1px solid rgba(255,159,10,.28);
  border-radius:999px;padding:3px 10px}
/* When this file was generated. The one mark that answers "am I looking at a
   cached copy", which is otherwise a network-tab question. */
header .build{font:10px var(--mono);color:var(--faint);background:var(--bg-2);
  border:1px solid var(--sep);border-radius:999px;padding:3px 9px;cursor:help}
header select{font:inherit;font-size:12px;background:var(--bg-2);color:var(--ink);
  border:1px solid var(--sep);border-radius:8px;padding:6px 10px}
header .zoom{margin-left:auto;display:flex;gap:6px;align-items:center}
header .zoom .lvl{font:11px var(--mono);color:var(--dim);min-width:74px;text-align:right}
svg{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
svg.grabbing{cursor:grabbing}
.lanebg{fill:none;stroke:var(--lane);stroke-width:1}
.lanelabel{font:11.5px var(--mono);fill:var(--faint)}
.lanelabel.human{fill:var(--ink);font-family:var(--sans);font-size:12.5px;font-weight:600}
/* A lane where something is happening right now. The only lane label that is lit. */
.lanelabel.live{fill:var(--ink)}
.lanelive{fill:none;stroke:var(--sep);stroke-width:1}

/* The rope. Two strokes: a wide translucent glow, a narrow bright line. */
.glow{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-width:12;opacity:.14}
.rope{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-width:2.6}
.rope.ignored,.glow.ignored{opacity:.20}
.rope.ignored{stroke-dasharray:2 5}
.rope.open{stroke-dasharray:9 5;opacity:.75}
.glow.open{opacity:.06}
/* Not yet begun: order is known, time is not. Drawn faint and hollow. */
.rope.pending,.glow.pending{opacity:.22}
.rope.pending{stroke-dasharray:2 6}
.rope.unknown,.glow.unknown{display:none}
.gap{stroke:var(--faint);stroke-width:1;stroke-dasharray:1 4;opacity:.7}
.stop{stroke-width:3.5;stroke-linecap:butt}

/* Running: the one thing that moves on its own, and it moves because a step is
   open right now. The dash offset is animated in JS from the same clock as everything
   else, so if nothing is running nothing on this surface moves at all. */
.rope.running{stroke-dasharray:14 9;stroke-width:3.2}
.glow.running{opacity:.30}

.bead{stroke:#000;stroke-width:2}
.bead.pendingbead{fill:none;stroke-width:1.5}
.beadhit{fill:transparent;cursor:pointer}
.pkt{stroke:#000;stroke-width:1.25}
.braid{fill:none;stroke:#fff;stroke-width:1;opacity:.22}
.axis{stroke:var(--sep);stroke-width:1}
.axistext{font:10px var(--mono);fill:var(--faint)}
.nowline{stroke:var(--ink);stroke-width:1;opacity:.30}
.nowtext{font:10px var(--mono);fill:var(--dim)}
.dimmed{opacity:.14}
.sel .rope{stroke-width:4.2}

/* A collapsed band: what a thread stands for when the window is too wide to
   draw its steps. Semantic zoom, not a smaller picture. */
.band{opacity:.85}
.bandtext{font:10px var(--mono);fill:var(--dim)}

aside{background:var(--bg-2);border-left:1px solid var(--sep);padding:18px 20px;overflow:auto}
aside h2{margin:0 0 2px;font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint)}
aside h3{margin:9px 0 3px;font-size:16px;font-weight:600;letter-spacing:-.02em}
aside .sub{font:11px var(--mono);color:var(--dim);margin-bottom:12px}
aside .fld{display:grid;grid-template-columns:88px 1fr;gap:3px 10px;font-size:11.5px;margin:0 0 5px}
aside .fld .k{color:var(--faint)}
aside .at{display:block;font-family:var(--mono);font-size:10.5px;color:var(--teal);
  text-decoration:none;border:0;background:none;padding:0;margin-top:2px;cursor:pointer;
  text-align:left;border-bottom:1px dotted rgba(100,210,255,.4)}
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
button[aria-selected="true"]{background:rgba(10,132,255,.18);border-color:rgba(10,132,255,.5);
  color:#9CCEFF}
aside .key{margin-top:16px;border-top:1px solid var(--sep);padding-top:13px}
aside .key h4{margin:0 0 7px;font-size:10px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--faint);font-weight:600}
aside .key .row{display:flex;align-items:center;gap:9px;padding:3px 0;font-size:11.5px;line-height:1.4}
aside .key svg{width:44px;height:10px;flex:0 0 44px}
aside .note{margin-top:14px;font-size:11.5px;line-height:1.55;color:var(--dim)}
aside .warn{margin-top:12px;font-size:11px;line-height:1.5;color:var(--orange);
  background:rgba(255,159,10,.08);border:1px solid rgba(255,159,10,.22);border-radius:8px;
  padding:9px 11px}
.tourbar{position:fixed;left:20px;bottom:20px;z-index:400;display:flex;align-items:center;gap:12px;
  background:var(--bg-2);border:1px solid var(--sep);border-radius:999px;
  box-shadow:0 8px 28px rgba(0,0,0,.6);padding:8px 16px 8px 8px;
  max-width:min(760px,calc(100vw - 400px))}
.tourbar button{border-radius:999px;background:var(--ink);color:#000;border-color:var(--ink);
  font-weight:650;padding:6px 16px}
.tourbar button:hover{background:#fff;border-color:#fff}
.tourbar span{font-size:12px;line-height:1.45;color:var(--dim)}

aside .bytes{font-family:var(--mono);font-size:10.5px;line-height:1.6;background:#000;
  border:1px solid var(--sep);border-radius:8px;padding:10px 11px;margin-top:9px;
  max-height:220px;overflow:auto;white-space:pre-wrap;word-break:break-word;color:#C7C7CC}
`;

const JS = String.raw`
(() => {
  const S = window.__THREADS__;
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a) => { const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]); return e; };

  /**
   * Hue is identity. Apple's dark system colours, in a fixed order so a thread
   * keeps its colour across a zoom, a pan and a scene change.
   */
  const HUES = ['#0A84FF','#FF9F0A','#30D158','#BF5AF2','#FF375F','#64D2FF','#5E5CE6','#FFD60A'];

  let scene = S[0], sel = null, mode = 'read';
  const svg = document.getElementById('stage');
  const panel = document.getElementById('panel');
  const pick = document.getElementById('scene');
  const lvl = document.getElementById('lvl');

  const M = { l: 172, r: 40, t: 84, b: 64 };
  let W = 0, H = 0, laneY = {};

  /**
   * The window: which slice of time is on screen.
   *
   * view.t1 is the right edge and it is **now** — the latest moment anything
   * was recorded. Panning left goes back. 'view.w' is how many minutes fit, and
   * zooming changes it. Both are in the same minutes-since-t0 units the layout
   * produces, so nothing here has to know what a date is.
   */
  let view = { t1: 0, w: 0 };
  const resetView = () => { view = { t1: scene.world.span, w: scene.world.span * 1.04 }; };

  const measure = () => {
    const r = svg.getBoundingClientRect();
    W = r.width; H = r.height;
    const lanes = scene.world.lanes;
    const usable = H - M.t - M.b;
    const gap = lanes.length > 1 ? usable / (lanes.length - 1) : 0;
    laneY = {};
    lanes.forEach((l, i) => { laneY[l.id] = M.t + i * gap; });
  };
  const xOf = (t) => M.l + ((t - (view.t1 - view.w)) / view.w) * (W - M.l - M.r);
  const tOf = (px) => (view.t1 - view.w) + ((px - M.l) / (W - M.l - M.r)) * view.w;

  const OFF = 7;
  const yOf = (lane, t) => {
    const n = scene.world.threads.length;
    if (n < 2) return laneY[lane];
    return laneY[lane] + (t.ordinal - (n - 1) / 2) * OFF;
  };
  const hueOf = (t) => HUES[t.ordinal % HUES.length];

  const restPath = (r, t) => 'M' + xOf(r.x0) + ' ' + yOf(r.lane, t) + ' H' + xOf(r.x1);
  const crossPath = (c, t) => {
    const x0 = xOf(c.x0), x1 = xOf(c.x1);
    const y0 = yOf(c.from, t), y1 = yOf(c.to, t);
    const k = Math.max(14, (x1 - x0) * 0.55);
    return 'M' + x0 + ' ' + y0 + ' C' + (x0 + k) + ' ' + y0 + ' ' + (x1 - k) + ' ' + y1 +
           ' ' + x1 + ' ' + y1;
  };

  /**
   * Is a step open right now?
   *
   * The only source of motion on this surface. 'running' means an attempt
   * started and never closed, which is a fact in the store — not a mood, not a
   * decoration, and not something the renderer may decide on its own.
   */
  const isLive = (r) => r.state === 'running';
  const isPending = (r) => r.state === 'pending' || r.state === 'waiting' || r.state === 'draft';
  const liveCount = () =>
    scene.world.threads.reduce((n, t) => n + t.rests.filter(isLive).length, 0);

  /**
   * Semantic zoom.
   *
   * A thread whose whole life is narrower than this many pixels is not drawn
   * step by step — there is nothing to see and the marks lie about their own
   * precision. It collapses to one band that **says what it stands for**: how
   * many steps, over how long. That is 'zoom.ts''s rule, which exists because a
   * viewer sampling at one rate cannot faithfully observe change faster than
   * half of it, and a picture that pretends otherwise invites a reader to see
   * structure in aliasing.
   */
  const COLLAPSE_PX = 58;

  const fmtDur = (min) => {
    if (min < 1) return Math.round(min * 60) + 's';
    if (min < 90) return Math.round(min) + 'm';
    if (min < 60 * 48) return (min / 60).toFixed(1) + 'h';
    return (min / 1440).toFixed(1) + 'd';
  };
  const fmtAgo = (min) => min <= 0.5 ? 'now' : fmtDur(min) + ' ago';

  function draw() {
    measure();
    svg.replaceChildren();
    const w = scene.world;
    const live = liveCount();

    // The axis, and 'now' at the right edge.
    const y = H - M.b + 20;
    svg.appendChild(el('line', { class: 'axis', x1: M.l, y1: y, x2: W - M.r, y2: y }));
    const ticks = 6;
    for (let i = 0; i <= ticks; i += 1) {
      const t = (view.t1 - view.w) + (view.w * i) / ticks;
      const px = xOf(t);
      svg.appendChild(el('line', { class: 'axis', x1: px, y1: y, x2: px, y2: y + 4 }));
      const lab = el('text', { class: 'axistext', x: px, y: y + 16, 'text-anchor': 'middle' });
      lab.textContent = fmtAgo(w.span - t);
      svg.appendChild(lab);
    }
    const nowX = xOf(w.span);
    if (nowX < W - M.r + 2) {
      svg.appendChild(el('line', { class: 'nowline', x1: nowX, y1: M.t - 26, x2: nowX, y2: y }));
      const nl = el('text', { class: 'nowtext', x: nowX - 6, y: M.t - 32, 'text-anchor': 'end' });
      nl.textContent = live ? live + ' running · now' : 'now · nothing running';
      svg.appendChild(nl);
    }

    // Lanes. A lane label is lit only while that agent is holding something open.
    const liveLanes = new Set();
    for (const t of w.threads) for (const r of t.rests) if (isLive(r)) liveLanes.add(r.lane);
    for (const l of w.lanes) {
      svg.appendChild(el('line', { class: 'lanebg', x1: M.l - 12, y1: laneY[l.id],
        x2: W - M.r, y2: laneY[l.id] }));
      const t = el('text', {
        class: 'lanelabel' + (l.kind === 'human' ? ' human' : '') + (liveLanes.has(l.id) ? ' live' : ''),
        x: M.l - 20, y: laneY[l.id] + 4, 'text-anchor': 'end' });
      t.textContent = l.label;
      svg.appendChild(t);
    }

    for (const c of scene.contentions) {
      const x0 = xOf(c.x0), x1 = xOf(c.x1);
      if (x1 - x0 < 3) continue;
      // The braid: where two threads hold one agent at one moment, the ropes
      // twist. Drawn from the contention the layout found, never decoratively.
      const yy = laneY[c.lane], n = 5, d = [];
      for (let i = 0; i <= n * 8; i += 1) {
        const u = i / (n * 8), px = x0 + (x1 - x0) * u;
        d.push((i ? 'L' : 'M') + px + ' ' + (yy + Math.sin(u * Math.PI * n) * OFF));
      }
      svg.appendChild(el('path', { class: 'braid', d: d.join(' ') }));
    }

    for (const t of w.threads) {
      const hue = hueOf(t);
      const g = el('g', { 'data-flow': t.flowId });
      if (sel && sel.flowId && sel.flowId !== t.flowId) g.setAttribute('class', 'dimmed');
      else if (sel && sel.flowId === t.flowId) g.setAttribute('class', 'sel');

      const x0 = xOf(Math.min(...t.rests.map((r) => r.x0)));
      const x1 = xOf(Math.max(...t.rests.map((r) => r.x1)));

      // Off screen entirely: draw nothing rather than clipping a shape into a
      // mark that means something else.
      if (x1 < M.l - 40 || x0 > W - M.r + 40) { svg.appendChild(g); continue; }

      if (x1 - x0 < COLLAPSE_PX) {
        // Collapsed. One band, and a label that states what it stands for.
        const yTop = Math.min(...t.rests.map((r) => yOf(r.lane, t)));
        const yBot = Math.max(...t.rests.map((r) => yOf(r.lane, t)));
        const cx = (x0 + x1) / 2;
        g.appendChild(el('rect', { class: 'band', x: cx - 3, y: yTop, width: 6,
          height: Math.max(6, yBot - yTop), rx: 3, fill: hue, opacity: .55 }));
        const dur = Math.max(...t.rests.map((r) => r.x1)) - Math.min(...t.rests.map((r) => r.x0));
        const lab = el('text', { class: 'bandtext', x: cx, y: yTop - 8, 'text-anchor': 'middle' });
        lab.textContent = t.rests.length + ' steps · ' + fmtDur(dur);
        g.appendChild(lab);
        const hit = el('rect', { class: 'beadhit', x: cx - 14, y: yTop - 14,
          width: 28, height: Math.max(28, yBot - yTop + 28) });
        hit.addEventListener('pointerdown', (ev) => { ev.stopPropagation();
          zoomTo(Math.min(...t.rests.map((r) => r.x0)), Math.max(...t.rests.map((r) => r.x1))); });
        g.appendChild(hit);
        svg.appendChild(g);
        continue;
      }

      const put = (d, cls, meta) => {
        if (cls === 'unknown') { svg.appendChild(el('path', { class: 'gap', d: d })); return; }
        g.appendChild(el('glow' in {} ? 'path' : 'path', { class: 'glow ' + cls, d: d, stroke: hue }));
        g.appendChild(el('path', { class: 'rope ' + cls, d: d, stroke: hue }));
        if (meta) {
          const hit = el('path', { class: 'beadhit', d: d, stroke: 'transparent',
            'stroke-width': 18, fill: 'none' });
          hit.addEventListener('pointerdown', (ev) => { ev.stopPropagation(); select(meta); });
          g.appendChild(hit);
        }
      };

      for (const c of t.crosses)
        put(crossPath(c, t), c.state, { kind: 'cross', ...c, hue: hue, title: t.title });
      for (const r of t.rests) {
        const cls = isLive(r) ? 'running' : isPending(r) ? 'pending' : 'carried';
        put(restPath(r, t), cls, { kind: 'rest', ...r, hue: hue, title: t.title });
        const cx = (xOf(r.x0) + xOf(r.x1)) / 2;
        const b = el('circle', {
          class: 'bead' + (isPending(r) ? ' pendingbead' : ''),
          cx: cx, cy: yOf(r.lane, t), r: 4.5,
          fill: isPending(r) ? 'none' : (r.state === 'failed' || r.state === 'blocked') ? '#FF453A' : hue,
          stroke: isPending(r) ? hue : '#000' });
        g.appendChild(b);
      }

      if (!t.delivered) {
        const last = t.crosses[t.crosses.length - 1];
        const yy = yOf(last.to, t), x = xOf(last.x1);
        g.appendChild(el('line', { class: 'stop', x1: x + 3, y1: yy - 9, x2: x + 3, y2: yy + 9,
          stroke: last.state === 'open' ? '#98989F' : '#FF453A' }));
      }
      svg.appendChild(g);
    }

    lvl.textContent = fmtDur(view.w) + ' wide';
    if (!sel) renderRest();
    /**
     * Restart the loop if this redraw brought something live into view.
     *
     * The loop cancels itself when it finds nothing running, which is the point
     * — a still surface is a true statement. But a zoom that resolves a
     * collapsed band into its steps *creates* a running rope, and nothing was
     * waking the loop back up: the one open step in the demo sat there
     * motionless, so the surface said 'nothing is happening' about something
     * that was. Waking it here is the only place that catches every case,
     * because every case ends in a redraw.
     */
    animate();
  }

  /**
   * The one animation loop, and it stops.
   *
   * It runs only while something is genuinely open. When the last running step
   * closes there is nothing to animate and the loop cancels itself — so a still
   * surface is a true statement that nothing is happening, which is the only
   * thing that makes the moving surface worth believing.
   */
  let raf = null, phase = 0;
  const tick = () => {
    const running = svg.querySelectorAll('.rope.running');
    if (!running.length) { raf = null; return; }
    phase = (phase + 0.9) % 23;
    for (const p of running) p.setAttribute('stroke-dashoffset', String(-phase));
    for (const p of svg.querySelectorAll('.glow.running'))
      p.setAttribute('opacity', String(0.24 + 0.12 * Math.sin(phase / 23 * Math.PI * 2)));
    raf = requestAnimationFrame(tick);
  };
  const animate = () => { if (!raf) raf = requestAnimationFrame(tick); };

  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
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

  const select = (meta) => { sel = meta; mode = 'read'; draw(); renderPanel(); animate(); };

  const swatch = (cls, hue) =>
    '<svg viewBox="0 0 44 10"><path class="rope ' + cls + '" d="M2 5 H42" stroke="' +
    (hue || '#0A84FF') + '"/></svg>';

  const keyHtml = () => {
    const live = liveCount();
    return '<div class="key">' +
      '<h4>Hue — which thread</h4>' +
      scene.world.threads.map((t) =>
        '<div class="row">' + swatch('', hueOf(t)) + '<span>' + esc(t.title) + '</span></div>').join('') +
      '<h4 style="margin-top:12px">Texture — what happened</h4>' +
      '<div class="row">' + swatch('') + '<span><b>carried</b> — it moved and the next step used it</span></div>' +
      '<div class="row">' + swatch('ignored') + '<span><b>carried nothing forward</b> — it arrived and nothing used it</span></div>' +
      '<div class="row">' + swatch('open') + '<span><b>no verdict yet</b> — it arrived and nothing has been decided</span></div>' +
      '<div class="row">' + swatch('pending') + '<span><b>not begun</b> — its order is known, its time is not</span></div>' +
      '<div class="row"><svg viewBox="0 0 44 10"><path class="gap" d="M2 5 H42"/></svg>' +
        '<span><b>unrecorded</b> — nothing was recorded, so no rope is drawn</span></div>' +
      '<h4 style="margin-top:12px">Motion — what is happening now</h4>' +
      '<div class="row">' + swatch('running') + '<span><b>light travels</b> a step that is open right now' +
        (live ? '' : ' — <b>nothing is</b>, so nothing on this surface is moving') + '</span></div>' +
      '<div class="row"><svg viewBox="0 0 44 10"><path class="braid" d="M2 5 Q13 1 24 5 T44 5"/></svg>' +
        '<span><b>the braid</b> — two threads holding one agent at one moment' +
        (scene.contentions.length ? '' : ' — none here') + '</span></div>' +
      '</div>';
  };

  const basisHtml = () => scene.world.basis === 'clock'
    ? ''
    : '<div class="warn">This scene is drawn on <b>step order</b>, not on a clock: at least one ' +
      'attempt in it recorded no start time. Every step is the same width, and none of those ' +
      'widths is a duration.</div>';

  const noteHtml = () => '<div class="note">' + esc(scene.note) + '</div>';

  function renderRest() {
    if (sel) return;
    const live = liveCount();
    const total = scene.world.threads.reduce((n, t) => n + t.rests.length, 0);
    panel.innerHTML = '<h2>This scene</h2>' +
      '<h3>' + (live ? live + ' step' + (live === 1 ? '' : 's') + ' open right now' : 'nothing is running') + '</h3>' +
      '<div class="sub">' + scene.world.threads.length + ' threads · ' + total + ' steps · ' +
        scene.world.lanes.length + ' lanes · ' + fmtDur(scene.world.span) + ' of history</div>' +
      '<div class="note" style="margin-top:0">Drag to go back. Scroll or pinch to zoom. Click a rope, ' +
      'a bead, or a collapsed band.</div>' +
      basisHtml() + keyHtml() + noteHtml();
    wire();
  }

  function renderPanel() {
    if (!sel) { renderRest(); return; }
    const isCross = sel.kind === 'cross';
    const nice = (n) => String(n).replace('@human', 'you');
    const head = '<h2>' + (isCross ? 'A handoff' : 'A step') + '</h2>' +
      '<h3>' + esc(isCross ? nice(sel.from) + ' → ' + nice(sel.to) : sel.lane + ' · step ' + sel.index) + '</h3>' +
      '<div class="sub">' + esc(sel.title) + ' · ' + fmtDur(sel.x1 - sel.x0) + '</div>' +
      '<div class="sw"><button id="m-read"' + (mode==='read'?' aria-selected="true"':'') + '>Read it</button>' +
      '<button id="m-agent"' + (mode==='agent'?' aria-selected="true"':'') + '>Ask an agent</button></div>';

    let body;
    if (mode === 'read') {
      body = fld('state', isCross ? sel.state : sel.state) +
        fld(scene.world.basis === 'clock' ? 'took' : 'width',
            scene.world.basis === 'clock' ? fmtDur(sel.x1 - sel.x0) : 'one slot — this scene has no clock') +
        (isCross ? fld('because', sel.because) : '') +
        (sel.digest ? fld('observation', sel.digest, sel.source || undefined)
                    : fld('carried', 'nothing recorded — this is not a claim that nothing moved')) +
        (!isCross && sel.result ? fld('said', sel.result) : '');
    } else {
      const at = 'flow:' + sel.flowId + '#step-' + (isCross ? sel.toIndex : sel.index);
      const cites = sel.source ? [sel.source, at] : [at];
      let f;
      if (isCross && sel.from === '@human')
        f = { verdict: 'ok', cites: ['flow:' + sel.flowId],
              says: 'You asked for this. It is the one handoff on the thread that is not in question — everything after it is.' };
      else if (isCross && sel.to === '@human')
        f = { verdict: 'ok', cites: cites,
              says: 'The thread finished and came back to you. That is a statement about delivery, not about whether the answer is right — a gate says that, and this does not.' };
      else if (isCross && sel.state === 'unknown')
        f = { verdict: 'unknown', cites: [],
              says: 'Nothing was recorded for this hop, so there is nothing to read. That is not a pass: it is the absence of evidence either way.' };
      else if (isCross && sel.state === 'open')
        f = { verdict: 'unknown', cites: cites,
              says: 'It reached ' + nice(sel.to) + ' and no verdict has been reached. ' + sel.because + '. There is no result here to agree or disagree with — only an open question.' };
      else if (isCross && sel.state === 'ignored')
        f = { verdict: 'problem', cites: cites,
              says: 'It arrived and ' + sel.to + ' used none of it. ' + sel.because + '.' };
      else if (isCross && sel.state === 'blocked')
        f = { verdict: 'problem', cites: cites,
              says: 'It arrived, ' + sel.to + ' ran, and it did not pass. ' + sel.because + '.' };
      else if (isCross)
        f = { verdict: 'ok', cites: cites,
              says: nice(sel.from) + ' closed with an observation and ' + sel.to + ' read it. ' + sel.because + '.' };
      else if (isPending(sel))
        f = { verdict: 'unknown', cites: [],
              says: 'This step has not begun. Its place in the order is known and its time is not, so it is drawn after the last step that ran rather than anywhere on the clock.' };
      else if (isLive(sel))
        f = { verdict: 'unknown', cites: cites,
              says: sel.lane + ' has had this open for ' + fmtDur(sel.x1 - sel.x0) + ' and has not closed it. There is no observation yet, because it has not finished.' };
      else if (!sel.digest)
        f = { verdict: 'unknown', cites: [], says: 'This step recorded no observation, so there is nothing here to read.' };
      else
        f = { verdict: (sel.state === 'failed' || sel.state === 'blocked') ? 'problem' : 'ok', cites: cites,
              says: sel.lane + ' at step ' + sel.index + ' is ' + sel.state + ', took ' + fmtDur(sel.x1 - sel.x0) + ', and recorded ' + sel.digest + '.' };
      body = finding(f);
    }
    panel.innerHTML = head + body + basisHtml() + keyHtml() + noteHtml();
    wire();
  }

  const NOTES = {};
  const wire = () => {
    const r = panel.querySelector('#m-read'), a = panel.querySelector('#m-agent');
    if (r) r.onclick = () => { mode = 'read'; renderPanel(); };
    if (a) a.onclick = () => { mode = 'agent'; renderPanel(); };
    for (const b of panel.querySelectorAll('button.at'))
      b.onclick = () => {
        const at = b.dataset.open;
        const html = '<div class="bytes">' + esc(at) + '\n\n' + esc(NOTES[at] ||
          'This address is recorded on the segment above. This page has no filesystem, so the ' +
          'address is shown rather than the bytes — and no contents are invented for it.') + '</div>';
        const box = panel.querySelector('.bytes');
        if (box) box.outerHTML = html; else panel.insertAdjacentHTML('beforeend', html);
        wire();
      };
  };

  /** Zoom to a span, with a margin, clamped so you cannot lose the threads. */
  function zoomTo(a, b) {
    const pad = Math.max(2, (b - a) * 0.25);
    view = { t1: b + pad, w: Math.max(2, (b - a) + pad * 2) };
    sel = null; draw(); animate();
  }

  // ---- pan and zoom --------------------------------------------------------
  let drag = null;
  svg.addEventListener('pointerdown', (ev) => {
    drag = { x: ev.clientX, t1: view.t1 };
    svg.classList.add('grabbing');
    svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const perPx = view.w / (W - M.l - M.r);
    view.t1 = drag.t1 - (ev.clientX - drag.x) * perPx;
    draw();
  });
  const endDrag = (ev) => {
    if (drag && Math.abs(ev.clientX - drag.x) < 3) { sel = null; draw(); renderRest(); }
    drag = null; svg.classList.remove('grabbing');
  };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', () => { drag = null; svg.classList.remove('grabbing'); });

  /**
   * Zoom around the pointer, not around the centre.
   *
   * Zooming toward the middle moves whatever you were looking at, which makes a
   * wide history impossible to explore: you aim, you zoom, and the thing you
   * aimed at has left.
   */
  svg.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const anchor = tOf(ev.clientX - svg.getBoundingClientRect().left);
    const k = Math.exp(ev.deltaY * 0.0016);
    const nw = Math.min(scene.world.span * 1.6, Math.max(0.6, view.w * k));
    view.t1 = anchor + (view.t1 - anchor) * (nw / view.w);
    view.w = nw;
    draw();
  }, { passive: false });

  document.getElementById('zin').onclick = () => { view.w = Math.max(0.6, view.w / 1.8); draw(); };
  document.getElementById('zout').onclick = () => {
    view.w = Math.min(scene.world.span * 1.6, view.w * 1.8); draw();
  };
  document.getElementById('znow').onclick = () => { resetView(); sel = null; draw(); renderRest(); };

  pick.addEventListener('change', () => {
    scene = S.find((x) => x.id === pick.value) || S[0];
    sel = null; resetView(); draw(); renderRest(); animate();
  });
  window.addEventListener('resize', draw);

  resetView();
  draw();
  animate();

  /**
   * Play — the demo, driving itself.
   *
   * Same rule the desk's tour lives under, and it is the only thing that makes a
   * tour honest: every beat below **operates the real controls** — the real zoom
   * buttons, the real scene selector, the real segments — and then lets the page
   * react however it reacts. Nothing here draws a frame, animates a fake, or
   * asserts an outcome.
   *
   * The property that buys: **if the surface breaks, the tour breaks.** A
   * scripted animation of a product is a second implementation of it, and it
   * goes on looking correct for as long as nobody checks. This cannot.
   *
   * And it never fights the person watching: any real pointer or key event stops
   * it where it is and leaves the view exactly as the tour left it.
   */
  (() => {
    const bar = document.createElement('div');
    bar.className = 'tourbar';
    bar.innerHTML = '<button id="tourgo">Play</button><span id="tourcap">' +
      'Watch it use itself — every beat is a real control, not a recording.</span>';
    document.body.appendChild(bar);

    let running = false, stop = false;
    const cap = (t) => { document.getElementById('tourcap').textContent = t; };
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const halt = () => {
      if (!running) return;
      stop = true;
      cap('Stopped — it is yours. Drag, zoom, click anything.');
    };
    for (const ev of ['pointerdown', 'keydown', 'wheel'])
      window.addEventListener(ev, (e) => { if (e.isTrusted) halt(); }, true);

    const press = async (id, n) => {
      for (let i = 0; i < (n || 1) && !stop; i += 1) {
        document.getElementById(id).click();
        await sleep(190);
      }
    };
    /** Select a real segment by clicking the hit path the renderer drew. */
    const clickSeg = async (pred) => {
      const paths = [...document.querySelectorAll('#stage .beadhit')];
      const hit = paths.find(pred) || paths[0];
      if (!hit) return;
      hit.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerId: 1 }));
      await sleep(240);
    };

    const beats = [
      async () => {
        cap('Horizontal is time. The right edge is now, and every thread starts with a person asking for it.');
        await press('znow');
        await sleep(3200);
      },
      async () => {
        cap('Three days wide, no thread is drawn step by step — there would be nothing to see. Each collapses to a band that says what it stands for.');
        await sleep(3600);
      },
      async () => {
        cap('Two of those bands are the same six agents doing the same six steps. Thirty hours apart. Nothing before this could have told you that.');
        await sleep(3800);
      },
      async () => {
        cap('Zoom in and a band resolves into its steps. A step is as wide as it took; the space between two is time nobody was working.');
        await press('zin', 4);
        await sleep(3200);
      },
      async () => {
        cap('One rope stops against a bar and never returns. A gate declared before the run measured 2.592e-4 against a tolerance of 1.0e-4, so it cannot freeze.');
        await sleep(3600);
      },
      async () => {
        cap('Closer still: one step is open right now, and it is the only thing here that moves. And where nothing was recorded at all, no rope is drawn — you see the dark through it, because did not run is not passed.');
        await press('zin', 2);
        await sleep(4600);
      },
      async () => {
        cap('Click any segment and the panel is about it. Ask an agent instead of reading it yourself — and every sentence it gives back carries the address it read.');
        await clickSeg((p) => true);
        await sleep(900);
        const a = document.getElementById('m-agent');
        if (a) a.click();
        await sleep(4000);
      },
      async () => {
        cap('The other project. No closed form exists there, so the judge itself goes on trial — and one rope frays instead of stopping, because no verdict has been reached.');
        const sel = document.getElementById('scene');
        sel.value = 'hemo-verified';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        await sleep(1400);
        await press('zin', 3);
        await sleep(4200);
      },
      async () => {
        cap('0.9056 against a kill threshold written down first. Alone, six of its seven oracles are near a coin flip — A5 is 0.5209. That is the number nobody publishes.');
        await sleep(4200);
      },
      async () => {
        cap('One more. Two threads index the same notes with the same agents, and both are green all the way through.');
        const sel = document.getElementById('scene');
        sel.value = 'memory-lab';
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        await sleep(1200);
        await press('zin', 3);
        await sleep(4200);
      },
      async () => {
        cap('Two threads, the same agents, both green — and one used nothing it was given. The rope goes dark from where that landed. That is the finding this whole system exists to make visible.');
        await sleep(4600);
      },
      async () => {
        cap('It is yours. Drag left to go back, scroll to zoom, click any rope or bead.');
        await sleep(2000);
      },
    ];

    const play = async () => {
      if (running) { halt(); return; }
      running = true; stop = false;
      document.getElementById('tourgo').textContent = 'Stop';
      for (const beat of beats) {
        if (stop) break;
        try { await beat(); }
        catch (e) { cap('The tour hit something the page did not expect: ' + e.message); break; }
      }
      running = false; stop = false;
      document.getElementById('tourgo').textContent = 'Play again';
    };
    document.getElementById('tourgo').onclick = play;
  })();
})();
`;

const html = `<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ai-os — threads of thought</title>
<style>${CSS}</style>
<div class="wrap">
  <div class="stage">
    <header>
      <b>ai-os</b>
      <span class="sk">Simulated — no core, no model, nothing stored</span>
      <span class="build" title="When this file was generated. If this has not changed, you are looking at a cached copy.">build ${new Date()
        .toISOString()
        .slice(5, 16)
        .replace("T", " ")}</span>
      <select id="scene">${payload
        .map((s) => `<option value="${s.id}">${s.label}</option>`)
        .join("")}</select>
      <span class="zoom">
        <span class="lvl" id="lvl"></span>
        <button id="zout" title="Zoom out — see more time">−</button>
        <button id="zin" title="Zoom in — see more detail">+</button>
        <button id="znow" title="Back to now, all of it">All</button>
      </span>
    </header>
    <svg id="stage" aria-label="Threads of thought between the agents and you"></svg>
  </div>
  <aside id="panel"></aside>
</div>
<script>window.__THREADS__ = ${JSON.stringify(payload).replace(/</g, "\\u003c")};</script>
<script>${JS}</script>
</html>`;

writeFileSync(out, html);
console.log(`wrote ${out} — ${(html.length / 1024).toFixed(0)} kB, self-contained`);
for (const s of payload) {
  const seg = s.world.threads.reduce((n, t) => n + t.rests.length + t.crosses.length, 0);
  console.log(
    `  ${s.id}: ${s.world.lanes.length} lane(s), ${s.world.threads.length} thread(s), ` +
      `${seg} segment(s), ${s.contentions.length} contention(s)`,
  );
}
