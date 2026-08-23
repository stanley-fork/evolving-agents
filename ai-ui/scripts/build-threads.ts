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
import { hemoFlows } from "../src/hemo-demo.ts";

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
      "Two chains, the same six agents, the same six steps. One came home. The other stops at " +
      "AUDITOR, because a gate declared before the run measured 2.592e-4 against a tolerance of 1.0e-4.",
    world: threadsOf(docsOf(cochleaFlows(DEMO_AT) as never)),
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
  --bg:#0B0E13; --bg-2:#11151C; --lane:#1B212B; --lane-lit:#2A3340;
  --ink:#E8EDF3; --dim:#8A96A6; --faint:#5A6675;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Monaco,"Roboto Mono",monospace;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
  font:13.5px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
.wrap{display:grid;grid-template-columns:1fr 336px;height:100%}
.stage{position:relative;overflow:hidden}
header{position:absolute;top:0;left:0;right:0;z-index:5;display:flex;gap:14px;align-items:center;
  padding:14px 20px;background:linear-gradient(180deg,var(--bg) 62%,transparent)}
header b{font-weight:600;letter-spacing:-.01em}
header .sk{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  color:#FFC46B;background:rgba(255,176,32,.10);border:1px solid rgba(255,176,32,.30);
  border-radius:999px;padding:3px 10px}
header select{margin-left:auto;font:inherit;font-size:12px;background:var(--bg-2);color:var(--ink);
  border:1px solid var(--lane-lit);border-radius:6px;padding:5px 9px}
svg{display:block;width:100%;height:100%;cursor:crosshair;touch-action:none}
.lanebg{fill:none;stroke:var(--lane);stroke-width:1}
.lanelabel{font:11.5px var(--mono);fill:var(--faint)}
.lanelabel.human{fill:#9FB2C7;font-family:var(--sans);font-size:12px;font-weight:600}
.lanehit{fill:transparent}
.lanehit:hover + .lanelabel{fill:var(--ink)}

/* The rope.
   Two strokes: a wide translucent one for the glow, a narrow bright one for the
   line. No filter — a blur on five ropes is free and on a thousand is not, and
   the shape of this cost should be visible from the sketch. */
.glow{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-width:11;opacity:.16}
.rope{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-width:2.6}
/* Landed and nothing used it: from here on the rope is nearly out. */
.rope.ignored,.glow.ignored{opacity:.18}
.rope.ignored{stroke-dasharray:2 5}
/* Held, no verdict: it frays. */
.rope.open{stroke-dasharray:9 5;opacity:.7}
.glow.open{opacity:.07}
/* Ran and did not pass: full brightness up to a stop. The rope is not dimmed —
   a negative result is a result, and dimming it would hide the most informative
   thing on the page. */
.rope.blocked{}
/* Unrecorded is not drawn at all. See the gap rule below. */
.rope.unknown,.glow.unknown{display:none}
.gap{stroke:var(--faint);stroke-width:1;stroke-dasharray:1 4;opacity:.75}
.gapcap{fill:var(--faint)}
.stop{stroke-width:3.5;stroke-linecap:butt}

.bead{stroke:var(--bg);stroke-width:2}
.beadhit{fill:transparent;cursor:pointer}
.pip{stroke:none}
.contend{fill:none;stroke:#fff;stroke-width:1;opacity:.20}
.tick{stroke:var(--lane);stroke-width:1}
.playhead{stroke:#fff;stroke-width:1;opacity:.5}
.playgrab{fill:transparent;cursor:ew-resize}
.now{font:10px var(--mono);fill:var(--dim)}
.title{font:12px var(--sans);font-weight:600}
.sel .rope{stroke-width:4}
.dimmed{opacity:.16}

/* The panel. Same two positions as the desk's Inspector, same rule about
   citations — the vocabulary transfers or the metaphor is a rewrite. */
aside{background:var(--bg-2);border-left:1px solid var(--lane);padding:18px 20px;overflow:auto}
aside h2{margin:0 0 2px;font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
  color:var(--faint)}
aside h3{margin:10px 0 3px;font-size:15px;font-weight:600;letter-spacing:-.01em}
aside .sub{font:11px var(--mono);color:var(--dim);margin-bottom:12px}
aside .fld{display:grid;grid-template-columns:88px 1fr;gap:3px 10px;font-size:11.5px;margin:0 0 5px}
aside .fld .k{color:var(--faint)}
aside .at{display:block;font-family:var(--mono);font-size:10.5px;color:#6FC3FF;text-decoration:none;
  border-bottom:1px dotted rgba(111,195,255,.45);margin-top:2px;cursor:pointer;
  background:none;border-left:0;border-right:0;border-top:0;padding:0;text-align:left}
aside .fnd{border:1px solid var(--lane-lit);border-radius:8px;padding:11px 13px;margin-top:12px;
  background:rgba(255,255,255,.02)}
aside .fnd .vd{font-size:10px;letter-spacing:.08em;text-transform:uppercase;font-weight:700}
aside .fnd.ok .vd{color:#5BE39B}
aside .fnd.problem .vd{color:#FF6B8A}
aside .fnd.unknown{background:repeating-linear-gradient(45deg,rgba(255,255,255,.02) 0 6px,transparent 6px 12px)}
aside .fnd.unknown .vd{color:#9AA6B5}
aside .fnd .sy{margin:5px 0 0;font-size:12.5px;line-height:1.5}
aside .fnd .ct{margin-top:7px;font-size:10.5px;color:var(--faint)}
aside .sw{display:flex;gap:6px;margin:10px 0 12px}
aside .sw button{flex:1 1 0}
button{font:inherit;font-size:11.5px;font-weight:500;padding:5px 11px;color:var(--ink);
  background:rgba(255,255,255,.04);border:1px solid var(--lane-lit);border-radius:6px;cursor:pointer}
button:hover{background:rgba(255,255,255,.08)}
button[aria-selected="true"]{background:rgba(111,195,255,.13);border-color:rgba(111,195,255,.45);color:#BFE4FF}
aside .key{margin-top:16px;border-top:1px solid var(--lane);padding-top:13px}
aside .key h4{margin:0 0 7px;font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--faint);
  font-weight:600}
aside .key .row{display:flex;align-items:center;gap:9px;padding:3px 0;font-size:11.5px}
aside .key svg{width:44px;height:10px;flex:0 0 44px}
aside .note{margin-top:14px;font-size:11.5px;line-height:1.55;color:var(--dim)}
aside .bytes{font-family:var(--mono);font-size:10.5px;line-height:1.6;background:#05070A;
  border:1px solid var(--lane);border-radius:6px;padding:10px 11px;margin-top:9px;max-height:220px;
  overflow:auto;white-space:pre-wrap;word-break:break-word;color:#B8C4D2}
@media (prefers-reduced-motion: no-preference){
  .rope,.glow{transition:opacity .18s linear}
  .pulse{animation:pulse 2.6s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:.16}50%{opacity:.34}}
}
`;

const JS = String.raw`
(() => {
  const S = window.__THREADS__;
  // Hue is identity. Five, because you can follow five ropes and not nine.
  const HUES = ['#4CC2FF','#FFB020','#5BE39B','#FF6B8A','#B388FF'];
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a) => { const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]); return e; };

  let scene = S[0], sel = null, mode = 'read', now = null;
  const svg = document.getElementById('stage');
  const panel = document.getElementById('panel');
  const pick = document.getElementById('scene');

  const M = { l: 168, r: 44, t: 76, b: 58 };
  let W = 0, H = 0, laneY = {}, xOf = (t) => t;

  /**
   * A small vertical offset per thread.
   *
   * The first version drew every rope on the lane's centre line, and the two
   * membrane chains — the same six agents, the same six steps — landed exactly
   * on top of each other. That is a *true* picture and a useless one: the reader
   * sees one rope and the whole point of the scope is that there are two, built
   * the same way, and one of them is wrong.
   *
   * Offsetting them is not decoration. It is the difference between "these are
   * identical" being visible and being asserted.
   */
  const OFF = 7;
  const yOf = (lane, t) => {
    const n = scene.world.threads.length;
    if (n < 2) return laneY[lane];
    return laneY[lane] + (t.ordinal - (n - 1) / 2) * OFF;
  };

  const measure = () => {
    const r = svg.getBoundingClientRect();
    W = r.width; H = r.height;
    const lanes = scene.world.lanes;
    const usable = H - M.t - M.b;
    const gap = lanes.length > 1 ? usable / (lanes.length - 1) : 0;
    laneY = {};
    lanes.forEach((l, i) => { laneY[l.id] = M.t + i * gap; });
    const span = Math.max(1, scene.world.span);
    xOf = (t) => M.l + (t / span) * (W - M.l - M.r);
  };

  /**
   * One segment of rope, as a path.
   *
   * A rest is a straight run in its lane. A crossing is a symmetric cubic with
   * horizontal tangents at both ends, so the rope leaves and arrives flat --
   * which is what makes a chain of them read as one continuous line rather than
   * a sequence of arcs.
   */
  const restPath = (r, t) =>
    'M' + xOf(r.x0) + ' ' + yOf(r.lane, t) + ' H' + xOf(r.x1);
  const crossPath = (c, t) => {
    const x0 = xOf(c.x0), x1 = xOf(c.x1);
    const y0 = yOf(c.from, t), y1 = yOf(c.to, t);
    const k = (x1 - x0) * 0.55;
    return 'M' + x0 + ' ' + y0 + ' C' + (x0 + k) + ' ' + y0 + ' ' + (x1 - k) + ' ' + y1 +
           ' ' + x1 + ' ' + y1;
  };

  const hueOf = (t) => HUES[t.ordinal % HUES.length];

  function draw() {
    measure();
    svg.replaceChildren();
    const w = scene.world;

    // Lanes.
    for (const l of w.lanes) {
      svg.appendChild(el('line', { class: 'lanebg', x1: M.l - 14, y1: laneY[l.id],
        x2: W - M.r, y2: laneY[l.id] }));
      const t = el('text', { class: 'lanelabel' + (l.kind === 'human' ? ' human' : ''),
        x: M.l - 22, y: laneY[l.id] + 4, 'text-anchor': 'end' });
      t.textContent = l.label;
      svg.appendChild(t);
    }

    // Where one agent holds two thoughts at once. Drawn under the ropes, as a
    // band across the lane -- a collision, which is what it is, rather than the
    // multiplier badge the desk used.
    /**
     * Two thoughts in one lane at one moment, as a bracket.
     *
     * This was a filled block behind the ropes, and at the width of a whole step
     * it read as the rope's background rather than as a collision — the eye took
     * it for a highlight and skipped it. A bracket is a mark, and a mark is read.
     */
    for (const c of scene.contentions) {
      const x0 = xOf(c.x0), x1 = xOf(c.x1), y = laneY[c.lane], h = 15;
      svg.appendChild(el('path', { class: 'contend',
        d: 'M' + x0 + ' ' + (y - h) + ' H' + x1 + ' M' + x0 + ' ' + (y + h) + ' H' + x1 +
           ' M' + x0 + ' ' + (y - h) + ' V' + (y - h + 4) + ' M' + x1 + ' ' + (y - h) + ' V' + (y - h + 4) +
           ' M' + x0 + ' ' + (y + h) + ' V' + (y + h - 4) + ' M' + x1 + ' ' + (y + h) + ' V' + (y + h - 4) }));
    }

    for (const t of w.threads) {
      const hue = hueOf(t);
      const g = el('g', { 'data-flow': t.flowId });
      if (sel && sel.flowId && sel.flowId !== t.flowId) g.setAttribute('class', 'dimmed');
      if (sel && sel.flowId === t.flowId) g.setAttribute('class', 'sel');

      const put = (d, state, meta) => {
        if (state === 'unknown') {
          // Not drawn. A gap you see the background through, with two ticks so a
          // reader can tell "nothing was recorded" from "nothing is here".
          const p = el('path', { class: 'gap', d: d });
          svg.appendChild(p);
          return;
        }
        g.appendChild(el('path', { class: 'glow ' + state, d: d, stroke: hue }));
        const rope = el('path', { class: 'rope ' + state, d: d, stroke: hue });
        g.appendChild(rope);
        if (meta) {
          const hit = el('path', { class: 'beadhit', d: d, stroke: 'transparent',
            'stroke-width': 16, fill: 'none' });
          hit.addEventListener('pointerdown', (ev) => { ev.stopPropagation(); select(meta); });
          g.appendChild(hit);
        }
      };

      for (const c of t.crosses) put(crossPath(c, t), c.state, { kind: 'cross', ...c, hue: hue, title: t.title });
      for (const r of t.rests) {
        put(restPath(r, t), 'carried', { kind: 'rest', ...r, hue: hue, title: t.title });
        // A bead where the thought rests: the step, as something to stand on.
        const cx = xOf((r.x0 + r.x1) / 2);
        const b = el('circle', { class: 'bead', cx: cx, cy: yOf(r.lane, t), r: 4.5, fill: hue });
        if (r.state === 'failed' || r.state === 'blocked') b.setAttribute('fill', '#FF6B8A');
        g.appendChild(b);
      }

      // The stop. A rope that ran into a threshold ends against a bar rather
      // than trailing off, because it did not trail off -- something refused it.
      if (!t.delivered) {
        const last = t.crosses[t.crosses.length - 1];
        const y = yOf(last.to, t), x = xOf(last.x1);
        g.appendChild(el('line', { class: 'stop', x1: x + 3, y1: y - 9, x2: x + 3, y2: y + 9,
          stroke: last.state === 'open' ? '#8A96A6' : '#FF6B8A' }));
      }
      svg.appendChild(g);
    }

    if (now !== null) drawPlayhead();
  }

  /**
   * Standing on the rope.
   *
   * The playhead is the one gesture this whole surface is for: drag along X and
   * you are standing at a moment, and the panel says what every lane was holding
   * then. It is the question "what happened here" asked positionally.
   */
  function drawPlayhead() {
    const x = xOf(now);
    svg.appendChild(el('line', { class: 'playhead', x1: x, y1: M.t - 22, x2: x, y2: H - M.b + 10 }));
    const holding = [];
    for (const t of scene.world.threads)
      for (const r of t.rests)
        if (now >= r.x0 && now <= r.x1) {
          holding.push({ lane: r.lane, t: t, r: r });
          svg.appendChild(el('circle', { class: 'pip', cx: x, cy: yOf(r.lane, t), r: 6.5,
            fill: hueOf(t), opacity: .9 }));
        }
    const lab = el('text', { class: 'now', x: x, y: M.t - 30, 'text-anchor': 'middle' });
    lab.textContent = holding.length
      ? holding.length + (holding.length === 1 ? ' agent is holding a thought' : ' agents are holding a thought')
      : 'nobody is holding anything here';
    svg.appendChild(lab);
    if (!sel || sel.kind !== 'cross') renderNow(holding);
  }

  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const fld = (k, v, at) => '<div class="fld"><span class="k">' + esc(k) + '</span><span>' +
    esc(v) + (at ? '<button class="at" data-open="' + esc(at) + '">' + esc(at) + '</button>' : '') +
    '</span></div>';

  const finding = (f) => {
    // The same rule as the desk: a verdict with no address is not renderable.
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

  function select(meta) {
    sel = meta; mode = 'read'; draw(); renderPanel();
  }

  function renderNow(holding) {
    if (sel) return;
    panel.innerHTML = '<h2>Standing here</h2>' +
      '<h3>' + (holding.length ? holding.length + ' in flight' : 'nothing in flight') + '</h3>' +
      '<div class="sub">drag anywhere to move along the threads</div>' +
      holding.map((h) => fld(h.lane, 'step ' + h.r.index + ' · ' + h.r.state)).join('') +
      keyHtml() + noteHtml();
    wire();
  }

  function renderPanel() {
    if (!sel) { now = now === null ? 0.001 : now; draw(); return; }
    const isCross = sel.kind === 'cross';
    const head = '<h2>' + (isCross ? 'A handoff' : 'A step') + '</h2>' +
      '<h3>' + esc(isCross ? sel.from.replace('@human','you') + ' → ' + sel.to.replace('@human','you')
                           : sel.lane + ' · step ' + sel.index) + '</h3>' +
      '<div class="sub">' + esc(sel.title) + '</div>' +
      '<div class="sw"><button id="m-read"' + (mode==='read'?' aria-selected="true"':'') + '>Read it</button>' +
      '<button id="m-agent"' + (mode==='agent'?' aria-selected="true"':'') + '>Ask an agent</button></div>';

    let body;
    if (mode === 'read') {
      body = isCross
        ? fld('state', sel.state) + fld('because', sel.because) +
          (sel.digest ? fld('observation', sel.digest, sel.source || undefined)
                      : fld('carried', 'nothing recorded — this is not a claim that nothing moved'))
        : fld('state', sel.state) +
          (sel.digest ? fld('observation', sel.digest, sel.source || undefined) : '') +
          (sel.result ? fld('said', sel.result) : '');
    } else {
      /**
       * What a claim about a segment was read out of.
       *
       * The packet's source when there is one, and otherwise the flow store's
       * own record of the step — which is a real address the page can resolve,
       * and is where a claim like "step 3 is blocked" actually comes from.
       *
       * Written this way because the first version threw in the browser: a
       * crossing with a verdict and no source hit the guard, which is the guard
       * doing its job on a real gap rather than a mistake in itself. The desk
       * hit the identical case and took the identical way out. Relaxing the rule
       * was never an option; the fix is to cite what was read.
       */
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
              says: 'The thought reached ' + sel.to.replace('@human','you') + ' and it has not reached a verdict. ' + sel.because + '. There is no result here to agree or disagree with — only an open question.' };
      else if (isCross && sel.state === 'ignored')
        f = { verdict: 'problem', cites: cites,
              says: 'It arrived and ' + sel.to + ' used none of it. ' + sel.because + '. Every step on this thread reported cleanly; this is the instrument that disagrees with them.' };
      else if (isCross && sel.state === 'blocked')
        f = { verdict: 'problem', cites: cites,
              says: 'It arrived, ' + sel.to + ' ran, and it did not pass. ' + sel.because + '.' };
      else if (isCross)
        f = { verdict: 'ok', cites: cites, says: sel.from.replace('@human','you') + ' closed with an observation and ' + sel.to + ' read it. ' + sel.because + '.' };
      else if (!sel.digest)
        f = { verdict: 'unknown', cites: [], says: 'This step recorded no observation, so there is nothing here to read.' };
      else
        f = { verdict: sel.state === 'failed' || sel.state === 'blocked' ? 'problem' : 'ok', cites: cites,
              says: sel.lane + ' at step ' + sel.index + ' is ' + sel.state + ', and it recorded ' + sel.digest + '.' };
      body = finding(f);
    }
    panel.innerHTML = head + body + keyHtml() + noteHtml();
    wire();
  }

  const swatch = (cls, extra) =>
    '<svg viewBox="0 0 44 10"><path class="rope ' + cls + '" d="M2 5 H42" stroke="' +
    (extra || '#4CC2FF') + '"/></svg>';

  const keyHtml = () => '<div class="key">' +
    '<h4>Hue — which thread</h4>' +
    scene.world.threads.map((t) =>
      '<div class="row">' + swatch('', HUES[t.ordinal % HUES.length]) +
      '<span>' + esc(t.title) + '</span></div>').join('') +
    '<h4 style="margin-top:12px">Texture — what happened</h4>' +
    '<div class="row">' + swatch('') + '<span><b>carried</b> — it moved and the next step used it</span></div>' +
    '<div class="row">' + swatch('ignored') + '<span><b>carried nothing forward</b> — it arrived and nothing used it</span></div>' +
    '<div class="row">' + swatch('open') + '<span><b>no verdict yet</b> — it arrived and nothing has been decided</span></div>' +
    '<div class="row"><svg viewBox="0 0 44 10"><path class="gap" d="M2 5 H42"/></svg>' +
      '<span><b>unrecorded</b> — nothing was recorded, so no rope is drawn</span></div>' +
    '</div>';

  const noteHtml = () => '<div class="note">' + esc(scene.note) + '</div>';

  const NOTES = {};
  const wire = () => {
    const r = panel.querySelector('#m-read'), a = panel.querySelector('#m-agent');
    if (r) r.onclick = () => { mode = 'read'; renderPanel(); };
    if (a) a.onclick = () => { mode = 'agent'; renderPanel(); };
    for (const b of panel.querySelectorAll('button.at'))
      b.onclick = () => {
        const at = b.dataset.open;
        const box = panel.querySelector('.bytes');
        const html = '<div class="bytes">' + esc(at) + '\n\n' + esc(NOTES[at] ||
          'This address is recorded on the segment above. This page has no filesystem, so the ' +
          'address is shown rather than the bytes — and no contents are invented for it.') + '</div>';
        if (box) box.outerHTML = html; else panel.insertAdjacentHTML('beforeend', html);
        wire();
      };
  };

  // Dragging the stage moves you along the threads.
  let dragging = false;
  const at = (ev) => {
    const r = svg.getBoundingClientRect();
    const span = Math.max(1, scene.world.span);
    const t = ((ev.clientX - r.left) - M.l) / (W - M.l - M.r) * span;
    return Math.max(0, Math.min(span, t));
  };
  svg.addEventListener('pointerdown', (ev) => {
    dragging = true; sel = null; now = at(ev); svg.setPointerCapture(ev.pointerId); draw();
  });
  svg.addEventListener('pointermove', (ev) => { if (dragging) { now = at(ev); draw(); } });
  window.addEventListener('pointerup', () => { dragging = false; });

  pick.addEventListener('change', () => {
    scene = S.find((x) => x.id === pick.value) || S[0];
    sel = null; now = null; draw(); renderPanel();
  });
  window.addEventListener('resize', draw);

  draw();
  now = scene.world.span * 0.42;
  draw();
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
      <span class="sk">Sketch — not the product, not measured</span>
      <select id="scene">${payload
        .map((s) => `<option value="${s.id}">${s.label}</option>`)
        .join("")}</select>
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
