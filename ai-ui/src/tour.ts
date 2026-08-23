/**
 * Play — the demo, driving itself.
 *
 * A person who opens the desk sees an arrangement and no idea what any of it is
 * for. The gestures that make it worth having — following a packet along a wire,
 * opening what it carried, putting an agent on a flow and then checking the
 * artefact it cites — are all invisible until somebody performs one, and a
 * visitor on a website will not.
 *
 * ## What it is about
 *
 * Two real projects, and the beat it exists for is the seventh: the system agent
 * makes a claim, and one click puts you in front of the thing it read. Everything
 * before it is setup for that, and everything after it is the same move on the
 * project where no closed form exists.
 *
 * It used to narrate an invented web project for seven of thirteen beats and
 * never reach the cochlea gate at all — the best thing in the repository, three
 * clicks away and never shown.
 *
 * ## It drives the real client. It does not play a movie.
 *
 * This is the same rule [simulate.ts](simulate.ts) lives under, one level up, and
 * it is the only thing that makes a tour honest: every beat below **dispatches
 * the events a person would** — `pointerdown`, `pointermove`, `pointerup` on the
 * actual cube, `click` on the actual button — and then lets the desk react
 * however it reacts. Nothing here draws a frame, animates a fake, or asserts an
 * outcome.
 *
 * The property that buys: **if the desk breaks, the tour breaks.** A scripted
 * animation of a product is a second implementation of it, and it goes on looking
 * correct for as long as nobody checks. This cannot: drop the cube on a document
 * that no longer accepts one and the step does not appear, visibly.
 *
 * ## It never fights the person watching
 *
 * Any real pointer or key event stops the tour where it is and leaves the desk
 * exactly as the tour left it. A demo that keeps moving things while somebody is
 * trying to click is worse than no demo, and "it will finish in a second" is not
 * a defence — they came to touch it.
 *
 * ## Demo only
 *
 * Injected next to the simulation and only when `simulate` is set, so the product
 * cannot ship a thing that moves the user's documents around. A test asserts the
 * unsimulated page does not contain it.
 */
export const TOUR_JS = String.raw`
(() => {
  const $ = (s) => document.querySelector(s);
  const docs = () => [...document.querySelectorAll('.docnode')];
  const docByTitle = (t) => docs().find((d) => d.innerText.includes(t));
  // By id, not by text: a creature's label now carries what it is doing next to
  // its name, so matching innerText found nothing the moment an agent was busy.
  const cubeNamed = (n) => [...document.querySelectorAll('.acube')].find((c) => c.dataset.id === n);

  const bar = document.createElement('div');
  bar.className = 'tourbar';
  bar.innerHTML =
    '<button id="tourgo">▶ Play</button>' +
    '<span id="tourcap">Watch the desk use itself — every step below is a real gesture, not a recording.</span>';
  document.body.appendChild(bar);

  // A visible pointer. The drag is the one gesture that is incomprehensible
  // without seeing where the hand is.
  const hand = document.createElement('div');
  hand.className = 'tourhand';
  document.body.appendChild(hand);

  let running = false, stop = false;
  // Published because the mascot ([mascot.ts](mascot.ts)) reacts to the same
  // events the tour dispatches, and two voices narrating one gesture is worse
  // than either alone. It stays quiet while this is true.
  window.__TOUR__ = { running: false };
  const cap = (t) => { $('#tourcap').textContent = t; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const halt = () => {
    if (!running) return;
    stop = true;
    hand.style.opacity = '0';
    cap('Stopped — it is yours. Press Play to watch the rest.');
  };
  // A real gesture wins immediately. isTrusted is false for everything the tour
  // dispatches, so this cannot stop itself.
  for (const ev of ['pointerdown', 'keydown', 'wheel'])
    window.addEventListener(ev, (e) => { if (e.isTrusted) halt(); }, true);

  const moveHand = async (x, y, ms) => {
    const r = hand.getBoundingClientRect();
    const x0 = r.left || x, y0 = r.top || y;
    const steps = Math.max(1, Math.round(ms / 16));
    for (let i = 1; i <= steps && !stop; i++) {
      const t = i / steps, e = t < .5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2; // ease
      const cx = x0 + (x - x0) * e, cy = y0 + (y - y0) * e;
      hand.style.left = cx + 'px'; hand.style.top = cy + 'px';
      await sleep(16);
    }
  };

  const centre = (el) => {
    const r = el.getBoundingClientRect();
    return [r.left + r.width / 2, r.top + r.height / 2];
  };

  const pointAt = async (el, ms) => {
    hand.style.opacity = '1';
    const [x, y] = centre(el);
    await moveHand(x, y, ms || 700);
  };

  /** A real drag: the same three events a hand produces, on the real element. */
  const dragOnto = async (el, target) => {
    const [sx, sy] = centre(el);
    await pointAt(el, 600);
    el.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: sx, clientY: sy, pointerId: 1 }));
    const [tx, ty] = centre(target);
    const steps = 26;
    for (let i = 1; i <= steps && !stop; i++) {
      const t = i / steps;
      const cx = sx + (tx - sx) * t, cy = sy + (ty - sy) * t;
      hand.style.left = cx + 'px'; hand.style.top = cy + 'px';
      window.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: cx, clientY: cy, pointerId: 1 }));
      await sleep(22);
    }
    window.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, clientX: tx, clientY: ty, pointerId: 1 }));
  };

  const select = async (el) => {
    await pointAt(el, 700);
    el.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerId: 1 }));
  };

  const press = async (el) => {
    if (!el) return;
    await pointAt(el, 500);
    el.click();
  };

  const COCLEA_BEATS = [
    async () => {
      cap('Everything here is an agent. Each box is a thing with a name, a file, and a list of tools it is allowed to use — not a label on a diagram.');
      const c = cubeNamed('DERIVADOR');
      if (c) await select(c);
      await sleep(3000);
    },
    async () => {
      cap('A flow is a path through them, and the lines are the handoffs. The dot travelling one is what actually moved — click it and you get the address it was recorded at.');
      const w = document.querySelector('.wires path.hit');
      if (w) {
        const r = w.getBoundingClientRect();
        await moveHand(r.left + r.width / 2, r.top + r.height / 2, 800);
        w.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerId: 1 }));
      }
      await sleep(3600);
    },
    async () => {
      cap('Two chains, the same six agents, the same six steps. Both were built the same way. One is frozen and one is held.');
      const d = docByTitle('flux assembly');
      if (d) await select(d);
      await sleep(3000);
    },
    async () => {
      cap('Nothing in either trace separates them. Every step ran, settled and reported — and one of these chains is wrong in every number it produced.');
      const d = docByTitle('apex boundary');
      if (d) await select(d);
      await sleep(3400);
    },
    async () => {
      cap('What separates them is a gate: an oracle declared before the run. GATE-A01 measured 2.592e-4 against a tolerance of 1.0e-4, so this result cannot freeze.');
      await press(document.getElementById('tab-trace'));
      await sleep(4600);
    },
    async () => {
      /**
       * The gesture the whole redesign is for, and a real drag.
       *
       * Dropping one agent onto a thing declares a relationship — doc/15 phase
       * 5, specified in March and unbuilt until now. INSPECTOR has one tool,
       * read, so the relationship it declares is "I am reading this", and the
       * desk writes down the finding and the address rather than adding a step.
       */
      cap('Or do not read it yourself — put an agent on it. Drag INSPECTOR onto the flow. It has one tool: read.');
      const c = cubeNamed('INSPECTOR');
      const d = docByTitle('apex boundary');
      if (c && d) await dragOnto(c, d);
      await sleep(4200);
    },
    async () => {
      cap('And this is the part that matters: under the sentence is the artefact it read. One click, and you are looking at what it looked at.');
      const a = document.querySelector('.fnd button.at');
      if (a) await press(a);
      await sleep(4400);
    },
    async () => {
      cap('When there is nothing to read, it is required to say unknown — not to guess. A hop nobody recorded is drawn thin, grey and dashed, and it carries no packet at all.');
      const u = document.querySelector('.wires g:has(path.w.unknown) path.hit')
        || document.querySelector('.wires path.w.unknown');
      if (u) {
        const r = u.getBoundingClientRect();
        await moveHand(r.left + r.width / 2, r.top + r.height / 2, 700);
        u.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerId: 1 }));
      }
      await sleep(4200);
    },
    async () => {
      cap('Ask reads the trace, never the goal. The goal is what somebody meant to happen — it reads like an answer even when the work was never done.');
      await press(document.getElementById('m-read'));
      const q = document.getElementById('q');
      if (q) {
        q.value = '';
        for (const ch of 'what did this actually produce?') {
          if (stop) return;
          q.value += ch;
          q.dispatchEvent(new Event('input', { bubbles: true }));
          await sleep(34);
        }
        await press(document.getElementById('qgo'));
      }
      await sleep(2800);
    },
    async () => {
      cap('And a flow with a step still to run states the cost before the button is pressed. Advancing spends a model call, and the panel says so first.');
      const d = docByTitle('GATE-D1');
      if (d) await select(d);
      await sleep(1400);
      await press(document.getElementById('adv'));
      await sleep(3400);
    },
    async () => {
      cap('The other project. Here there is no closed form to check against — so the judge itself goes on trial. Switching scope reloads, and the tour continues there.');
      const sel = document.getElementById('scope');
      const wanted = [...sel.options].find((o) => o.value.indexOf('hemo') >= 0);
      if (!wanted) return;
      await pointAt(sel, 700);
      await sleep(1500);
      sel.value = wanted.value;
      resumeAt(HEMO_BEAT);
      sel.dispatchEvent(new Event('change', { bubbles: true }));
      await sleep(6000);
    },
  ];

  /**
   * The beats that run after the scope change.
   *
   * A separate array rather than an offset into one. The offset version shipped
   * as beats.length minus four when the answer was eleven, so the tour crossed the
   * navigation and resumed one beat *late* — silently skipping the beat that
   * introduces the project. An index computed by counting entries by hand is a
   * number that goes wrong every time somebody adds a beat, and nothing checks
   * it. This cannot be wrong: the resume point is where the second list starts.
   */
  const HEMO_BEATS = [
    async () => {
      cap('98 predictions built from exact solutions and corrupted by amounts chosen in advance, so the true error is known rather than estimated.');
      const d = docByTitle('physics checks');
      if (d) await select(d);
      await sleep(3600);
    },
    async () => {
      cap('The panel scores 0.9056 against a kill threshold of 0.80 written down before the run. Alone, six of its seven oracles are near a coin flip — A5 is 0.5209.');
      await press(document.getElementById('tab-trace'));
      await sleep(5000);
    },
    async () => {
      cap('That number is the one almost nobody publishes. You only get it by measuring your own judge, and the artefact records the hash of all seven oracles and the exact stack that produced them.');
      hand.style.opacity = '0';
      await sleep(4200);
    },
    async () => {
      cap('And the open question, left open: A4 alone read 0.706 here and 0.652 elsewhere. Different machines, so the honest verdict is not "disagrees" — it is unknown, and the desk draws it that way.');
      const d = docByTitle('second machine');
      if (d) await select(d);
      await sleep(1000);
      await press(document.getElementById('m-agent'));
      await sleep(5000);
    },
    async () => {
      cap('That is the desk. Every box is an agent, every line is a handoff, and every claim has an address. Drag anything — it is yours now.');
      hand.style.opacity = '0';
      await sleep(1800);
    },
  ];

  const beats = [...COCLEA_BEATS, ...HEMO_BEATS];
  /** Where the tour picks up on the other side of the navigation. */
  const HEMO_BEAT = COCLEA_BEATS.length;

  /**
   * Carry the tour across a real navigation.
   *
   * Changing scope reloads the page, because that is what changing scope does
   * against a server -- and a tour that faked it to keep its own state would be
   * the recording this file exists not to be. So the position is left in
   * sessionStorage and picked up on the other side. sessionStorage rather than
   * localStorage: a tour half-finished yesterday must not start playing at somebody
   * tomorrow.
   */
  const RESUME = 'ai-os.tour.resumeAt';
  const resumeAt = (i) => { try { sessionStorage.setItem(RESUME, String(i)); } catch (e) {} };
  const takeResume = () => {
    try {
      const v = sessionStorage.getItem(RESUME);
      sessionStorage.removeItem(RESUME);
      return v === null ? -1 : Number(v);
    } catch (e) { return -1; }
  };

  const play = async (from) => {
    if (running) { halt(); return; }
    running = true; stop = false;
    window.__TOUR__.running = true;
    $('#tourgo').textContent = '■ Stop';
    for (let i = from; i < beats.length; i += 1) {
      if (stop) break;
      const beat = beats[i];
      try { await beat(); } catch (e) { cap('The tour hit something the desk did not expect: ' + e.message); break; }
    }
    running = false; stop = false;
    window.__TOUR__.running = false;
    hand.style.opacity = '0';
    $('#tourgo').textContent = '▶ Play again';
  };

  $('#tourgo').onclick = () => play(0);

  // Landed here mid-tour. The documents have to exist before a beat can point at
  // one, and the desk draws them on its first render.
  const resume = takeResume();
  if (resume >= 0 && resume < beats.length) setTimeout(() => play(resume), 1200);
})();
`;

/**
 * The tour's chrome.
 *
 * Kept beside the script rather than in `DESK_CSS` so that nothing about the
 * product's stylesheet has to know a tour exists.
 */
export const TOUR_CSS = `
/* Above the memory drawer, which is 150px tall: a control that covers the
   thing it is describing is not a control. */
.tourbar{position:fixed;left:14px;bottom:174px;z-index:400;display:flex;align-items:center;gap:10px;
  background:var(--face);border:1px solid #000;box-shadow:3px 3px 0 rgba(0,0,0,.4);padding:6px 10px;
  max-width:min(760px,calc(100vw - 40px))}
.tourbar button{font:inherit;font-size:11px;font-weight:700;padding:3px 12px;white-space:nowrap}
.tourbar span{font-size:11px;line-height:1.4;color:#26292d}
/* The hand. A pointer you can follow, because a drag is unreadable without one. */
.tourhand{position:fixed;width:16px;height:16px;z-index:401;opacity:0;pointer-events:none;
  transition:opacity .25s;transform:translate(-2px,-2px);
  background:#16181a;clip-path:polygon(0 0,0 14px,4px 10px,7px 16px,10px 14px,7px 9px,12px 9px)}
@media (prefers-reduced-motion: reduce){ .tourhand{transition:none} }
`;
