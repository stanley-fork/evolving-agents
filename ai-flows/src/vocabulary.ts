/**
 * The visual vocabulary, in one place because two surfaces draw with it.
 *
 * The read-only explorer ([view.ts](view.ts)) and the canvas
 * (`ai-ui`) show the same system, and a person who learns what a purple cube
 * means on one must not find it means something else on the other. Two copies of
 * this table would diverge on the first change and nothing would fail.
 *
 * It lives in `ai-flows` rather than in `ai-ui` because the explorer is the
 * older surface and the canvas is the one being argued for — the dependency
 * points from the thing being justified to the thing already justified, never
 * the other way.
 *
 * ## Why colour at all
 *
 * The first explorer was a flat dark document and the one complaint it drew was
 * that it was unclear: everything the same weight, nothing telling you what kind
 * of thing you were looking at. System 7 and Windows 3.1 solved that before they
 * solved anything else — **kind and state visible before you read a word** — and
 * that is the whole reason the vocabulary is shaped this way.
 *
 * The rule that keeps it honest: **every colour drawn must appear in a legend
 * generated from these tables**, never from a hand-kept list. A legend that
 * drifts from the page it explains is worse than no colour at all.
 */
import type { ScopeRole } from "./conformation.ts";

/**
 * Scope roles, in their own hue family.
 *
 * `individual` was the same green as `done` and `unknown` the same red as
 * `failed` — the same collision `AGENT_COLOR` had with `running`, one table
 * over. A scope is a *place*, not a verdict, and drawing it in a verdict's
 * colour invites exactly the reading it should not get.
 */
export const SCOPE_COLORS: Record<ScopeRole, string> = {
  system: "#6B4FA8",
  project: "#2D6FB8",
  collective: "#1F8A70",
  team: "#A8681C",
  individual: "#4A7C9B",
  unknown: "#8C9096",
};

/**
 * Step states and flow states share this table on purpose: `done` means the same
 * thing at both levels, and giving them separate palettes would invent a
 * distinction the system does not have.
 *
 * Several names share a colour (`pending`, `waiting` and `draft` are all "not
 * started"), which is why the legend groups by colour rather than by name — it
 * must not claim a distinction the page cannot draw.
 */
export const STATE_COLORS: Record<string, string> = {
  done: "#2E7D4F",
  running: "#D4900F",
  pending: "#C9C7C1",
  failed: "#B23A2E",
  blocked: "#B23A2E",
  waiting: "#C9C7C1",
  abandoned: "#8C9096",
  draft: "#C9C7C1",
};

/**
 * Agents are not coloured by state, so they are not coloured in the state's hue.
 *
 * `AGENT_COLOR` used to be `#e0a020` and `STATE_COLORS.running` used to be
 * `#e0a020`. The same amber meant *this is an agent* and *this step is in
 * flight*, on the same surface, at the same time, next to each other — an agent
 * sitting idle on the desk was drawn in the colour of work happening.
 *
 * That is not a matter of taste. The rule this file states at the top is that
 * every colour drawn appears in a legend generated from these tables, so that
 * the page cannot claim a distinction it does not draw. Two entries sharing a
 * hue across two different tables defeats it silently: the legend lists amber
 * twice, under two headings, and looks correct.
 *
 * So kind is now a warm stone and its variations, and **every saturated colour
 * on the surface is a state or a wire**. A reader who learns the state palette
 * has learned all the colour there is.
 */
export const AGENT_COLOR = "#C8B79A";
export const SUBAGENT_COLOR = "#B3A184";
/** A name declared in an agent's `subagents:` with no file behind it. */
export const MISSING_COLOR = "#B23A2E";
export const PERSON_COLOR = "#7C8590";

/** Group the state palette by colour, so a legend cannot claim a distinction the page cannot draw. */
export function statesByColor(): Array<[string, string[]]> {
  const byColor = new Map<string, string[]>();
  for (const [name, color] of Object.entries(STATE_COLORS)) {
    byColor.set(color, [...(byColor.get(color) ?? []), name]);
  }
  return [...byColor];
}

/**
 * The chrome primitives: the desk surface, a window, a bevel, a cube, a document.
 *
 * Shared as a string rather than a stylesheet file because both surfaces must
 * stay **self-contained** — no external requests, so they open from a file days
 * later with no server running. That rules out a `<link>`, which is exactly the
 * kind of convenience that breaks a page when nobody is watching.
 */
export const CHROME_CSS = `
:root{
  /* The ground recedes. It used to be a mid grey with a 50% dither on it, which
     is a picture of a 1991 desktop rather than a surface for work: the loudest
     thing on the page was the part with no information in it. */
  --desk:#DEDDD8;
  --face:#FFFFFF; --paper:#FFFFFF;
  --ink:#14171A; --dim:#646B73; --faint:#949BA3;
  --line:#E4E3DF; --line-2:#CFCEC9;
  --accent:#2D6FB8; --accent-soft:#EAF1F9;
  /* Elevation, in two steps that mean two things: resting, and lifted. Every
     panel used to carry the same 3px hard shadow, so nothing was above anything.
     A shadow that is everywhere is a texture, not a depth. */
  --sh-1:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.07);
  --sh-2:0 4px 10px rgba(16,24,40,.07),0 10px 28px rgba(16,24,40,.10);
  --r:7px; --r-sm:5px;
  --lite:rgba(255,255,255,.75); --dark:rgba(0,0,0,.10);
  /* The platform's own type. "Geneva, Verdana" was a costume, and it rendered as
     whatever each machine happened to have. */
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Monaco,"Cascadia Mono","Roboto Mono","DejaVu Sans Mono",monospace;
}
*{box-sizing:border-box}
code{font-family:var(--mono);font-size:12px}
.dim,.muted{color:var(--dim)}
.warn{color:#8A5A0A}
.err{color:#A6342A}
.ok{color:#2E7D4F}

html,body{min-height:100%}
body{margin:0;color:var(--ink);font:14px/1.55 var(--sans);
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}

/* The ground. Flat, quiet, and one shade darker than the things standing on it,
   which is the whole job: separation without pattern. */
.deskbg{background:var(--desk)}

/* A window: the unit of "one thing ends here". Now a hairline and a resting
   shadow rather than a black border and a hard offset. */
.win{background:var(--face);border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--sh-1);overflow:hidden}
.win .bar{display:flex;align-items:center;gap:8px;padding:9px 13px;
  border-bottom:1px solid var(--line);background:var(--face)}
.win .bar h1,.win .bar h2{font-size:12px;margin:0;font-weight:650;letter-spacing:.01em;
  color:var(--ink);white-space:nowrap}
.win .bar .count{font-weight:400;color:var(--dim)}
/* The two squares on every title bar closed nothing and zoomed nothing. They are
   gone rather than restyled: a control that does not control is the clearest
   case there is of decoration wearing a control's clothes. */
.win .box{display:none}
.win-body{padding:13px 15px 15px;background:var(--paper)}
/* Recessed: a container that holds things, rather than a thing. */
.win.tray .win-body{background:#F4F3F0;box-shadow:none;
  border-top:1px solid var(--line);padding:12px}

/* The cube: the unit of kind and state. Flat fill, hairline, small radius --
   the inset bevel was drawing a plastic button at 13 pixels, where the highlight
   and the shadow together ate a third of the swatch and shifted the hue. */
.cube{display:inline-block;width:12px;height:12px;flex:0 0 12px;background:var(--c);
  border:1px solid rgba(0,0,0,.10);border-radius:3px;
  vertical-align:-2px;margin-right:6px}
.cube.sm{width:9px;height:9px;flex-basis:9px;border-radius:2px;vertical-align:0;margin-right:5px}
.cube.lg{width:16px;height:16px;flex-basis:16px;border-radius:4px;vertical-align:-3px}
.strip{display:inline-flex;align-items:center;gap:3px}
.strip .cube{margin-right:0}

/* A page with a folded corner, in two boxes and a triangle. */
.doc-icon{position:relative;width:15px;height:19px;flex:0 0 15px;background:#fff;
  border:1px solid var(--line-2);border-radius:2px}
.doc-icon::before{content:"";position:absolute;right:-1px;top:-1px;border-width:0 6px 6px 0;
  border-style:solid;border-color:transparent var(--line-2) transparent transparent}
.doc-icon::after{content:"";position:absolute;left:3px;top:8px;width:8px;height:1px;color:var(--line-2);
  box-shadow:0 0 0 0 currentColor,0 3px 0 currentColor,0 6px 0 currentColor;background:currentColor}

.menubar{background:var(--face);border-bottom:1px solid var(--line);
  padding:9px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:12px}
.menubar .apple{font-weight:650;letter-spacing:.01em}
.menubar .sep{color:var(--line-2)}
.menubar .right{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--faint)}

.key{display:flex;gap:22px;flex-wrap:wrap;font-size:12px}
.key ul{list-style:none;margin:5px 0 0;padding:0}
.key li{display:flex;align-items:center;padding:2px 0}
.key strong{font-size:11px;font-weight:650;letter-spacing:.05em;text-transform:uppercase;color:var(--dim)}
`;
