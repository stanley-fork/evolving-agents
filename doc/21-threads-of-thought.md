# Threads of thought

*A proposal, what it is better at, what it would cost, and the order to build it
in. Written alongside a working sketch rather than instead of one.*

---

## 0. The proposal

> A horizontal flow of threads of thought between the agents and the human, where
> you can stand on a rope or on an agent and inspect it and work there. That
> mimics the idea of a flow. It could have vivid colours, and maybe a
> implementation in three.js or similar.

Three separable claims: **a metaphor**, **a palette**, and **an engine**. They
have very different answers, and taking them as one package is the mistake.

---

## 1. The metaphor is better, and here is the property that makes it so

**The desk has no time axis.** Cards sit in a grid. The order a flow happened in
is only inferable by following the wires, and the arrangement itself is whatever
the layout engine or somebody's hand decided. A flow *is* a sequence, and the
surface drawing it threw away its one intrinsic dimension.

In the thread view, **X is time and Y is who is holding the thought**. One flow
is one continuous rope: it starts in your lane, dips into `DERIVADOR`'s, rises to
`CONSTRUCTOR`'s, and — if it finishes — comes back to you.

Four things the rope shows that a wire cannot:

| | the rope | the desk's wire |
|---|---|---|
| **Duration** | a long stretch of rope | nothing |
| **Simultaneity** | two ropes crossing one lane at one X | a `×2` badge on a cube |
| **Absence** | *no rope* — you see the dark through it | a dashed line, which is a line |
| **Delivery** | it comes home, or it does not | nothing |

That last row is the one that surprised me while building it. A finished flow
**delivers something to somebody**, and a blocked one does not. The desk could
not say that at all, and once the rope says it you cannot stop reading it: on
`coclea-sr`, two ropes run side by side, identical the whole way, and then one
stops against a bar while the other climbs home. That is the entire argument of
that project in one glance — and it took the desk a panel, a tab, and a paragraph.

---

## 2. The palette can be loud here, and could not be on the desk

The rule [doc 04](04-ai-ui.md) arrived at is **colour is state and evidence, and
nothing else on the surface is coloured**. That reads like "be quiet", and it is
not — it is *one channel, one meaning*. A dark ground buys a second channel:

- **Hue is identity.** Which thread. Five saturated colours, enough to follow one
  rope through eight lanes among four others.
- **Luminance and texture are state.** Carried is lit. *Carried nothing forward*
  goes dark from where it landed. *Did not pass* runs at full brightness into a
  bar — a negative result is a result, and dimming it would hide the most
  informative thing on the page. *No verdict yet* frays. *Unrecorded* is not
  drawn: a gap you can see the background through.

Two orthogonal channels, each meaning exactly one thing, and a legend that states
both. On a beige card grid a saturated palette is noise; on a dark field it is
the information. **So: vivid, yes — and the ground goes dark first, or it isn't.**

---

## 3. The engine is the right second move and the wrong first one

This is where I would push back, and not on taste.

**The question worth answering first is whether standing on a rope helps you say
what happened.** That question is answerable in a day with paths and transforms.
The sketch beside this document is that day. If the answer is yes, an engine buys
real things — depth of field, a camera you can fly, a thousand ropes instead of
five, particles on the wire — and it will have been paid for. If the answer is
no, an engine has made the failure **expensive, beautiful, and much harder to
abandon**, which is the worst combination available.

There is also a specific property that would be spent on the way, and this
repository has already argued about it. The demo is **one self-contained file
that opens from disk, with no server and no network**. That is why anybody can
check it, why the provenance script can read it, and why it survives being
emailed. A 600 kB library and a bundler is precisely the instalment
[doc 08](08-roadmap.md) says this pillar must not run up before it has earned one.

**The order:** SVG sketch → run the stopwatch on it → *then* decide whether the
thing it is missing is depth.

---

## 4. What building it found

Three things, all of which are arguments about the metaphor rather than bugs.

**The two ropes were exactly on top of each other.** Both membrane chains use the
same six agents in the same order, so the first render drew one rope. True, and
useless — the whole point is that there are two, built the same way, and one is
wrong. A small per-thread offset makes "these are identical" *visible* instead of
asserted.

**Lane order is a design decision, not a detail.** First-appearance ordering made
the picture depend on which flow happened to be listed first: reorder the input
and every rope changes shape. Lanes are ordered by where each agent *tends* to
act, so a rope tends downward and following one is following a slope rather than
a search. A test asserts the ordering does not depend on argument order — a
layout that reshuffles for a reason the reader cannot see is one nobody can
learn.

**The citation guard fired again, in the browser, correctly.** A crossing with a
verdict and no source hit `assertCited` and threw. Same case the desk hit, same
resolution: cite the flow store's own record of the step, which is a real address
the page resolves. Relaxing the rule was never available. The one honest special
case is the first crossing — *you asked for this* — which is the single handoff
in the system that is not in question, and it cites the flow.

---

## 5. What Ive would ask before letting any of it ship

Six questions, and the sketch answers four of them.

1. **What is the one gesture?** — Drag along the rope. You are standing at a
   moment, and the panel says what every lane was holding then. *Answered.*
2. **What does it look like stopped?** — It is stopped 99% of the time, and a
   surface that only works while animating is a screensaver. *Answered: nothing
   moves unless you move it.*
3. **Does it survive a screenshot?** — Evidence has to be quotable, and a still
   frame of this carries what the page carries. *Answered.*
4. **Does the vocabulary transfer, or is this a rewrite?** — All five wire states
   come across unchanged, including the split between *did not pass* and *no
   verdict yet*. It is a redesign. *Answered, and asserted by a test.*
5. **What happens at a hundred threads?** — Unanswered. Five ropes and eight
   lanes is a picture; two hundred and forty is a hairball, and the honest fix is
   probably aggregation rather than rendering, which is a different problem than
   the one an engine solves.
6. **Is it faster than the desk at the task the stopwatch measures?** —
   Unanswered, and it is the only question that decides anything. Neither surface
   has been measured against a person and a three-day-old flow.

---

## 6. The clock was already there, and the projection was dropping it

The strongest thing this sketch found is not about the sketch.

`ai-flows`' store has recorded `Attempt.startedAt` and `Attempt.finishedAt` since
the beginning. **`trace.ts` dropped both.** Every surface built on that projection
therefore had no clock at all — so "the desk has no time axis" was never a limit
of the data. It was a lossy hop between the store and the screen, and nothing
failed when it happened.

That is the expensive kind of gap: the desk laid documents out in a grid because
it had no other option *available to it*, while the answer sat one call upstream.
Carrying the two fields through is four lines. What it changes:

- **A step's width is how long it took.** Not a slot.
- **The space between two steps is time nobody was working**, which is often the
  most interesting width on the page.
- **The two membrane chains ran thirty hours apart.** On the desk they looked
  like a pair. They are not a pair; they are a run and a re-run a day later, and
  no surface had ever said so.
- **A test of mine was asserting something false.** It claimed the two chains
  contend for the same agents "at the same steps" — true on a sequence axis, and
  false about the world. The clock deleted it. That is the surface's own argument
  turned on its own test suite, which is the only place it counts.

Two rules had to be written down to keep it honest:

**The basis is all-or-nothing and it is stated.** If any settled attempt lacks a
start, the whole world falls back to sequence and the panel says so in as many
words. Mixing a clock-drawn thread with a sequence-drawn one on one axis puts two
incomparable things in a picture and invites a reader to compare their widths.

**A step that has not begun has no time, and is not given one.** The first
version fell back to the start of the world for a step with no attempts, so
GATE-D1 — whose last two steps are pending — was drawn spanning the entire
seventy-two hour window: a flow that started forty minutes ago, drawn as three
days of work. A step that has not begun belongs just after the last one that did.
That is a statement about *order*, which is known, and it is drawn dim and hollow
so it cannot be read as a statement about time.

---

## 7. Motion, and the only rule that makes it worth having

> **Everything that moves is a measurement. If nothing moves, nothing is
> happening.**

That rule is what separates this from a screensaver, and it is expensive to keep.
Exactly one thing on the surface animates on its own: a segment whose step is
`running` — an attempt that started and never closed, which is a fact in the
store. The animation loop **cancels itself** when it finds none, so a still
surface is a true statement rather than an idle one. Verified in a browser: on
the scope with one open step the dash offset advances; on the settled scopes
`requestAnimationFrame` is not scheduled at all.

It also produced a bug worth recording, and the bug is the rule working. Zooming
in resolves a collapsed band into its steps and therefore *creates* a running
rope — and nothing was waking the loop. The one open step in the demo sat
motionless, so the surface said *nothing is happening* about something that was.
Waking on every redraw is the only place that catches it, because every case ends
in a redraw.

**Zoom is the answer to a hundred threads, and it has to aggregate rather than
shrink.** Below fifty-eight pixels a thread is not drawn step by step: there is
nothing to see, and the marks would be lying about their own precision. It
collapses to one band that states what it stands for — *6 steps · 58m*. That is
`zoom.ts`'s existing rule, which exists because a viewer sampling at one rate
cannot faithfully observe change faster than half of it, and a picture that
pretends otherwise invites a reader to find structure in aliasing.

**On colour: hue is identity, and Apple's dark system palette is used for it.**
`systemBlue`, `systemOrange`, `systemGreen`, `systemPurple`, `systemPink` and the
rest, in a fixed order so a thread keeps its colour across a zoom, a pan and a
scene change. State never moves the hue — it moves luminance and texture. Two
channels, two meanings, one legend that states both.

**And the biology, since it was raised.** The agent files are the DNA: stored,
inert, and copied from. A thread is the transcript — one strand, carrying a task
through the machinery, folding where it must. The agents are the proteins that
act on it. The braid is the one place the double helix is literally right: two
transcripts held by one machine at one moment. In this data there is no such
moment — the surface says so rather than drawing the mark unexplained — and the
mark exists, tested, for the first time there is.

---

## 8. Status

**The bundle is the demo, as of 2026-08-23.** `/demo/` is `build-helix.ts`: one
axis that is time, every flow a strand wound around it, and depth standing for
attention. The swimlane view in this document and the desk before it are both
still built, still tested, and no longer published — the desk is what `make up`
serves.

The bundle is not a different idea from the swimlanes; it is the same data with
the second dimension spent differently. Swimlanes spend Y on *which agent holds
it*, which is legible and does not scale: eleven agents is eleven rows. The
bundle spends Y and Z on *which strand*, and puts the agent on the strand as a
body riding it — which is both the RNA-and-polymerase picture and the right way
round, because an agent holds a task for a while and hands it on, and it is the
task that persists.

What the bundle can do that the swimlanes cannot: **turn**. Rotation is a way of
paying attention — a strand comes forward without anything else moving out of the
way — and because it is a claim about what deserves looking at, it carries the
same address rule as every finding on the surface. `assertJustified` throws on a
reason with no address.

What it costs: strands at 2π/n are legible as a bundle up to maybe eight, and
past that the same aggregation problem returns in a new shape.
That was the author's call, made when the sketch was working and the desk was
live beside it, and the reasoning is worth recording because it is not obvious:

The two surfaces answer different questions. The desk answers *what is the state
and what can I do* — it has drag, drop, advance, the gesture that puts an agent
on a flow. The thread view answers *what happened, when, and what is happening
now*. For a visitor who has thirty seconds and no account, the second question is
the one worth answering, and the first screen is the only screen most people see.

What is lost by not publishing the desk is real and should be said: the phase-5
gesture — dragging `INSPECTOR` onto a flow — has no equivalent here yet. Selecting
a segment and pressing *Ask an agent* gives the same finding with the same
citation rule, but it is a click on a panel rather than a thing you do with your
hands, and that difference is the whole of `doc/15`.

**It still has not been measured against anything.** `ai-ui/src/threads.ts` is the
layout, with fourteen tests asserting the properties above;
`ai-ui/scripts/build-threads.ts` renders it to one self-contained file over the
two real projects.

The next move is not more surface. It is `doc/04`'s stopwatch, run on both with
the same person and the same flow — which is the thing [NEXT.md](../NEXT.md) has
been asking for since before either of them existed, and which now has two
candidates to compare instead of one to defend.

## 9. Making it read as a helix, which was four errors and no decoration

Reference offered: an illustration of a replication fork — two backbones, base
pairs filling the tube, polymerase riding it. The instruction was to take the
idea the way Ive would, which means taking the *principles* and refusing the
artifact: one projection, hard occlusion, and a ladder whose rhythm is what turns
two curves into one object. Nothing borrowed for its looks.

What that turned up was four separate errors, none of them about style.

**The pitch was longer than the work.** `PITCH` was a constant ninety minutes. A
flow in these projects runs about an hour, so a strand existed for less than one
turn and *could not wind*: five flows, five slow arcs, crossing. A pitch fixed in
advance is honest right up until it is longer than the thing it is supposed to
measure, and then it measures nothing. It is now a quarter — call it two and a
half turns — of how long a typical flow in *this* scope actually runs, and the
readout says how long a turn is, the way a map says how long an inch is. The
number is on screen either way; this one is true of what you are looking at.

**The window opened on a keyhole.** `resetWin` fits the frame to the most recent
cluster of work, which was the right fix for two-thirds of an empty canvas and
the wrong one here: it opened twenty-eight minutes cut out of three days, and a
keyhole onto a coil shows a curve. `TWIST_FLOOR_PX` already refuses to draw a
turn narrower than the marks available to draw it. A turn *wider than the frame*
is the same failure upside down, and now has the same guard — the opening window
is at least wide enough that a turn is no wider than the rope is thick. Zooming
in past it stays allowed, because up close a helix really is a long slow arc.

**Depth was quantised per step.** Width and opacity came from each segment's
*mean* depth, so a strand changed thickness in one jump at a step boundary and
stayed flat in between. A coil does not read from brightness; it reads from a
thickness that changes continuously as the curve turns away. The strand is now
drawn in chunks of three samples — about five a turn — each with the width its own
depth asks for, and the dash pattern carries its accumulated length as an offset
so `carried nothing forward` and `no verdict yet` survive being cut up. Opacity
is out of the depth business entirely: what it says now is what happened.

**The taper was four to one, which the projection says and the screen refused.**
At one pixel the far half of every turn stopped being a rope and became a wire,
and a wire crossing the whole amplitude reads as a separate straight object laid
over the coil. In any drawing of a helix the far side is barely narrower; what
tells you it is behind is that the near side covers it. Occlusion is the depth
cue, width is the confirmation, and the ratio came down to under two to one.

The rungs were the one borrowed element, and they had to earn it. A rung is the
boundary of a step — a moment something was recorded to have started or finished —
so the density of rungs is the density of recorded events, and a stretch of
strand with no rungs is a stretch where nothing was written down. Drawn across
the whole tube they looked like grid lines, because a base pair has a second
backbone to hold at the far end and here there is nothing there. So a rung stops
at the axis, where something really is, and is also the second thing it always
was: a tick against the time line, dropped from the moment it marks.

Everything above changed geometry or removed a channel. Not one of it added a
mark that is not a measurement.
