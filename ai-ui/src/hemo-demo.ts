/**
 * The second real project on the desk — and the one that disagrees with the first.
 *
 * ## Why two projects, and why these two
 *
 * `cochlea-demo.ts` is a project where **truth is derivable**. The eigenvalues
 * of a fixed-free string have a closed form; `truth/` computes it and is
 * forbidden to import `src/`; a gate is arithmetic against a number nobody gets
 * to argue with.
 *
 * That is the easy half, and a repository that only showed that half would be
 * making a claim about a narrow world. Most work worth doing has no closed form.
 *
 * `hemo-verified` is the other half. Blood flow through a tube has exact
 * solutions in two idealised cases and nothing in the cases that matter, so
 * there is no oracle to check a prediction against. The usual move at that point
 * is to have a model judge it and report the judgement confidently.
 *
 * This project does the other thing. It builds a panel of seven physics checks,
 * corrupts *exact* solutions by amounts chosen in advance so the true error is
 * known by construction, and then **measures the panel** — how well does the
 * judge actually rank errors it has never been told about?
 *
 * The answer is humbling on purpose. The composite reaches 0.906. Six of the
 * seven oracles alone are close to a coin flip; the weakest, A5, is 0.521. You
 * only learn that by measuring, and almost nobody measures.
 *
 * > When you can derive the answer, derive it and let code check.
 * > When you cannot, do not substitute a confident model — measure how good your
 * > judge actually is, and publish the number with the hash of what produced it.
 *
 * Those two sentences are why this scope exists next to the cochlea one.
 *
 * ## Where these numbers come from
 *
 * Every constant below is read out of `projects/hemo-verified/gates/reports/h0.json`,
 * the attested artifact, and `test/hemo-demo.test.ts` asserts each one against
 * that file. The demo therefore cannot drift from the project, and if somebody
 * regenerates `h0.json` and the numbers move, the build says so rather than the
 * website quietly showing last month's result.
 *
 * That test is not ceremony. Two cells of the same table in the project's README
 * had been wrong — A5 and A6 transposed, and A4 recorded as 0.706 — and they were
 * only caught by a script that resolves each published number to its artifact.
 */

/**
 * H0 as attested. Field for field, `gates/reports/h0.json`.
 *
 * Written at full precision rather than rounded for display. A number rounded at
 * the point it enters the demo can no longer be compared against its source, and
 * comparing it against its source is the entire discipline here.
 */
export const H0 = {
  n: 98,
  accepted: 48,
  rejected: 32,
  escalated: 18,
  aucComposite: 0.9056331246990852,
  spearmanComposite: 0.822093176586156,
  falseAcceptRate: 0.020833333333333332,
  killThreshold: 0.8,
  badFraction: 0.3163265306122449,
  /** Each oracle scored alone, which is the point of the table. */
  perOracle: {
    A1: 0.5847376023110256,
    A2: 0.5895522388059702,
    A3: 0.8382282137698603,
    A4: 0.6521425132402504,
    A5: 0.5209436687530091,
    A6: 0.5219065960519981,
    A10: 0.6391429947038999,
  },
  /** The content address of each oracle at the moment it was run. */
  oracleHashes: {
    A1: "0e3f8a0c1e75e0d3",
    A2: "0e7295bbf6230eef",
    A3: "aa6cd43798ab17f4",
    A4: "a5fa0bee914cf5f5",
    A5: "3cb9f1a9b33b31c1",
    A6: "1bd10a4ffe9a0c99",
    A10: "e8c4c47d198d684c",
  },
  /**
   * The stack it was produced on.
   *
   * Recorded because the alternative was discovered the hard way: an artifact
   * with no environment cannot tell *this disagrees* from *this was produced
   * somewhere else*, and the second flow in this scope is entirely about that
   * distinction.
   */
  environment: { python: "3.13.12", numpy: "2.5.2", scipy: "1.18.1", machine: "x86_64" },
} as const;

/** A4 alone, measured on a second machine. The finding of 2026-08-23. */
export const A4_ELSEWHERE = 0.652;
export const A4_HERE = 0.706;
/** How many of A4's 98 measurements sit at exactly zero — the tie block. */
export const A4_TIES = 66;
export const A4_DIFFER = 49;

/** The ablation: the panel with its two strongest members removed. */
export const ABLATION_WITHOUT_A3_A4 = 0.896;

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const f4 = (v: number) => v.toFixed(4);
const f3 = (v: number) => v.toFixed(3);

/**
 * The seven agents of the hemodynamics lab.
 *
 * `ORACULISTA` and `EVALUADOR` are deliberately separate, and neither may do the
 * other's job: the agent that declares what counts as a failure must not be the
 * agent that finds out whether it happened. That separation is the same one the
 * cochlea scope enforces between `DERIVADOR` and `CONSTRUCTOR`, and it is the
 * only structural defence against a threshold quietly moving to meet a result.
 */
export function hemoAgents() {
  return [
    {
      name: "ORACULISTA",
      description:
        "Declares the oracles and their thresholds before anything is run, and hashes each one.",
      tools: ["read", "write"],
      child: false,
      missing: false,
    },
    {
      name: "PERTURBADOR",
      description:
        "Corrupts exact solutions by amounts chosen in advance, so the true error is known rather than estimated.",
      tools: ["read", "write", "execute"],
      child: true,
      missing: false,
    },
    {
      name: "EJECUTOR",
      description: "Runs the panel over every row. Never sees the true error.",
      tools: ["read", "execute"],
      child: true,
      missing: false,
    },
    {
      name: "EVALUADOR",
      description:
        "Scores the judge against the truth it was kept away from. Produces the AUCs.",
      tools: ["read", "execute"],
      child: true,
      missing: false,
    },
    {
      name: "AUDITOR-H0",
      description:
        "Holds the pre-registered kill threshold. Its only power is to refuse a freeze.",
      tools: ["read"],
      child: false,
      missing: false,
    },
    {
      name: "REPRODUCTOR",
      description:
        "Rebuilds the run from a clean clone and asks whether the answer survives a different machine.",
      tools: ["read", "execute"],
      child: false,
      missing: false,
    },
    {
      name: "ATESTIGUADOR",
      description:
        "Writes the artifact: the numbers, the oracle hashes, and the stack that produced them.",
      tools: ["read", "write", "publish"],
      child: false,
      missing: false,
    },
  ];
}

interface Stage {
  agent: string;
  said: string;
  state: "done" | "failed" | "blocked";
  /**
   * The observation an attempt closed with.
   *
   * `null` is not "green with nothing to show". It is the wire state `unknown`
   * in [bus.ts](bus.ts): no artifact was recorded, so nothing may be claimed.
   * Exactly one step in this scope sets it, and that step is the finding.
   */
  observation: { digest: string; source: string } | null;
  series?: number[];
  note: string;
}

function flow(
  id: string,
  title: string,
  goal: string,
  state: string,
  stages: Stage[],
  updatedAt: number,
): Record<string, unknown> {
  const steps = stages.map((st, index) => ({
    index,
    state: st.state,
    agent: st.agent,
    intent: `Call your \`delegate\` tool with agent="${st.agent}". The task: ${goal}`,
    result: st.said,
    attempts: [
      {
        n: 1,
        state: st.state,
        runId: `run-${id}-${index}`,
        error: st.state === "done" ? null : `held at step ${index}`,
        observation: st.observation,
      },
    ],
    ...(st.series ? { series: st.series } : {}),
    contribution: { carried: 1, inputTokens: 2000, note: st.note },
  }));
  return {
    id,
    title,
    goal,
    state,
    updatedAt,
    done: steps.filter((s) => s.state === "done").length,
    total: steps.length,
    steps,
  };
}

/** Per-oracle AUCs as a series, in the order the table publishes them. */
const ORACLE_SERIES = [
  H0.perOracle.A3,
  H0.perOracle.A4,
  H0.perOracle.A10,
  H0.perOracle.A2,
  H0.perOracle.A1,
  H0.perOracle.A6,
  H0.perOracle.A5,
];

export function hemoFlows(at: number) {
  return [
    /**
     * The open question, first, because the desk lists newest first and a
     * visitor should land on work in progress rather than on something settled.
     *
     * This flow is the one that earns the scope its place. Its last step is
     * `blocked` with **no observation at all**, which draws the wire into it as
     * `unknown` — and the whole argument of that state is that a picture which
     * renders "was produced somewhere else" the same as "disagrees" has thrown
     * away the distinction that matters.
     */
    flow(
      "flow-a4-machines",
      "A4 alone, on a second machine",
      "find out whether H0 survives being run somewhere else",
      "blocked",
      [
        {
          agent: "REPRODUCTOR",
          said:
            "Built the suite from a clean clone on a different BLAS and ran `make reproduce`. " +
            `Python ${H0.environment.python} / numpy ${H0.environment.numpy} / scipy ${H0.environment.scipy} ` +
            "on one side; a different stack on the other.",
          state: "done",
          observation: { digest: "repro-clean", source: "eval/reproduce.py" },
          note: "the instrument that was missing until 2026-08-23",
        },
        {
          agent: "EVALUADOR",
          said:
            `Composite AUC ${f4(H0.aucComposite)}, Spearman ${f4(H0.spearmanComposite)}, ` +
            `ACCEPT ${H0.accepted} / ESCALATE ${H0.escalated} / REJECT ${H0.rejected}, ` +
            `false-accept ${pct(H0.falseAcceptRate)} — every one of them bit-identical to the ` +
            "attested artifact. Then A4 alone: " +
            `${f3(A4_HERE)} here, ${f3(A4_ELSEWHERE)} there.`,
          state: "done",
          observation: { digest: "h0-composite", source: "gates/reports/h0.json" },
          series: ORACLE_SERIES,
          note: "the composite is untouched; one member of the panel is not",
        },
        {
          agent: "REPRODUCTOR",
          said:
            `${A4_DIFFER} of ${H0.n} A4 measurements differ in their last decimals, which is ordinary. ` +
            `What is not: ${A4_TIES} of the ${H0.n} are exactly 0.0, so a single uncorrupted case ` +
            "sitting at 1.03e-13 on one machine and 0.0 on the other crosses into a " +
            `${A4_TIES}-wide tie block and drags the rank statistic with it. A4 is a HARD gate — ` +
            "it contributes pass/fail against a threshold far above the noise floor, never its score.",
          state: "done",
          observation: { digest: "a4-ties", source: "gates/reports/h0.json" },
          note: "localised: a rank statistic over a tie block, not a physics disagreement",
        },
        {
          agent: "AUDITOR-H0",
          said:
            "Held. The two runs were produced on different stacks, so this cannot be recorded as " +
            "*A4 disagrees* — only as *A4 was measured somewhere else*. There is no artifact that " +
            "would settle it, and inventing the verdict either way is the failure this project is about. " +
            "Open until the second stack is pinned and re-run.",
          state: "blocked",
          // Deliberately null. See `Stage.observation`.
          observation: null,
          note: "no artifact exists for this comparison; the honest state is unknown, not red",
        },
      ],
      at - 3 * 3600_000,
    ),

    /**
     * H0 itself: the pre-registered kill gate, frozen because it survived.
     *
     * Note the shape of the argument in the steps. The threshold is written down
     * *before* the run and hashed, the perturbations have known true error by
     * construction, and the agent that scores the judge never sees the truth the
     * judge was kept away from. None of that is a property of the model doing
     * the work; all of it is a property of how the flow is arranged.
     */
    flow(
      "flow-h0",
      "H0 — can physics checks catch a wrong flow field?",
      "measure whether the oracle panel ranks errors it was never shown, and kill the project if it does not",
      "done",
      [
        {
          agent: "ORACULISTA",
          said:
            "Seven oracles declared and hashed before a single row was scored: A1 global mass, " +
            "A2 local mass, A3 momentum residual, A4 no-slip, A5 inlet vs BC, A6 energy budget, " +
            `A10 temporal envelope. Kill threshold ${H0.killThreshold} composite AUC, written down first. ` +
            "A7 and A8 are specified and unbuilt — neither quantity exists on an analytical pipe.",
          state: "done",
          observation: { digest: H0.oracleHashes.A3, source: "oracles/thresholds.yaml" },
          note: "the threshold is declared by an agent that never sees the result",
        },
        {
          agent: "PERTURBADOR",
          said:
            `${H0.n} predictions built from exact Poiseuille and Womersley solutions, corrupted by ` +
            "noise, bias, slip, divergence and phase in amounts chosen in advance. " +
            `${pct(H0.badFraction)} of them are worse than 5% true error — known by construction, ` +
            "not estimated, which is the only reason any of the rest is measurable.",
          state: "done",
          observation: { digest: "rows-98", source: "gates/reports/h0.json" },
          note: "true error known by construction",
        },
        {
          agent: "EJECUTOR",
          said:
            "Scored every row with every oracle. The panel never receives the true error, and the " +
            "corruption label is not in its input.",
          state: "done",
          observation: { digest: "panel-run", source: "gates/reports/h0.json" },
          note: "the judge is kept away from the answer",
        },
        {
          agent: "EVALUADOR",
          said:
            `Composite AUC ${f4(H0.aucComposite)}, Spearman ${f4(H0.spearmanComposite)}, ` +
            `false-accept ${pct(H0.falseAcceptRate)}. Alone, six of the seven are near a coin flip — ` +
            `A3 ${f3(H0.perOracle.A3)} is the only strong one, and A5 is ${f3(H0.perOracle.A5)}. ` +
            `Remove A3 and A4 entirely and the panel still reaches ${f3(ABLATION_WITHOUT_A3_A4)}: ` +
            "individually weak gates cover different failures.",
          state: "done",
          observation: { digest: "auc-per-oracle", source: "gates/reports/h0.json" },
          series: ORACLE_SERIES,
          note: "the portfolio beats every member; this is the number nobody publishes",
        },
        {
          agent: "AUDITOR-H0",
          said:
            `GATE-H0 green: ${f4(H0.aucComposite)} against a kill threshold of ${H0.killThreshold}, ` +
            "declared before the run. Survives. The one false accept is a Womersley bias case whose " +
            "true error is 0.050 against a bad-case boundary of 0.050 — the boundary itself, not a miss.",
          state: "done",
          observation: { digest: "gate-h0", source: "gates/reports/h0.json" },
          note: "gate H0 discriminates, and passed",
        },
        {
          agent: "ATESTIGUADOR",
          said:
            "Wrote gates/reports/h0.json: every number above, the content hash of all seven oracles, " +
            `and the stack — python ${H0.environment.python}, numpy ${H0.environment.numpy}, ` +
            `scipy ${H0.environment.scipy}, ${H0.environment.machine}. The environment field exists ` +
            "because an earlier artifact did not have one, and could not have been produced by the " +
            "code committed beside it. Freeze authorised.",
          state: "done",
          observation: { digest: "h0-attested", source: "gates/reports/h0.json" },
          note: "a report whose provenance nobody checks is a report, not an attestation",
        },
      ],
      at - 26 * 3600_000,
    ),

    /**
     * The honest limit, as a flow rather than as a footnote.
     *
     * H0 shows the gates rank errors *of a kind the author thought of*, because
     * the corruptions and the oracles were designed by the same person. That is
     * a real ceiling and it is not fixable by making H0 nicer. It is stated here
     * as a `blocked` flow with one step, so the desk shows it as open work
     * rather than letting a green scope imply the question is closed.
     */
    flow(
      "flow-h1",
      "H1 — the surrogate that does not exist yet",
      "find out whether the panel ranks the errors a trained surrogate actually makes",
      "blocked",
      [
        {
          agent: "ORACULISTA",
          said:
            "H0's corruptions and H0's oracles were designed by the same author. So H0 shows the " +
            "panel ranks errors of a kind we thought of, and cannot show it ranks the errors a " +
            "trained surrogate makes. No surrogate has been trained. Every number in this scope is " +
            "also on a rigid axisymmetric tube — the simplest geometry with an exact solution, and " +
            "nothing like an atrium. This is the next gate, not a refinement of the last one.",
          state: "blocked",
          observation: null,
          note: "stated as open work, because a scope with nothing red in it reads as a finished one",
        },
      ],
      at - 9 * 3600_000,
    ),
  ];
}
