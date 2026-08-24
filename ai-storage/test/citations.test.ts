/**
 * Every citation in MODEL.json resolves to something that exists.
 *
 * The rest of this repository refuses to render a finding whose address points
 * at nothing. `MODEL.json` is a document of nothing but claims, so the same
 * rule applies to it — and it is the document whose claims decide what every
 * benchmark number means.
 *
 * This is a cheap test that catches an expensive failure: a section renumbered,
 * a document renamed, and suddenly the provenance of the platform's model is a
 * dead link that still looks authoritative.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const model = JSON.parse(readFileSync(join(root, "MODEL.json"), "utf8")) as unknown;

/** Every `citedAs` anywhere in the tree, with the key path that carries it. */
function citations(node: unknown, at: string[] = []): Array<{ where: string; address: string }> {
  if (Array.isArray(node)) return node.flatMap((v, i) => citations(v, [...at, String(i)]));
  if (node === null || typeof node !== "object") return [];
  const out: Array<{ where: string; address: string }> = [];
  for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
    if (k === "citedAs" && typeof v === "string") out.push({ where: at.join(".") || "(root)", address: v });
    else out.push(...citations(v, [...at, k]));
  }
  return out;
}

const found = citations(model);

test("MODEL.json actually carries citations", () => {
  // A version of this file with the citations quietly removed would pass every
  // other test here.
  assert.ok(found.length >= 8, `expected the model's claims to be cited; found ${found.length}`);
});

test("every cited document exists", () => {
  for (const c of found) {
    const [path] = c.address.split("#");
    assert.ok(path, `${c.where}: empty citation`);
    assert.ok(
      existsSync(join(root, path)),
      `${c.where} cites "${c.address}" and ${path} does not exist`,
    );
  }
});

test("every cited anchor exists in the document it names", () => {
  const cache = new Map<string, string>();
  for (const c of found) {
    const [path, anchor] = c.address.split("#");
    if (!anchor) continue;
    if (!cache.has(path!)) cache.set(path!, readFileSync(join(root, path!), "utf8"));
    const text = cache.get(path!)!;
    assert.ok(
      text.includes(`<a id="${anchor}"></a>`),
      `${c.where} cites "${c.address}" and ${path} has no anchor "${anchor}"`,
    );
  }
});

test("nothing claims to be verified while the weights are unreachable", () => {
  // The state this repository is actually in, asserted so it cannot drift.
  // When `scripts/verify-model.ts` has been run against a real server, this
  // test is what has to be edited — deliberately, by somebody who saw the
  // server say so.
  const flags: Array<{ where: string; verified: unknown }> = [];
  const walk = (node: unknown, at: string[]) => {
    if (Array.isArray(node)) return node.forEach((v, i) => walk(v, [...at, String(i)]));
    if (node === null || typeof node !== "object") return;
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
      if (k === "verified") flags.push({ where: at.join(".") || "(root)", verified: v });
      else walk(v, [...at, k]);
    }
  };
  walk(model, []);
  assert.ok(flags.length >= 8, "every claim should carry a verified flag");
  for (const f of flags)
    assert.equal(f.verified, false, `${f.where} claims verification that nothing performed`);
});

test("the effective context in MODEL.json is the one the code enforces", () => {
  // Two copies of the same number is how a configuration file and the code
  // that reads it drift apart until nobody knows which one is running.
  const m = model as { effectiveContext: number; profiles: Record<string, { effectiveContext: number }> };
  assert.equal(m.effectiveContext, 8192);
  for (const [name, p] of Object.entries(m.profiles))
    assert.equal(p.effectiveContext, m.effectiveContext, `profile "${name}" disagrees with the cap`);
});
