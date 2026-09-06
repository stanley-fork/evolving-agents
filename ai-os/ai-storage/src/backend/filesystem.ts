/**
 * The store on disk, and the only way anything is written to it.
 *
 * ## Atomic or nothing
 *
 *   temporary file → fsync → atomic rename → journal commit
 *
 * A note half-written is worse than a note absent, because absence is a hole
 * the archivist will fill and half a note is a claim nobody can check. `rename`
 * within a filesystem is atomic, so a reader either sees the old bytes or the
 * new ones and never a prefix.
 *
 * ## The journal, and what it is for
 *
 * Not durability — the rename gives that. The journal exists so that a crash
 * between two writes of the *same* logical change leaves something that says
 * what was being attempted. A note and its index entry are two files; a crash
 * between them leaves a note nothing points at, and the recovery pass needs to
 * know that was one operation rather than two.
 *
 * ## Paths are capabilities
 *
 * `resolve` is the only function that turns a name into a path, and it refuses
 * anything that leaves the store's root. Every caller goes through it. That is
 * what makes the model's `../../.ssh/id_rsa` a rejected name rather than a read
 * — a name never becomes a path by concatenation anywhere in this component.
 */
import { createHash } from "node:crypto";
import { open, mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve as resolvePath, sep } from "node:path";
import type { Range } from "../provenance/ranges.ts";
import type { SourceReader } from "../provenance/evidence.ts";

export class PathEscapes extends Error {
  constructor(name: string) {
    super(
      `ai-storage: "${name}" resolves outside the store. A name never becomes a path by ` +
        `concatenation here — that is what makes a hallucinated path a rejected name ` +
        `rather than a read.`,
    );
    this.name = "PathEscapes";
  }
}

/** Turn a store-relative name into an absolute path, or refuse. */
export function resolveInside(root: string, name: string): string {
  if (name.includes("\0")) throw new PathEscapes(name);
  const abs = resolvePath(root, name);
  const rel = relative(root, abs);
  if (rel === "" || rel.startsWith("..") || rel.startsWith(".." + sep) || resolvePath(rel) === rel)
    throw new PathEscapes(name);
  return abs;
}

/**
 * Write a file so that a reader never sees a prefix of it.
 *
 * The fsync on the file *and* on its directory both matter: the first makes the
 * bytes durable, the second makes the rename durable. Skipping the second is
 * the classic way to lose a file that every layer above believes was written.
 */
export async function atomicWrite(path: string, bytes: string | Uint8Array): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.${Math.abs(hash32(path + String(bytes.length)))}.tmp`;
  const fh = await open(tmp, "w");
  try {
    await fh.writeFile(bytes);
    await fh.sync();
  } finally {
    await fh.close();
  }
  await rename(tmp, path);
  const dir = await open(dirname(path), "r");
  try {
    await dir.sync();
  } catch {
    // Some platforms refuse to fsync a directory handle. The rename has still
    // happened; what is lost is durability of the rename across a power cut,
    // which is a weaker guarantee than the one this function's callers need and
    // not one worth failing the write over.
  } finally {
    await dir.close();
  }
}

function hash32(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return h;
}

export interface JournalEntry {
  /** A monotonic id within one store. */
  seq: number;
  /** What was being attempted, e.g. `note.create`. */
  op: string;
  /** Store-relative paths this operation touches, all of them. */
  paths: string[];
  /** ISO 8601, passed in rather than read from a clock. */
  at: string;
  state: "begin" | "commit";
}

/**
 * The store.
 *
 * One directory, four scope levels underneath it, and nothing outside it. The
 * only object in this component that touches the filesystem.
 */
export class FileStore {
  readonly root: string;

  constructor(root: string) {
    this.root = resolvePath(root);
  }

  path(name: string): string {
    return resolveInside(this.root, name);
  }

  async init(): Promise<void> {
    for (const d of ["system", "users", "projects", "flows", ".journal"])
      await mkdir(join(this.root, d), { recursive: true });
  }

  async readText(name: string): Promise<string | null> {
    try {
      return await readFile(this.path(name), "utf8");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw err;
    }
  }

  async readJson<T>(name: string): Promise<T | null> {
    const text = await this.readText(name);
    if (text === null) return null;
    return JSON.parse(text) as T;
  }

  async writeJson(name: string, value: unknown): Promise<void> {
    // Sorted keys and a trailing newline: a note that is byte-identical between
    // two runs is a note whose digest can be compared, and a store that
    // re-serialises differently every time makes every diff noise.
    await atomicWrite(this.path(name), JSON.stringify(value, sortedKeys(value), 2) + "\n");
  }

  async writeText(name: string, text: string): Promise<void> {
    await atomicWrite(this.path(name), text);
  }

  async exists(name: string): Promise<boolean> {
    try {
      await stat(this.path(name));
      return true;
    } catch {
      return false;
    }
  }

  async sizeOf(name: string): Promise<number | null> {
    try {
      return (await stat(this.path(name))).size;
    } catch {
      return null;
    }
  }

  async list(name: string): Promise<string[]> {
    try {
      return (await readdir(this.path(name))).sort();
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw err;
    }
  }

  async remove(name: string): Promise<void> {
    await rm(this.path(name), { force: true, recursive: true });
  }

  /** Bytes of `[from, to)`, read without loading the whole file. */
  async slice(name: string, range: Range): Promise<Uint8Array> {
    const fh = await open(this.path(name), "r");
    try {
      const length = range.to - range.from;
      const buf = new Uint8Array(length);
      const { bytesRead } = await fh.read(buf, 0, length, range.from);
      return buf.subarray(0, bytesRead);
    } finally {
      await fh.close();
    }
  }

  // ---- the journal ---------------------------------------------------------

  async #nextSeq(): Promise<number> {
    const names = await this.list(".journal");
    let max = 0;
    for (const n of names) {
      const m = /^(\d+)\.json$/.exec(n);
      if (m) max = Math.max(max, Number(m[1]));
    }
    return max + 1;
  }

  /**
   * Record an intention, run it, record that it finished.
   *
   * A crash leaves a `begin` with no `commit`, which is exactly the information
   * a recovery pass needs: these paths were mid-change. Nothing here rolls
   * back — rolling back a rename that already happened would need a copy of the
   * old bytes, and `pending()` plus a re-run of the operation is both simpler
   * and, because every operation in this component is idempotent by id, enough.
   */
  async transact<T>(op: string, paths: string[], at: string, body: () => Promise<T>): Promise<T> {
    const seq = await this.#nextSeq();
    const name = `.journal/${String(seq).padStart(8, "0")}.json`;
    const entry: JournalEntry = { seq, op, paths, at, state: "begin" };
    await this.writeJson(name, entry);
    const out = await body();
    await this.writeJson(name, { ...entry, state: "commit" });
    return out;
  }

  /** Operations that began and never committed. Empty on a clean store. */
  async pending(): Promise<JournalEntry[]> {
    const out: JournalEntry[] = [];
    for (const n of await this.list(".journal")) {
      const e = await this.readJson<JournalEntry>(`.journal/${n}`);
      if (e && e.state !== "commit") out.push(e);
    }
    return out.sort((a, b) => a.seq - b.seq);
  }

  /** Forget committed history. Never removes a pending entry. */
  async pruneJournal(): Promise<number> {
    let n = 0;
    for (const name of await this.list(".journal")) {
      const e = await this.readJson<JournalEntry>(`.journal/${name}`);
      if (e && e.state === "commit") {
        await this.remove(`.journal/${name}`);
        n += 1;
      }
    }
    return n;
  }

  /** A `SourceReader` over this store's `sources/`, for the provenance pipeline. */
  sourceReader(scopeDir: string): SourceReader {
    const under = (artifact: string) => `${scopeDir}/sources/${artifact}`;
    return {
      sizeOf: async (a) => this.sizeOf(under(a)),
      slice: async (a, r) => this.slice(under(a), r),
      idOf: async (a) => "src_" + createHash("sha256").update(under(a)).digest("hex").slice(0, 16),
      pathOf: async (a) => under(a),
    };
  }
}

/** Stable key order, so the same value always serialises to the same bytes. */
function sortedKeys(_root: unknown) {
  return (_k: string, v: unknown) => {
    if (v === null || typeof v !== "object" || Array.isArray(v)) return v;
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(v as Record<string, unknown>).sort())
      out[k] = (v as Record<string, unknown>)[k];
    return out;
  };
}
