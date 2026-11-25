/**
 * Opt-in disk spool for observations the in-memory buffer could not keep.
 *
 * Layout: one JSONL file per segment, one observation body per line, the
 * exact bytes that would have been POSTed (so a replay resends the same
 * body and the same correlation_id, and ingestion dedupes it).
 *
 *   tc-obs-<ms>-<pid>-<rand>-<seq>.open    being written by a live process
 *   tc-obs-<ms>-<pid>-<rand>-<seq>.jsonl   sealed (rotated, or the writer exited)
 *   <either>.claimed-<pid>-<rand>          taken by a process replaying it
 *
 * A writer only ever appends to its own `.open` segment and seals it when it
 * reaches the segment size or the process exits. Replay takes sealed
 * segments, plus `.open` and `.claimed` ones nobody has touched for
 * STALE_MS (their process died). Claiming is a rename, so two processes
 * sharing a directory never both take the same file; the worst case is a
 * duplicate line, which the server's correlation_id dedupe absorbs.
 *
 * The directory is kept under `maxBytes` by deleting the oldest sealed
 * segments; the lines in them are counted as dropped.
 *
 * Writes during normal running are async (they happen off the caller's
 * hot path); the exit path writes synchronously because the event loop is
 * gone by then.
 */

import { randomBytes } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

export const DEFAULT_SPOOL_MAX_BYTES = 64 * 1024 * 1024;
export const DEFAULT_SPOOL_SEGMENT_BYTES = 1024 * 1024;
const STALE_MS = 5 * 60_000;
const PREFIX = "tc-obs-";
/** Lines waiting for the async writer before new spills are refused. */
const MAX_PENDING_LINES = 10_000;

export interface SpoolOptions {
  maxBytes?: number;
  segmentBytes?: number;
  /** Called with the number of lines lost (deleted for space, write failed). */
  onLost?: (lines: number, reason: string) => void;
}

export class Spool {
  readonly dir: string;
  private readonly maxBytes: number;
  private readonly segmentBytes: number;
  private readonly onLost: (lines: number, reason: string) => void;
  private readonly instance = `${process.pid}-${randomBytes(4).toString("hex")}`;

  private seq = 0;
  private openPath: string | null = null;
  private openBytes = 0;
  /** Bytes in the directory, measured once and then tracked for our writes. */
  private totalBytes: number | null = null;
  private dirReady = false;

  private pending: string[] = [];
  private writing: string[] | null = null;
  private chain: Promise<void> | null = null;

  constructor(dir: string, opts: SpoolOptions = {}) {
    this.dir = dir;
    this.maxBytes = opts.maxBytes ?? DEFAULT_SPOOL_MAX_BYTES;
    this.segmentBytes = opts.segmentBytes ?? DEFAULT_SPOOL_SEGMENT_BYTES;
    this.onLost = opts.onLost ?? (() => {});
  }

  /** Queue one body for an async append. Never throws, never blocks. */
  append(body: string): boolean {
    if (this.pending.length >= MAX_PENDING_LINES) return false;
    this.pending.push(body);
    if (!this.chain) {
      this.chain = this.drain().finally(() => {
        this.chain = null;
      });
    }
    return true;
  }

  /** Resolves when every queued append has reached the disk (or failed). */
  async idle(): Promise<void> {
    while (this.chain) await this.chain;
  }

  /**
   * Exit path: write whatever is queued (including a batch an async write
   * may not have finished) plus `bodies`, then seal the segment.
   */
  appendSync(bodies: string[]): void {
    const lines = [...(this.writing ?? []), ...this.pending, ...bodies];
    this.pending = [];
    this.writing = null;
    try {
      if (lines.length > 0) {
        if (!this.dirReady) {
          fs.mkdirSync(this.dir, { recursive: true });
          this.dirReady = true;
        }
        if (this.totalBytes === null) this.totalBytes = this.measureSync();
        const data = lines.join("\n") + "\n";
        const size = Buffer.byteLength(data);
        this.makeRoomSync(size);
        fs.appendFileSync(this.currentPath(), data);
        this.openBytes += size;
        this.totalBytes += size;
      }
      this.sealSync();
    } catch (err) {
      this.onLost(lines.length, `spool write failed: ${(err as Error).message}`);
    }
  }

  /**
   * Take the oldest segment left by an earlier run (or a dead process),
   * delete it, and return its lines. Null when there is nothing to take.
   */
  async claimNext(): Promise<string[] | null> {
    let names: string[];
    try {
      names = await fs.promises.readdir(this.dir);
    } catch {
      return null;
    }
    const now = Date.now();
    const candidates = names.filter((n) => n.startsWith(PREFIX)).sort();
    for (const name of candidates) {
      const full = path.join(this.dir, name);
      if (full === this.openPath) continue;
      if (!name.endsWith(".jsonl")) {
        let st: fs.Stats;
        try {
          st = await fs.promises.stat(full);
        } catch {
          continue;
        }
        // rename() bumps ctime, so a fresh claim is never mistaken for stale.
        const touched = Math.max(st.mtimeMs, st.ctimeMs);
        if (now - touched < STALE_MS) continue;
      }
      const base = name.replace(/\.claimed-.*$/, "");
      const claimed = path.join(this.dir, `${base}.claimed-${this.instance}`);
      try {
        await fs.promises.rename(full, claimed);
      } catch {
        continue; // someone else got it
      }
      try {
        const text = await fs.promises.readFile(claimed, "utf8");
        await fs.promises.unlink(claimed);
        if (this.totalBytes !== null) {
          this.totalBytes = Math.max(0, this.totalBytes - Buffer.byteLength(text));
        }
        return text.split("\n").filter((l) => l.length > 0);
      } catch {
        continue;
      }
    }
    return null;
  }

  private async drain(): Promise<void> {
    while (this.pending.length > 0) {
      const batch = this.pending;
      this.pending = [];
      this.writing = batch;
      try {
        if (!this.dirReady) {
          await fs.promises.mkdir(this.dir, { recursive: true });
          this.dirReady = true;
        }
        if (this.totalBytes === null) this.totalBytes = await this.measure();
        const data = batch.join("\n") + "\n";
        const size = Buffer.byteLength(data);
        await this.makeRoom(size);
        await fs.promises.appendFile(this.currentPath(), data);
        this.openBytes += size;
        this.totalBytes += size;
        if (this.openBytes >= this.segmentBytes) await this.seal();
      } catch (err) {
        this.onLost(batch.length, `spool write failed: ${(err as Error).message}`);
      } finally {
        this.writing = null;
      }
    }
  }

  private currentPath(): string {
    if (!this.openPath) {
      this.seq += 1;
      const stamp = String(Date.now()).padStart(15, "0");
      this.openPath = path.join(
        this.dir,
        `${PREFIX}${stamp}-${this.instance}-${String(this.seq).padStart(6, "0")}.open`,
      );
      this.openBytes = 0;
    }
    return this.openPath;
  }

  private async seal(): Promise<void> {
    if (!this.openPath) return;
    const from = this.openPath;
    this.openPath = null;
    try {
      await fs.promises.rename(from, from.replace(/\.open$/, ".jsonl"));
    } catch {
      /* a stale-segment claim may have taken it; nothing to do */
    }
  }

  private sealSync(): void {
    if (!this.openPath) return;
    const from = this.openPath;
    this.openPath = null;
    try {
      fs.renameSync(from, from.replace(/\.open$/, ".jsonl"));
    } catch {
      /* as above */
    }
  }

  private async measure(): Promise<number> {
    let total = 0;
    for (const name of await fs.promises.readdir(this.dir)) {
      if (!name.startsWith(PREFIX)) continue;
      try {
        total += (await fs.promises.stat(path.join(this.dir, name))).size;
      } catch {
        /* gone */
      }
    }
    return total;
  }

  private measureSync(): number {
    let total = 0;
    for (const name of fs.readdirSync(this.dir)) {
      if (!name.startsWith(PREFIX)) continue;
      try {
        total += fs.statSync(path.join(this.dir, name)).size;
      } catch {
        /* gone */
      }
    }
    return total;
  }

  /** Oldest sealed segments first; never our own open one. */
  private sealedOldestFirst(names: string[]): string[] {
    return names
      .filter((n) => n.startsWith(PREFIX) && n.endsWith(".jsonl"))
      .sort()
      .map((n) => path.join(this.dir, n))
      .filter((p) => p !== this.openPath);
  }

  private async makeRoom(incoming: number): Promise<void> {
    for (let pass = 0; pass < 2; pass += 1) {
      if (this.totalBytes! + incoming <= this.maxBytes) return;
      for (const victim of this.sealedOldestFirst(await fs.promises.readdir(this.dir))) {
        if (this.totalBytes! + incoming <= this.maxBytes) return;
        try {
          const text = await fs.promises.readFile(victim, "utf8");
          await fs.promises.unlink(victim);
          this.dropped(text);
        } catch {
          /* raced with a claim */
        }
      }
      // Only our own segment is left to give up; seal it so it can go.
      await this.seal();
    }
    if (this.totalBytes! + incoming > this.maxBytes) {
      throw new Error(`spool directory is over its ${this.maxBytes}-byte cap`);
    }
  }

  private makeRoomSync(incoming: number): void {
    for (let pass = 0; pass < 2; pass += 1) {
      if (this.totalBytes! + incoming <= this.maxBytes) return;
      for (const victim of this.sealedOldestFirst(fs.readdirSync(this.dir))) {
        if (this.totalBytes! + incoming <= this.maxBytes) return;
        try {
          const text = fs.readFileSync(victim, "utf8");
          fs.unlinkSync(victim);
          this.dropped(text);
        } catch {
          /* raced with a claim */
        }
      }
      this.sealSync();
    }
    if (this.totalBytes! + incoming > this.maxBytes) {
      throw new Error(`spool directory is over its ${this.maxBytes}-byte cap`);
    }
  }

  private dropped(text: string): void {
    this.totalBytes = Math.max(0, this.totalBytes! - Buffer.byteLength(text));
    this.onLost(countLines(text), "spool full; oldest segment removed");
  }
}

function countLines(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i += 1) if (text.charCodeAt(i) === 10) n += 1;
  return n;
}
