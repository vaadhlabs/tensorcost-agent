/**
 * Process-exit hooks for observation transports.
 *
 * `beforeExit` fires when the event loop has run dry on a normal exit. We
 * drain what is queued for up to EXIT_FLUSH_MS; the timers that wait keep
 * the loop alive only that long, and requests still open at the cap are
 * aborted. It fires again once our work finishes, so a transport is flushed
 * once per idle period, not in a loop.
 *
 * `exit` is synchronous and last: whatever is still in memory goes to the
 * spool when one is configured, and is counted and reported otherwise.
 *
 * SIGTERM is the awkward one. Once any listener exists Node no longer exits
 * on it, so a listener that only flushed would leave a process that ignores
 * SIGTERM. When the app has its own SIGTERM handler, it owns shutdown and
 * we only start the drain alongside it (its eventual exit runs `exit`
 * above). When ours is the only listener, we drain for up to the cap, spool
 * what is left, remove ourselves and re-send SIGTERM, so the process dies
 * the way it would have without us, at most EXIT_FLUSH_MS later. We never
 * call process.exit().
 *
 * Nothing is installed until a transport with flushOnExit (or a spool) is
 * created, and nothing in worker threads, which do not receive signals.
 */

import { isMainThread } from "node:worker_threads";

export const EXIT_FLUSH_MS = 2_000;

export interface ExitAware {
  readonly flushOnExit: boolean;
  /** True when there is something to deliver and no exit drain has run since the last post. */
  needsExitFlush(): boolean;
  flushForExit(timeoutMs: number): Promise<void>;
  /** Synchronous: spool or count whatever is still in memory. */
  spillRemainingSync(): void;
}

const transports = new Set<ExitAware>();
let installed = false;
let sigtermInstalled = false;
let draining: Promise<void> | null = null;

function hasProcessEvents(): boolean {
  return typeof process !== "undefined" && typeof process.on === "function";
}

export function registerExitHooks(t: ExitAware): void {
  transports.add(t);
  if (!hasProcessEvents()) return;
  if (!installed) {
    installed = true;
    process.on("beforeExit", onBeforeExit);
    process.on("exit", onExit);
  }
  if (t.flushOnExit && !sigtermInstalled && isMainThread) {
    sigtermInstalled = true;
    process.on("SIGTERM", onSigterm);
  }
}

export function unregisterExitHooks(t: ExitAware): void {
  transports.delete(t);
}

function drainAll(): Promise<void> {
  if (draining) return draining;
  const work = [...transports].filter((t) => t.flushOnExit && t.needsExitFlush());
  if (work.length === 0) return Promise.resolve();
  draining = Promise.allSettled(work.map((t) => t.flushForExit(EXIT_FLUSH_MS))).then(
    () => {
      draining = null;
    },
  );
  return draining;
}

function onBeforeExit(): Promise<void> {
  return drainAll();
}

function onExit(): void {
  for (const t of transports) {
    try {
      t.spillRemainingSync();
    } catch {
      /* never throw from an exit handler */
    }
  }
}

function onSigterm(): void {
  const appHandlesIt = process.listenerCount("SIGTERM") > 1;
  const drained = drainAll();
  if (appHandlesIt) return;
  void drained.finally(() => {
    onExit();
    process.removeListener("SIGTERM", onSigterm);
    sigtermInstalled = false;
    process.kill(process.pid, "SIGTERM");
  });
}

export const __test__ = {
  EXIT_FLUSH_MS,
  beforeExit: onBeforeExit,
  exit: onExit,
  registered: (): ExitAware[] => [...transports],
  reset(): void {
    transports.clear();
    draining = null;
    if (hasProcessEvents()) {
      process.removeListener("beforeExit", onBeforeExit);
      process.removeListener("exit", onExit);
      process.removeListener("SIGTERM", onSigterm);
    }
    installed = false;
    sigtermInstalled = false;
  },
};
