/**
 * Delivery guarantees for observations: retry with backoff, Retry-After,
 * no retry on caller errors, a bounded buffer that drops the oldest, flush
 * on exit, and the optional disk spool.
 */
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { resolveConfig } from "../config.js";
import { __test__ as exitHooks } from "../exit-flush.js";
import { observationStats } from "../prewarm.js";
import { getSharedRuntime, __test__ as runtimeTest } from "../runtime.js";
import { ObservationTransport, type TransportOptions } from "../transport.js";
import type { Observation } from "../types.js";

function obs(id: string): Observation {
  return {
    sdk_version: "tensorcost-node/test",
    provider: "openai",
    model: "gpt-4o-mini",
    operation: "chat.completions",
    request_at: "2026-11-15T00:00:00.000Z",
    response_at: "2026-11-15T00:00:01.000Z",
    input_tokens: 10,
    output_tokens: 20,
    cost_usd_cents: null,
    status: "success",
    error_message: null,
    correlation_id: id,
  };
}

type Reply = number | Response | Error | "hang";

/**
 * A fetch stand-in. Token exchange always succeeds; each observation POST
 * takes the next reply from `script` (the last one repeats).
 */
function server(script: Reply[]) {
  const posts: { body: string; at: number }[] = [];
  let inFlight = 0;
  let maxInFlight = 0;
  let exchanges = 0;
  const fetchImpl: typeof fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as URL).toString();
    if (url.includes("sdk-token/exchange")) {
      exchanges += 1;
      return new Response(JSON.stringify({ token: "tok", expires_in: 3600 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    posts.push({ body: init.body as string, at: Date.now() });
    const reply = script.length > 1 ? script.shift()! : script[0];
    if (reply === "hang") {
      return new Promise<Response>((_, reject) => {
        init.signal?.addEventListener("abort", () => {
          const e = new Error("aborted");
          e.name = "AbortError";
          reject(e);
        });
      });
    }
    inFlight += 1;
    maxInFlight = Math.max(maxInFlight, inFlight);
    await Promise.resolve();
    inFlight -= 1;
    if (reply instanceof Error) throw reply;
    if (reply instanceof Response) return reply;
    return new Response(null, { status: reply });
  };
  return {
    fetchImpl,
    posts,
    ids: () => posts.map((p) => JSON.parse(p.body).correlation_id as string),
    get maxInFlight() {
      return maxInFlight;
    },
    get exchanges() {
      return exchanges;
    },
    setScript(next: Reply[]) {
      script.splice(0, script.length, ...next);
    },
  };
}

function transport(
  fetchImpl: typeof fetch,
  extra: Partial<TransportOptions> = {},
): ObservationTransport {
  return new ObservationTransport({
    baseUrl: "http://api.local",
    apiKey: "k",
    tenantId: "t-1",
    fetchImpl,
    // Deterministic backoff: the top of the jitter range, so delays are
    // exactly 1 s, 2 s, 4 s ... 60 s.
    random: () => 1,
    ...extra,
  });
}

async function settle(): Promise<void> {
  await vi.advanceTimersByTimeAsync(0);
  for (let i = 0; i < 20; i += 1) await Promise.resolve();
}

let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  warn = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  warn.mockRestore();
  exitHooks.reset();
});

describe("retry with backoff", () => {
  it("retries after a 503 and delivers exactly once", async () => {
    const s = server([503, 202]);
    const t = transport(s.fetchImpl);
    t.post(obs("a"));
    await settle();
    expect(s.posts).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(1_000);
    expect(s.posts).toHaveLength(2);

    // Nothing more is sent once it has landed.
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(s.posts).toHaveLength(2);
    expect(t.stats()).toMatchObject({ buffered: 0, inFlight: 0, dropped: 0 });
  });

  it("resends the same body with the same correlation_id", async () => {
    const s = server([503, new Error("ECONNRESET"), 202]);
    const t = transport(s.fetchImpl);
    t.post(obs("corr-1"));
    await settle();
    await vi.advanceTimersByTimeAsync(1_000);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(s.posts).toHaveLength(3);
    expect(new Set(s.posts.map((p) => p.body)).size).toBe(1);
    expect(s.ids()).toEqual(["corr-1", "corr-1", "corr-1"]);
    expect(t.stats().buffered).toBe(0);
  });

  it("honours Retry-After on a 429", async () => {
    const s = server([
      new Response(null, { status: 429, headers: { "Retry-After": "7" } }),
      202,
    ]);
    const t = transport(s.fetchImpl);
    t.post(obs("a"));
    await settle();
    expect(s.posts).toHaveLength(1);

    // The 1 s backoff would have fired by now; Retry-After holds it.
    await vi.advanceTimersByTimeAsync(6_900);
    expect(s.posts).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(200);
    expect(s.posts).toHaveLength(2);
    expect(s.posts[1].at - s.posts[0].at).toBeGreaterThanOrEqual(7_000);
  });

  it("reads an HTTP-date Retry-After", async () => {
    const retryAt = new Date(Date.now() + 5_000).toUTCString();
    const s = server([
      new Response(null, { status: 429, headers: { "Retry-After": retryAt } }),
      202,
    ]);
    const t = transport(s.fetchImpl);
    t.post(obs("a"));
    await settle();
    await vi.advanceTimersByTimeAsync(3_000);
    expect(s.posts).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(2_100);
    expect(s.posts).toHaveLength(2);
  });

  it.each([400, 401, 403, 404, 413, 422])("does not retry a %i", async (status) => {
    const s = server([status]);
    const t = transport(s.fetchImpl);
    t.post(obs("a"));
    await settle();
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(s.posts).toHaveLength(1);
    expect(warn).toHaveBeenCalledWith(
      `tensorcost: observation POST failed status=${status}; not retried ` +
        "(1 rejected since the last warning, 1 in total)",
    );
    expect(t.stats()).toMatchObject({ buffered: 0, inFlight: 0, rejected: 1, dropped: 0 });
  });

  it("warns about rejected observations once, then at most once a minute", async () => {
    const s = server([400]);
    const t = transport(s.fetchImpl);
    for (let i = 0; i < 50; i += 1) t.post(obs(`r-${i}`));
    await settle();
    const rejectedWarnings = () =>
      warn.mock.calls.filter((c) => String(c[0]).includes("not retried")).length;
    expect(s.posts).toHaveLength(50);
    expect(rejectedWarnings()).toBe(1);
    await vi.advanceTimersByTimeAsync(61_000);
    t.post(obs("late"));
    await settle();
    expect(rejectedWarnings()).toBe(2);
    expect(warn).toHaveBeenLastCalledWith(
      "tensorcost: observation POST failed status=400; not retried " +
        "(50 rejected since the last warning, 51 in total)",
    );
    expect(t.stats().rejected).toBe(51);
  });

  it("retries network errors and timeouts", async () => {
    const s = server([new Error("ECONNREFUSED"), "hang", 202]);
    const t = transport(s.fetchImpl, { timeoutMs: 2_000 });
    t.post(obs("a"));
    await settle();
    await vi.advanceTimersByTimeAsync(1_000); // retry 1: hangs
    expect(s.posts).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(2_000); // timeout fires
    await vi.advanceTimersByTimeAsync(2_000); // second backoff
    expect(s.posts).toHaveLength(3);
    expect(t.stats().buffered).toBe(0);
  });

  it("retries when the token exchange itself is down", async () => {
    let exchangeCalls = 0;
    const posts: string[] = [];
    const fetchImpl: typeof fetch = async (input, init = {}) => {
      const url = String(input);
      if (url.includes("sdk-token/exchange")) {
        exchangeCalls += 1;
        if (exchangeCalls === 1) return new Response(null, { status: 503 });
        return new Response(JSON.stringify({ token: "tok" }), { status: 200 });
      }
      posts.push(init.body as string);
      return new Response(null, { status: 202 });
    };
    const t = transport(fetchImpl);
    t.post(obs("a"));
    await settle();
    expect(posts).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(posts).toHaveLength(1);
  });

  it("a TensorCost outage loses nothing: delivered once each after it comes back, with no flush", async () => {
    // The 1.8.0 field test: inference-proxy scaled to 0, the gateway
    // answers 502 then 503 to the token exchange; 12 calls are made; the
    // proxy comes back; nobody calls flush. Through 1.8.0 all 12 were gone.
    let down = true;
    let exchanges = 0;
    const delivered: string[] = [];
    const fetchImpl: typeof fetch = async (input, init = {}) => {
      const url = String(input);
      if (url.includes("sdk-token/exchange")) {
        exchanges += 1;
        if (down) return new Response(null, { status: exchanges === 1 ? 502 : 503 });
        return new Response(JSON.stringify({ token: "tok", expires_in: 900 }), { status: 200 });
      }
      if (down) return new Response(null, { status: 503 });
      delivered.push(JSON.parse(init.body as string).correlation_id as string);
      return new Response(null, { status: 202 });
    };
    const t = transport(fetchImpl);
    const ids = Array.from({ length: 12 }, (_, i) => `call-${i}`);
    for (const id of ids) t.post(obs(id));
    await settle();
    await vi.advanceTimersByTimeAsync(30_000);
    expect(delivered).toHaveLength(0);
    expect(t.stats()).toMatchObject({ buffered: 12, dropped: 0 });
    // While down, one probe per backoff step, not one exchange per observation.
    expect(exchanges).toBeLessThan(12);

    down = false;
    await vi.advanceTimersByTimeAsync(60_000);
    expect([...delivered].sort()).toEqual([...ids].sort());
    expect(t.stats()).toMatchObject({ buffered: 0, inFlight: 0, dropped: 0, deliveredAfterRetry: 12 });
    expect(warn.mock.calls.some((c) => String(c[0]).startsWith("tensorcost: observation delivery recovered"))).toBe(true);

    // Nothing is sent twice afterwards.
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(delivered).toHaveLength(12);
  });

  it("with a token already held, a 503 on the POST is retried until it lands, once", async () => {
    let failing = 0;
    let exchanges = 0;
    const accepted: string[] = [];
    const fetchImpl: typeof fetch = async (input, init = {}) => {
      if (String(input).includes("sdk-token/exchange")) {
        exchanges += 1;
        return new Response(JSON.stringify({ token: "tok", expires_in: 900 }), { status: 200 });
      }
      if (failing > 0) {
        failing -= 1;
        return new Response(null, { status: 503 });
      }
      accepted.push(JSON.parse(init.body as string).correlation_id as string);
      return new Response(null, { status: 202 });
    };
    const t = transport(fetchImpl);
    t.post(obs("warm"));
    await settle();
    failing = 4;
    for (const id of ["x", "y"]) t.post(obs(id));
    await settle();
    await vi.advanceTimersByTimeAsync(30_000);
    expect(accepted).toEqual(["warm", "x", "y"]);
    expect(t.stats()).toMatchObject({ buffered: 0, dropped: 0, deliveredAfterRetry: 2 });
    expect(exchanges).toBe(1);
  });

  it("doubles the wait from 1 s and caps it at 60 s", async () => {
    const s = server([503]);
    const t = transport(s.fetchImpl);
    t.post(obs("a"));
    await settle();
    for (let i = 0; i < 9; i += 1) await vi.advanceTimersByTimeAsync(60_000);
    const gaps = s.posts.slice(1).map((p, i) => p.at - s.posts[i].at);
    expect(gaps.slice(0, 8)).toEqual([
      1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 60_000, 60_000,
    ]);
    expect(t.stats().buffered).toBe(1);
  });

  it("sends one probe at a time while the endpoint is failing", async () => {
    const s = server([503]);
    const t = transport(s.fetchImpl);
    t.post(obs("first"));
    await settle();
    for (let i = 0; i < 20; i += 1) t.post(obs(`o${i}`));
    await settle();
    // Still backing off: the 20 new ones wait instead of piling on.
    expect(s.posts).toHaveLength(1);
    s.setScript([202]);
    await vi.advanceTimersByTimeAsync(1_000);
    await settle();
    expect(s.posts).toHaveLength(22);
    expect(t.stats().buffered).toBe(0);
  });

  it("gives up on an observation after the retry horizon", async () => {
    const s = server([503]);
    const t = transport(s.fetchImpl, { maxRetryAgeMs: 10_000 });
    t.post(obs("a"));
    await settle();
    await vi.advanceTimersByTimeAsync(60_000);
    const sent = s.posts.length;
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(s.posts.length).toBe(sent);
    expect(t.stats()).toMatchObject({ buffered: 0, dropped: 1 });
  });

  it("post() stays synchronous while the buffer is busy", () => {
    const s = server(["hang"]);
    const t = transport(s.fetchImpl);
    const t0 = performance.now();
    for (let i = 0; i < 2_000; i += 1) t.post(obs(`o${i}`));
    expect(performance.now() - t0).toBeLessThan(200);
  });
});

describe("bounded buffer", () => {
  it("drops the oldest when full and counts the drops", async () => {
    const s = server([503]);
    const t = transport(s.fetchImpl, { maxBufferedObservations: 3 });
    t.post(obs("a"));
    await settle(); // a failed, now in the retry buffer
    for (const id of ["b", "c", "d", "e", "f"]) t.post(obs(id));
    await settle();
    expect(t.stats()).toMatchObject({ buffered: 3, dropped: 3 });
    expect(t.droppedCount).toBe(3);

    s.setScript([202]);
    await vi.advanceTimersByTimeAsync(1_000);
    await settle();
    expect(s.ids().slice(1).sort()).toEqual(["d", "e", "f"]);
  });

  it("caps the buffer by approximate bytes too", async () => {
    const s = server([503]);
    const size = JSON.stringify(obs("x0")).length;
    const t = transport(s.fetchImpl, { maxBufferedBytes: size * 2 + 1 });
    t.post(obs("x0"));
    await settle();
    for (let i = 1; i <= 4; i += 1) t.post(obs(`x${i}`));
    await settle();
    expect(t.stats()).toMatchObject({ buffered: 2, dropped: 3 });
  });

  it("warns about drops at most once a minute", async () => {
    const s = server([503]);
    const t = transport(s.fetchImpl, { maxBufferedObservations: 1 });
    const dropWarnings = () =>
      warn.mock.calls.filter((c) => String(c[0]).includes("dropped")).length;
    for (let i = 0; i < 50; i += 1) t.post(obs(`o${i}`));
    await settle();
    expect(dropWarnings()).toBe(1);
    await vi.advanceTimersByTimeAsync(30_000);
    for (let i = 0; i < 50; i += 1) t.post(obs(`p${i}`));
    await settle();
    expect(dropWarnings()).toBe(1);
    await vi.advanceTimersByTimeAsync(31_000);
    t.post(obs("q"));
    await settle();
    expect(dropWarnings()).toBe(2);
  });

  it("does not drop when only the in-flight limit is reached", async () => {
    const s = server(["hang"]);
    const t = transport(s.fetchImpl);
    for (let i = 0; i < 1_100; i += 1) t.post(obs(`o${i}`));
    await settle();
    expect(t.stats()).toMatchObject({ inFlight: 1_024, buffered: 76, dropped: 0 });
  });
});

describe("flush", () => {
  it("drains the queue and the retry buffer", async () => {
    const s = server([
      new Response(null, { status: 429, headers: { "Retry-After": "30" } }),
      202,
    ]);
    const t = transport(s.fetchImpl);
    t.post(obs("buffered"));
    await settle();
    t.post(obs("fresh"));
    const done = t.flush(3_000);
    await vi.advanceTimersByTimeAsync(100);
    expect(await done).toBe(0);
    expect(s.ids()).toEqual(["buffered", "buffered", "fresh"]);
    expect(t.stats()).toMatchObject({ buffered: 0, inFlight: 0 });
  });

  it("returns once the endpoint is still down instead of waiting out the timeout", async () => {
    const s = server([503]);
    const t = transport(s.fetchImpl);
    t.post(obs("a"));
    const started = Date.now();
    let finished = 0;
    let left = -1;
    void t.flush(3_000).then((n) => {
      finished = Date.now();
      left = n;
    });
    // The 1 s backoff fits in the 3 s budget, so it waits for that retry;
    // the next wait (2 s more) would land on the deadline, so it returns.
    await vi.advanceTimersByTimeAsync(3_100);
    expect(finished).toBeGreaterThan(0);
    expect(finished - started).toBeLessThanOrEqual(1_100);
    expect(left).toBe(1);
    expect(t.stats().buffered).toBe(1);
  });
});

describe("flush on exit", () => {
  it("drains the queue and the retry buffer from beforeExit", async () => {
    const s = server([503, 202]);
    const t = transport(s.fetchImpl, { flushOnExit: true });
    t.post(obs("retrying"));
    await settle();
    t.post(obs("queued"));
    await settle();
    const done = exitHooks.beforeExit();
    await vi.advanceTimersByTimeAsync(1_100);
    await done;
    expect(s.ids()).toEqual(["retrying", "retrying", "queued"]);
    expect(t.stats()).toMatchObject({ buffered: 0, inFlight: 0 });
  });

  it("never holds the process past the cap", async () => {
    const s = server(["hang"]);
    const t = transport(s.fetchImpl, { flushOnExit: true });
    t.post(obs("a"));
    await settle();
    let finished = false;
    void exitHooks.beforeExit().then(() => {
      finished = true;
    });
    await vi.advanceTimersByTimeAsync(exitHooks.EXIT_FLUSH_MS + 100);
    expect(finished).toBe(true);
    // The aborted request is back in the buffer for the spool or the count.
    expect(t.stats()).toMatchObject({ inFlight: 0, buffered: 1 });
  });

  it("runs once per idle period, not on every beforeExit", async () => {
    const s = server([202]);
    const t = transport(s.fetchImpl, { flushOnExit: true });
    t.post(obs("a"));
    const first = exitHooks.beforeExit();
    await vi.advanceTimersByTimeAsync(100);
    await first;
    const second = exitHooks.beforeExit();
    await vi.advanceTimersByTimeAsync(100);
    await second;
    expect(s.posts).toHaveLength(1);
    expect(t.needsExitFlush()).toBe(false);
  });

  it("leaves transports with flushOnExit off alone", async () => {
    const s = server([202]);
    const t = transport(s.fetchImpl, { flushOnExit: false });
    t.post(obs("a"));
    expect(exitHooks.registered()).not.toContain(t);
  });

  it("counts what is left at exit when there is no spool", async () => {
    const s = server(["hang"]);
    const t = transport(s.fetchImpl, { flushOnExit: true });
    t.post(obs("a"));
    await settle();
    exitHooks.exit();
    expect(t.stats().dropped).toBe(1);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("1 observation(s) were not delivered"),
    );
  });
});

describe("disk spool", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "tc-spool-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  // Spool writes are real fs promises, which fake timers do not hold back.
  const waitForSpoolWrites = (t: ObservationTransport) => t.spoolIdle();

  it("round-trips buffer overflow and exit leftovers through the next start", async () => {
    const down = server([503]);
    const a = transport(down.fetchImpl, {
      maxBufferedObservations: 2,
      spoolDir: dir,
      flushOnExit: true,
    });
    a.post(obs("o1"));
    await settle();
    for (const id of ["o2", "o3", "o4", "o5"]) a.post(obs(id));
    await settle();
    await waitForSpoolWrites(a);
    expect(a.stats()).toMatchObject({ buffered: 2, spooled: 3, dropped: 0 });

    exitHooks.exit(); // what is still in memory goes to disk
    expect(a.stats()).toMatchObject({ buffered: 0, spooled: 5, dropped: 0 });
    const lines = readdirSync(dir)
      .flatMap((f) => readFileSync(join(dir, f), "utf8").split("\n"))
      .filter(Boolean);
    expect(lines.map((l) => JSON.parse(l).correlation_id).sort()).toEqual([
      "o1", "o2", "o3", "o4", "o5",
    ]);
    const originalBodies = new Set(down.posts.map((p) => p.body));
    expect(lines.filter((l) => originalBodies.has(l))).toContain(down.posts[0].body);

    exitHooks.reset();
    const up = server([202]);
    const b = transport(up.fetchImpl, { spoolDir: dir });
    vi.useRealTimers();
    await b.replayDone();
    await b.flush(1_000);
    expect(up.ids().sort()).toEqual(["o1", "o2", "o3", "o4", "o5"]);
    expect(readdirSync(dir)).toEqual([]);
  });

  it("keeps the spool under its size cap by removing the oldest segments", async () => {
    const down = server([503]);
    const size = JSON.stringify(obs("s000")).length + 1;
    const t = transport(down.fetchImpl, {
      maxBufferedObservations: 1,
      spoolDir: dir,
      spoolMaxBytes: size * 10,
      spoolSegmentBytes: size * 3,
    });
    t.post(obs("s000"));
    await settle();
    for (let i = 1; i <= 40; i += 1) {
      t.post(obs(`s${String(i).padStart(3, "0")}`));
      await settle();
      await waitForSpoolWrites(t);
    }
    const files = readdirSync(dir);
    const bytes = files
      .map((f) => readFileSync(join(dir, f), "utf8").length)
      .reduce((x, y) => x + y, 0);
    expect(bytes).toBeLessThanOrEqual(size * 10);
    const kept = files
      .flatMap((f) => readFileSync(join(dir, f), "utf8").split("\n"))
      .filter(Boolean)
      .map((l) => JSON.parse(l).correlation_id);
    // The newest survive; the oldest were rotated out and counted.
    expect(kept).toContain("s039");
    expect(kept).not.toContain("s001");
    expect(t.stats().dropped).toBeGreaterThan(0);
  });
});

describe("wrap() configuration", () => {
  const env = { ...process.env };
  afterEach(() => {
    process.env = { ...env };
    runtimeTest.clearRuntimes();
  });

  it("flushes on exit by default and can be turned off", () => {
    expect(resolveConfig({ apiKey: "k", tenantId: "t" }).flushOnExit).toBe(true);
    expect(resolveConfig({ apiKey: "k", tenantId: "t", flushOnExit: false }).flushOnExit).toBe(false);
    process.env.TENSORCOST_FLUSH_ON_EXIT = "0";
    expect(resolveConfig({ apiKey: "k", tenantId: "t" }).flushOnExit).toBe(false);
    process.env.TENSORCOST_FLUSH_ON_EXIT = "false";
    expect(resolveConfig({ apiKey: "k", tenantId: "t" }).flushOnExit).toBe(false);
  });

  it("takes the spool directory from options or TENSORCOST_SPOOL_DIR", () => {
    expect(resolveConfig({ apiKey: "k", tenantId: "t" }).spoolDir).toBeNull();
    process.env.TENSORCOST_SPOOL_DIR = "/var/spool/tc";
    expect(resolveConfig({ apiKey: "k", tenantId: "t" }).spoolDir).toBe("/var/spool/tc");
    expect(resolveConfig({ apiKey: "k", tenantId: "t", spoolDir: "/x" }).spoolDir).toBe("/x");
  });

  it("hands buffer limits and exit flushing to the shared transport", () => {
    const { transport: t } = getSharedRuntime({
      apiKey: "k",
      tenantId: "t",
      baseUrl: "http://api.local",
      maxBufferedObservations: 5,
    });
    expect(t.flushOnExit).toBe(true);
    expect(exitHooks.registered()).toContain(t);
    for (let i = 0; i < 20; i += 1) (t as unknown as { enqueue(e: unknown): void }).enqueue({ body: "{}", bytes: 2, firstAt: i });
    expect(t.stats()).toMatchObject({ buffered: 5, dropped: 15 });
    expect(observationStats()).toMatchObject({ buffered: 5, dropped: 15 });
  });
});
