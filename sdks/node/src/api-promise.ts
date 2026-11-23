/**
 * Recording a provider call without changing what the caller gets back.
 *
 * openai-node and @anthropic-ai/sdk return an `APIPromise` from `create()`:
 * a Promise subclass that only reads the body when it is awaited, with
 * `withResponse()`, `asResponse()` and `_thenUnwrap()` on top. Their own
 * helpers depend on those (Anthropic's `messages.stream()` calls
 * `create(...).withResponse()`, openai's `chat.completions.parse()` calls
 * `create(...)._thenUnwrap()`). Through 1.7.0 the wrapper replaced
 * `create()` with an async function, so callers got a plain Promise and
 * every one of those threw a TypeError.
 *
 * Two ways back to the same surface:
 *
 *  - When nothing TensorCost does has to finish before the provider is
 *    called (observe with the published layer already known), the wrapper
 *    calls the provider in the same tick and `observeApiCall` returns the
 *    provider's own APIPromise, derived with its `_thenUnwrap`, so it is an
 *    instance of the client's class.
 *  - When a check has to finish first (govern's /admit, route's proxy, or a
 *    cold process waiting up to 250 ms for the published layer), the
 *    wrapper returns a `TensorCostAPIPromise`: the same methods, delegating
 *    to the provider's APIPromise once the call is made, or answering from
 *    the proxy's response in route mode.
 *
 * Either way the call is recorded once, whichever way the caller reads it:
 * awaited, `withResponse()`, `asResponse()`, or not read yet when the
 * response arrives (then the usage is read off a clone of the response).
 * A stream is recorded when it ends, with the usage it carried (OpenAI's
 * final chunk with stream_options.include_usage; Anthropic's message_start
 * and message_delta), by watching the iterator the caller pulls from, never
 * by reading ahead of the caller.
 *
 * The provider's id for the call rides along with the usage (`id` on a
 * body or an OpenAI chunk, `message.id` on Anthropic's message_start). The
 * server keys the same call seen by the SDK, the proxy, a gateway or a cost
 * import on it (provider_response_id).
 */

import { billingAccountOfApiPromise, billingAccountOfCall } from "./billing-account.js";

export interface Usage {
  input: number | null;
  output: number | null;
  /** The provider's id for the call: chatcmpl-/cmpl-, msg_, Bedrock's request id. */
  responseId?: string | null;
}

/**
 * ObservationDto.provider_response_id is @MaxLength(200), and the DTO
 * refuses the whole observation over a bad field, so a longer id is left off.
 */
export const MAX_RESPONSE_ID_LENGTH = 200;

/** `v` as a provider response id the server will take, else null. */
export function responseId(v: unknown): string | null {
  if (typeof v !== "string") return null;
  const t = v.trim();
  return t.length > 0 && t.length <= MAX_RESPONSE_ID_LENGTH ? t : null;
}

/** `body.id`, the id OpenAI and Anthropic put on a response body or chunk. */
export function responseIdOf(body: unknown): string | null {
  return body && typeof body === "object" ? responseId((body as Record<string, unknown>)["id"]) : null;
}

/** The observation's provider_response_id, absent when there is none. */
export function responseIdField(id: unknown): { provider_response_id?: string } {
  const rid = responseId(id);
  return rid !== null ? { provider_response_id: rid } : {};
}

const NO_USAGE: Usage = { input: null, output: null };

export interface StreamUsageTracker {
  see(event: unknown): void;
  usage(): Usage;
}

export interface CallRecorder {
  /** The URL the client sends to (billing account). */
  url: string | null;
  usageOf(data: unknown): Usage;
  streamTracker(): StreamUsageTracker;
  success(usage: Usage, billingAccount: string | null): void;
  error(err: unknown): void;
}

type AnyApiPromise = PromiseLike<unknown> & {
  _thenUnwrap(fn: (data: unknown, props: { response?: Response }) => unknown): AnyApiPromise;
  asResponse(): Promise<Response>;
  withResponse(): Promise<{ data: unknown; response: Response; request_id?: string | null }>;
  parse?: () => Promise<unknown>;
};

/** openai-node / @anthropic-ai/sdk APIPromise, by shape. */
export function isApiPromise(v: unknown): v is AnyApiPromise {
  const o = v as Record<string, unknown> | null;
  return (
    !!o &&
    (typeof o === "object" || typeof o === "function") &&
    typeof o["then"] === "function" &&
    typeof o["_thenUnwrap"] === "function" &&
    typeof o["asResponse"] === "function"
  );
}

function isAsyncIterable(v: unknown): v is AsyncIterable<unknown> {
  return (
    !!v &&
    typeof v === "object" &&
    typeof (v as Record<symbol, unknown>)[Symbol.asyncIterator] === "function"
  );
}

function isEventStream(resp: Response): boolean {
  return (resp.headers.get("content-type") ?? "").includes("text/event-stream");
}

function once(rec: CallRecorder) {
  let done = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  return {
    get done() {
      return done;
    },
    set timer(t: ReturnType<typeof setTimeout> | undefined) {
      timer = t;
    },
    success(usage: Usage, billingAccount: string | null) {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try {
        rec.success(usage, billingAccount);
      } catch {
        /* recording never breaks the call */
      }
    },
    error(err: unknown) {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try {
        rec.error(err);
      } catch {
        /* recording never breaks the call */
      }
    },
  };
}

/** How long an unread streamed response waits to be read before it is recorded without usage. */
const UNREAD_STREAM_RECORD_MS = 60_000;

/**
 * Record `call` (what the provider's create() returned) and hand back the
 * same kind of object: the provider's APIPromise when it is one, otherwise
 * a promise of the same value.
 */
export function observeApiCall(call: unknown, rec: CallRecorder): unknown {
  const o = once(rec);
  if (!isApiPromise(call)) {
    return Promise.resolve(call).then(
      async (data) => {
        const billingAccount = await billingAccountOfApiPromise(call, rec.url);
        if (isAsyncIterable(data)) {
          return tapStream(data, rec.streamTracker(), (u) => o.success(u, billingAccount), o.error);
        }
        o.success(rec.usageOf(data), billingAccount);
        return data;
      },
      (err: unknown) => {
        o.error(err);
        throw err;
      },
    );
  }

  let reading = false;
  let rawWanted = false;
  const billing = (headers: Headers | null | undefined) => billingAccountOfCall(rec.url, headers ?? null);

  const out = call._thenUnwrap((data, props) => {
    const headers = props?.response?.headers;
    if (isAsyncIterable(data)) {
      return tapStream(data, rec.streamTracker(), (u) => o.success(u, billing(headers)), o.error);
    }
    o.success(rec.usageOf(data), billing(headers));
    return data;
  });

  // Reading the body (await, then/catch/finally, withResponse) goes through
  // parse() on both SDKs; marking it tells the response hook below that the
  // transform above will record the call.
  const target = out as AnyApiPromise & Record<string, unknown>;
  if (typeof target.parse === "function") {
    const parse = target.parse;
    target.parse = function (this: unknown, ...a: unknown[]) {
      reading = true;
      const p = (parse as (...x: unknown[]) => Promise<unknown>).apply(this, a);
      void Promise.resolve(p).catch(o.error);
      return p;
    };
  } else {
    reading = true;
  }

  // The raw Response, for a caller that never reads the body through us.
  const recordRaw = (resp: Response) => {
    if (reading || o.done) return;
    if (isEventStream(resp)) {
      if (rawWanted) o.success(NO_USAGE, billing(resp.headers));
      return;
    }
    let copy: Response;
    try {
      copy = resp.clone();
    } catch {
      o.success(NO_USAGE, billing(resp.headers));
      return;
    }
    copy.json().then(
      (d: unknown) => o.success(rec.usageOf(d), billing(resp.headers)),
      () => o.success(NO_USAGE, billing(resp.headers)),
    );
  };

  const asResponse = target.asResponse;
  target.asResponse = function (this: unknown) {
    rawWanted = true;
    return (asResponse as () => Promise<Response>).call(this).then((resp) => {
      recordRaw(resp);
      return resp;
    });
  };

  // The response is in: record it now unless someone is reading the body.
  // A stream nobody reads yet waits (reading it later records its usage).
  responseOf(call).then((resp) => {
    if (reading || o.done) return;
    if (isEventStream(resp)) {
      const t = setTimeout(() => {
        if (!reading) o.success(NO_USAGE, billing(resp.headers));
      }, UNREAD_STREAM_RECORD_MS);
      (t as { unref?: () => void }).unref?.();
      o.timer = t;
      return;
    }
    recordRaw(resp);
  }, o.error);

  return out;
}

/**
 * The call's raw Response, without asking the APIPromise for it.
 * `asResponse()` isn't free on @anthropic-ai/sdk: called before the body is
 * parsed it ends the call's tracing span there, without the body. The
 * props promise underneath it (`_responseProps()` there, `responsePromise`
 * on openai-node) has no such effect; asResponse() is the fallback.
 */
function responseOf(call: AnyApiPromise): Promise<Response> {
  const c = call as unknown as Record<string, unknown>;
  const props =
    typeof c["_responseProps"] === "function"
      ? (c["_responseProps"] as () => unknown).call(call)
      : c["responsePromise"];
  if (props && typeof (props as PromiseLike<unknown>).then === "function") {
    return Promise.resolve(props as PromiseLike<{ response: Response }>).then((p) => p.response);
  }
  return call.asResponse();
}

/**
 * Watch a provider stream as the caller reads it and record at its end.
 * The SDKs' Stream objects keep their source in an `iterator` property that
 * `for await`, `tee()` and `toReadableStream()` all call, so that is what
 * is wrapped, on the same object, keeping its class and controller.
 */
export function tapStream<T>(
  stream: T,
  tracker: StreamUsageTracker,
  end: (usage: Usage) => void,
  fail: (err: unknown) => void,
): T {
  const s = stream as Record<string | symbol, unknown>;
  const wrap = (it: AsyncIterator<unknown>) => tapIterator(it, tracker, end, fail);
  if (typeof s["iterator"] === "function") {
    const orig = s["iterator"] as (...a: unknown[]) => AsyncIterator<unknown>;
    s["iterator"] = function (this: unknown, ...a: unknown[]) {
      return wrap(orig.apply(this, a));
    };
    return stream;
  }
  const orig = s[Symbol.asyncIterator] as (() => AsyncIterator<unknown>) | undefined;
  if (typeof orig === "function") {
    Object.defineProperty(s, Symbol.asyncIterator, {
      value: function (this: unknown) {
        return wrap(orig.call(this));
      },
      configurable: true,
      writable: true,
    });
    return stream;
  }
  end(tracker.usage());
  return stream;
}

function tapIterator(
  it: AsyncIterator<unknown>,
  tracker: StreamUsageTracker,
  end: (usage: Usage) => void,
  fail: (err: unknown) => void,
): AsyncIterableIterator<unknown> {
  let settled = false;
  const finish = () => {
    if (settled) return;
    settled = true;
    end(tracker.usage());
  };
  const failed = (err: unknown) => {
    if (settled) return;
    settled = true;
    fail(err);
  };
  return {
    async next(...a: [] | [unknown]) {
      let r: IteratorResult<unknown>;
      try {
        r = await it.next(...a);
      } catch (err) {
        failed(err);
        throw err;
      }
      if (r.done) {
        finish();
      } else {
        try {
          tracker.see(r.value);
        } catch {
          /* a chunk we can't read costs its usage, not the stream */
        }
      }
      return r;
    },
    async return(value?: unknown) {
      finish();
      return typeof it.return === "function" ? it.return(value) : { done: true, value };
    },
    async throw(err?: unknown) {
      failed(err);
      if (typeof it.throw === "function") return it.throw(err);
      throw err;
    },
    [Symbol.asyncIterator]() {
      return this;
    },
  };
}

const num = (v: unknown): number | null => (typeof v === "number" ? v : null);

/** OpenAI chat chunks: usage rides on the final chunk with stream_options.include_usage. */
export function openAiStreamTracker(): StreamUsageTracker {
  let usage: Usage = NO_USAGE;
  let id: string | null = null;
  return {
    see(chunk) {
      id ??= responseIdOf(chunk);
      const u = (chunk as { usage?: Record<string, unknown> | null } | null)?.usage;
      if (u && typeof u === "object") {
        usage = { input: num(u["prompt_tokens"]), output: num(u["completion_tokens"]) };
      }
    },
    usage: () => ({ ...usage, responseId: id }),
  };
}

/**
 * Anthropic events: input tokens on message_start, cumulative output tokens
 * on message_delta (which newer API versions also give input tokens on).
 */
export function anthropicStreamTracker(): StreamUsageTracker {
  let input: number | null = null;
  let output: number | null = null;
  let id: string | null = null;
  return {
    see(event) {
      const e = event as Record<string, any> | null;
      if (!e || typeof e !== "object") return;
      if (e.type === "message_start") {
        id ??= responseIdOf(e.message);
        const u = e.message?.usage;
        if (u && typeof u === "object") {
          input = num(u.input_tokens) ?? input;
          output = num(u.output_tokens) ?? output;
        }
      } else if (e.type === "message_delta" && e.usage && typeof e.usage === "object") {
        output = num(e.usage.output_tokens) ?? output;
        input = num(e.usage.input_tokens) ?? input;
      }
    },
    usage: () => ({ input, output, responseId: id }),
  };
}

/** What a deferred call resolves to: the provider's call, or the proxy's answer. */
export type CallSource =
  | { kind: "call"; call: unknown }
  | { kind: "proxy"; data: unknown; response: Response | null };

type Unwrap = (data: unknown, props: { response: Response | null }) => unknown;

/**
 * The APIPromise surface for a call TensorCost had to check before making:
 * then/catch/finally read the result, `withResponse()`, `asResponse()` and
 * `_thenUnwrap()` behave as the provider SDK's do. Like theirs it never
 * rejects unread: a refusal surfaces when the caller reads it.
 */
export class TensorCostAPIPromise<T = unknown> extends Promise<T> {
  private readonly source: Promise<CallSource>;
  private readonly unwrap: Unwrap | null;
  private parsed: Promise<T> | null = null;
  private apiTarget: AnyApiPromise | null = null;

  constructor(source: Promise<CallSource>, unwrap: Unwrap | null = null) {
    super((resolve) => resolve(undefined as T));
    this.source = source;
    this.unwrap = unwrap;
    source.catch(() => undefined);
  }

  static override get [Symbol.species]() {
    return Promise;
  }

  static rejected<T = unknown>(err: unknown): TensorCostAPIPromise<T> {
    return new TensorCostAPIPromise<T>(Promise.reject(err));
  }

  /** The provider's APIPromise with this promise's unwrap applied, derived once. */
  private api(call: AnyApiPromise): AnyApiPromise {
    if (!this.apiTarget) {
      const unwrap = this.unwrap;
      this.apiTarget = unwrap
        ? call._thenUnwrap((d, props) => unwrap(d, { response: props?.response ?? null }))
        : call;
    }
    return this.apiTarget;
  }

  private parse(): Promise<T> {
    if (!this.parsed) {
      this.parsed = this.source.then(async (s) => {
        if (s.kind === "call") {
          if (isApiPromise(s.call)) return (await this.api(s.call)) as T;
          const data = await s.call;
          return (this.unwrap ? this.unwrap(data, { response: null }) : data) as T;
        }
        return (this.unwrap ? this.unwrap(s.data, { response: s.response }) : s.data) as T;
      });
    }
    return this.parsed;
  }

  override then<A = T, B = never>(
    onfulfilled?: ((value: T) => A | PromiseLike<A>) | null,
    onrejected?: ((reason: unknown) => B | PromiseLike<B>) | null,
  ): Promise<A | B> {
    return this.parse().then(onfulfilled, onrejected);
  }

  override catch<B = never>(onrejected?: ((reason: unknown) => B | PromiseLike<B>) | null): Promise<T | B> {
    return this.parse().catch(onrejected);
  }

  override finally(onfinally?: (() => void) | null): Promise<T> {
    return this.parse().finally(onfinally);
  }

  _thenUnwrap<U>(fn: (data: T, props: { response: Response | null }) => U): TensorCostAPIPromise<U> {
    const prev = this.unwrap;
    return new TensorCostAPIPromise<U>(this.source, (d, props) =>
      fn((prev ? prev(d, props) : d) as T, props),
    );
  }

  async withResponse(): Promise<{ data: T; response: Response | null; request_id: string | null }> {
    const s = await this.source;
    if (s.kind === "call") {
      if (isApiPromise(s.call)) {
        const r = await this.api(s.call).withResponse();
        return { data: r.data as T, response: r.response, request_id: r.request_id ?? requestId(r.response) };
      }
      return { data: await this.parse(), response: null, request_id: null };
    }
    const data = await this.parse();
    return { data, response: s.response, request_id: requestId(s.response) };
  }

  async asResponse(): Promise<Response> {
    const s = await this.source;
    if (s.kind === "call") {
      if (isApiPromise(s.call)) return this.api(s.call).asResponse();
      throw new TypeError("tensorcost: this client's create() has no asResponse()");
    }
    return replayResponse(s.data, s.response);
  }
}

function requestId(resp: Response | null | undefined): string | null {
  try {
    return resp?.headers.get("x-request-id") ?? null;
  } catch {
    return null;
  }
}

/**
 * A readable Response for asResponse() on a proxied call, whose own body
 * the SDK has already read: the JSON again, or the stream re-encoded as SSE
 * (reading it reads the stream, so its usage is still recorded).
 */
function replayResponse(data: unknown, resp: Response | null): Response {
  const status = resp?.status ?? 200;
  const headers = new Headers(resp?.headers ?? {});
  if (isAsyncIterable(data)) {
    const enc = new TextEncoder();
    const it = data[Symbol.asyncIterator]();
    const body = new ReadableStream<Uint8Array>({
      async pull(ctrl) {
        try {
          const r = await it.next();
          if (r.done) {
            ctrl.enqueue(enc.encode("data: [DONE]\n\n"));
            ctrl.close();
          } else {
            ctrl.enqueue(enc.encode(`data: ${JSON.stringify(r.value)}\n\n`));
          }
        } catch (err) {
          ctrl.error(err);
        }
      },
      async cancel() {
        await it.return?.();
      },
    });
    return new Response(body, { status, headers });
  }
  headers.set("content-type", "application/json");
  headers.delete("content-length");
  return new Response(JSON.stringify(data), { status, headers });
}
