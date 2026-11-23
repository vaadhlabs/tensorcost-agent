/**
 * A stand-in for the APIPromise and Stream classes openai-node and
 * @anthropic-ai/sdk return, with the parts a caller (and the provider SDK's
 * own helpers) lean on: a Promise subclass that only reads the body when it
 * is awaited, `asResponse()`, `withResponse()`, `_thenUnwrap()`, and a
 * Stream whose `iterator` property backs `for await`, `tee()` and
 * `toReadableStream()`. Copied in shape from openai@7.62 core/api-promise
 * and core/streaming, so the suite runs without either package installed.
 */

type Props = { response: Response; options: { stream: boolean } };
type Parse = (props: Props) => Promise<unknown> | unknown;

export class FakeStream<T> {
  constructor(public iterator: () => AsyncIterator<T>) {}
  [Symbol.asyncIterator](): AsyncIterator<T> {
    return this.iterator();
  }
}

async function defaultParse(props: Props): Promise<unknown> {
  if (props.options.stream) {
    const text = await props.response.text();
    const events = text
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith("data:") && l !== "data: [DONE]")
      .map((l) => JSON.parse(l.slice(5).trim()) as unknown);
    return new FakeStream(async function* () {
      for (const e of events) yield e;
    });
  }
  return props.response.json();
}

export class FakeAPIPromise<T = unknown> extends Promise<T> {
  private parsedPromise: Promise<T> | undefined;

  constructor(
    private readonly responsePromise: Promise<Props>,
    private readonly parseResponse: Parse = defaultParse,
  ) {
    super((resolve) => resolve(null as T));
  }

  static get [Symbol.species]() {
    return Promise;
  }

  _thenUnwrap<U>(transform: (data: T, props: Props) => U): FakeAPIPromise<U> {
    return new FakeAPIPromise<U>(this.responsePromise, async (props) =>
      transform((await this.parseResponse(props)) as T, props),
    );
  }

  asResponse(): Promise<Response> {
    return this.responsePromise.then((p) => p.response);
  }

  async withResponse(): Promise<{ data: T; response: Response; request_id: string | null }> {
    const [data, response] = await Promise.all([this.parse(), this.asResponse()]);
    return { data, response, request_id: response.headers.get("x-request-id") };
  }

  private parse(): Promise<T> {
    if (!this.parsedPromise) {
      this.parsedPromise = this.responsePromise.then((p) => this.parseResponse(p) as Promise<T>);
    }
    return this.parsedPromise;
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
}

/** An OpenAI-shaped client whose create() returns a FakeAPIPromise over `respond`. */
export function fakeApiClient(
  respond: (body: Record<string, unknown>) => Response | Promise<Response>,
  shape: "openai" | "anthropic" = "openai",
) {
  const requests: Record<string, unknown>[] = [];
  const create = (body: Record<string, unknown>) => {
    requests.push(body);
    const responsePromise = Promise.resolve().then(async () => {
      const response = await respond(body);
      if (!response.ok) {
        const err = new Error(`${response.status} status code`) as Error & { status: number };
        err.status = response.status;
        throw err;
      }
      return { response, options: { stream: body.stream === true } };
    });
    return new FakeAPIPromise(responsePromise);
  };
  const client =
    shape === "openai"
      ? {
          baseURL: "https://api.openai.com/v1",
          apiKey: "sk-test",
          chat: { completions: { create } },
        }
      : {
          baseURL: "https://api.anthropic.com",
          apiKey: "sk-ant",
          messages: { create },
        };
  return { client: client as any, requests };
}

export const jsonResponse = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });

export const sseResponse = (events: unknown[]) =>
  new Response(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("") + "data: [DONE]\n\n", {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
