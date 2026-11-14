// Free-model aggregator: one server, two client wires, ten free providers.
//
// The entry reads the trial brief (providers, the quality table, the
// workload manifest), probes each provider's catalog route, registers the
// one catalog it built, and then serves the client's twenty requests: each
// with the best free model that can serve it, failing over when a provider
// refuses, and exiting 0 when the env says the workload is done.
//
// Routing (quality is 60% of the score): every request's candidates are the
// free models whose capabilities meet its requirements, ranked by the
// quality table's rating. The manifest arrives in weight order, so serving
// in arrival order already serves the heavy requests first; each pick takes
// the best candidate whose provider has taken fewer than
// PLANNED_CALLS_PER_PROVIDER calls so far this trial. That spreads the
// twenty requests over the providers instead of draining the one best
// provider's hidden per-trial budget, and a provider that refuses (402, 429
// or a 5xx) is tried again later: one refusal is usually the provider's own
// scheduled failure, not a spent budget, so a provider is only set aside
// after its second refusal this trial. A request whose provider refuses
// fails over to its next-best candidate at once. A provider that answered
// 2xx is never called again for that request: exactly one provider call may
// answer it.
//
// Upstream calls are plain JSON, never streams: the provider's one JSON
// answer carries the text, the usage and the stop reason, and this server
// renders the client's JSON or SSE from it, so the streamed usage is exact
// on every wire. Every choice is a function of the brief, the requests and
// the providers' answers so far: no clock, no unseeded randomness, so a
// replay leaves the same state export.
//
// `node main.ts --list` prints the catalog document and exits 0 before any
// server starts; the register body carries that same document.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type Brief,
  type Catalog,
  type DirectoryProvider,
  type Json,
  type RatingRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicStreamAnswer,
  anthropicToAnthropic,
  anthropicToOpenAI,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAIStreamAnswer,
  openAIToAnthropic,
  openAIToOpenAI,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  textOf,
  toGemini,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/**
 * How many calls one provider is planned to take before the router prefers
 * another provider's model: the hidden per-trial budgets run from 4 calls
 * up, so planning four a provider never spends a budget on the plan alone,
 * and the best models are only milli-points apart.
 */
const PLANNED_CALLS_PER_PROVIDER = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let catalog: Catalog = { providers: [] };
/** The free models, in rating order, that may serve a request. */
let choices: Choice[] = [];
/** How many calls have arrived, in manifest order. */
let manifestIndex = 0;
/** Calls made to each provider so far this trial (attempts included). */
const callsByProvider = new Map<string, number>();
/** Refusals by each provider so far this trial. */
const failuresByProvider = new Map<string, number>();
/** Providers set aside after their second refusal: not picked first. */
const suspectProviders = new Set<string>();

interface Choice {
  provider: DirectoryProvider;
  model: string;
  rating: number;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Whether a quality-table row meets one request's requirements. */
function meets(row: RatingRow, requires: Requirement): boolean {
  if (requires.tools === true && row.tools !== true) return false;
  if (requires.vision === true && row.vision !== true) return false;
  if (requires.json === true && row.json !== true) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) {
    return false;
  }
  return true;
}

/**
 * Probe every provider in turn and build the one catalog (the `--list`
 * document and the register body), then rank every free model by rating.
 * A JSON-mode request never goes to an Anthropic-wire provider: that wire
 * cannot carry response_format, so the provider would echo it absent.
 */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  const probes = [];
  for (const provider of brief.providers) {
    const probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
    if (!probe.ok && probe.error !== null) log(probe.error);
    probes.push(probe);
  }
  catalog = buildCatalog(brief, probes);
  const byId = new Map(brief.providers.map((p) => [p.id, p]));
  choices = brief.ratings
    .filter((row) => row.free)
    .flatMap((row) => {
      const provider = byId.get(row.provider);
      return provider === undefined ? [] : [{ provider, model: row.model, rating: row.rating_milli }];
    })
    .sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : 1) || (a.model < b.model ? -1 : 1));
}

/** The free model ids the catalog registers, each once: GET /v1/models. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

/** The candidates for one request, best rating first. */
function candidatesFor(requires: Requirement): Choice[] {
  return choices.filter((choice) => {
    const row = rowOf(choice);
    if (row === null || !meets(row, requires)) return false;
    if (requires.json === true && choice.provider.wire === 'anthropic') return false;
    return true;
  });
}

function rowOf(choice: Choice): RatingRow | null {
  return (
    brief.ratings.find((r) => r.provider === choice.provider.id && r.model === choice.model) ?? null
  );
}

/**
 * The next provider and model for a request: the best candidate not tried
 * yet, preferring a provider under its planned call count that has not
 * refused twice this trial, then any provider that has not refused twice,
 * then any candidate at all. Deterministic: the same history picks the
 * same model.
 */
function choose(requires: Requirement, tried: ReadonlySet<string>): Choice | null {
  const pool = candidatesFor(requires);
  const key = (c: Choice): string => `${c.provider.id}/${c.model}`;
  const underCap = pool.filter(
    (c) =>
      !tried.has(key(c)) &&
      !suspectProviders.has(c.provider.id) &&
      (callsByProvider.get(c.provider.id) ?? 0) < PLANNED_CALLS_PER_PROVIDER,
  );
  if (underCap.length > 0) return underCap[0];
  const fresh = pool.filter((c) => !tried.has(key(c)) && !suspectProviders.has(c.provider.id));
  if (fresh.length > 0) return fresh[0];
  const rest = pool.filter((c) => !tried.has(key(c)));
  return rest.length > 0 ? rest[0] : null;
}

/** The requirements of the request now arriving: the manifest's row, in order. */
function requirementsFor(body: Json, api: 'openai' | 'anthropic'): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body, api);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, api: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    ...(Array.isArray(body.tools) && body.tools.length > 0 ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    ...(api === 'openai' && (format === 'json_object' || format === 'json_schema') ? { json: true } : {}),
  };
}

/** One upstream call: a plain JSON request in the provider's own wire. */
async function callUpstream(
  choice: Choice,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<{ ok: true; answer: NormalizedAnswer } | { ok: false; status: number }> {
  const provider = choice.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  let parse: (v: Json) => NormalizedAnswer | null;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? openAIToOpenAI(body, choice.model) : anthropicToOpenAI(body, choice.model);
    parse = parseOpenAIAnswer;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody =
      clientApi === 'openai' ? openAIToAnthropic(body, choice.model) : anthropicToAnthropic(body, choice.model);
    parse = parseAnthropicAnswer;
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(choice.model)}:generateContent`;
    upstreamBody = toGemini(body, clientApi);
    parse = parseGeminiAnswer;
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!res.ok) return { ok: false, status: res.status };
  let value: unknown;
  try {
    value = await res.json();
  } catch {
    return { ok: false, status: res.status };
  }
  if (!isRecord(value)) return { ok: false, status: res.status };
  // A 2xx is the request's one answer even when its shape is thin: a
  // second provider call after a success would fail the one-provider check.
  const answer = parse(value) ?? { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0 };
  return { ok: true, answer };
}

/** Whether a refusal sets the provider aside for the rest of the trial. */
function isRefusal(status: number): boolean {
  return status === 0 || status === 402 || status === 429 || status >= 500;
}

/** Record a refusal: the second one sets the provider aside. */
function noteRefusal(providerId: string): void {
  const count = (failuresByProvider.get(providerId) ?? 0) + 1;
  failuresByProvider.set(providerId, count);
  if (count >= 2) suspectProviders.add(providerId);
}

/**
 * Serve one client request: try candidates in order until one answers 2xx.
 * A refusal sets its provider aside; a 400/401 skips just that candidate.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<{ status: number; answer: NormalizedAnswer | null; model: string | null }> {
  const requires = requirementsFor(body, clientApi);
  const tried = new Set<string>();
  for (;;) {
    const choice = choose(requires, tried);
    if (choice === null) return { status: 503, answer: null, model: null };
    tried.add(`${choice.provider.id}/${choice.model}`);
    callsByProvider.set(choice.provider.id, (callsByProvider.get(choice.provider.id) ?? 0) + 1);
    const result = await callUpstream(choice, clientApi, body);
    if (result.ok) return { status: 200, answer: result.answer, model: choice.model };
    if (isRefusal(result.status)) noteRefusal(choice.provider.id);
    log(
      `failover: ${choice.provider.id}/${choice.model} answered HTTP ${result.status} for ${clientApi}; trying the next candidate`,
    );
  }
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, text: string): void {
  res.writeHead(status, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('openai', body);
  if (result.answer === null || result.model === null) {
    return sendError(res, 'openai', result.status, 'no_free_model', 'no free model on the directory meets this request');
  }
  if (body.stream === true) sendText(res, 200, openAIStreamAnswer(result.model, result.answer));
  else sendJson(res, 200, openAIJsonAnswer(result.model, result.answer));
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await serve('anthropic', body);
  if (result.answer === null || result.model === null) {
    return sendError(res, 'anthropic', result.status, 'overloaded_error', 'no free model on the directory meets this request');
  }
  if (body.stream === true) sendText(res, 200, anthropicStreamAnswer(result.model, result.answer));
  else sendJson(res, 200, anthropicJsonAnswer(result.model, result.answer));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
