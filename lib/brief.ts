// lib/brief.ts — reading the trial brief and building the one catalog.
//
// A trial hands this server a brief (the path in $MAKEFUN_PROMPT_FILE): prose
// wrapped around three JSON documents, found by their `schema` member:
//   - the provider directory (`.../providers/v1`): ten providers, each with an
//     id, wire (openai | anthropic | gemini), auth scheme, key env var, the key
//     the service takes, and the base path to join to $MAKEFUN_ENV_URL.
//   - the quality table (`.../ratings/v1`): every model's rating in
//     milli-points, its context length, what it can do (tools, vision, JSON
//     mode) and whether it is free. This is the sealed truth the scorer uses.
//   - the workload manifest (`.../workload/v1`): the twenty requests the
//     client will send, in order, with each one's api, streaming flag, weight
//     and requirements.
//
// The catalog this server registers (and `--list` prints) is built from the
// directory and the quality table, with each provider's live catalog route
// probed in its own wire to confirm what it lists. A provider whose probe
// fails still contributes its quality-table rows: the table is authoritative,
// the probe is a confirmation, so one refused probe never costs the whole
// provider's catalog checks. Everything here is a pure function of the brief
// and the probe answers: no clock, no randomness, so `--list` and the
// register body are the same document byte for byte.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** Any JSON object. */
export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief lists it. */
export interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path (e.g. `/p/groq`) or an absolute base URL. */
  base: string;
  key_env: string;
  key: string;
}

/** One row of the quality table. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** One request of the workload manifest. */
export interface ManifestRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  /** The sealed per-request best rating, when the table carries it. */
  best: { id: string; rating_milli: number }[];
  manifest: ManifestRequest[];
  error: string | null;
}

/** One model of the catalog document: what `--list` prints and /register sends. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/** One model as a provider's catalog route listed it. */
export interface ListedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
}

export interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  models: ListedModel[];
  error: string | null;
}

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumberOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const asBoolOrNull = (v: unknown): boolean | null =>
  typeof v === 'boolean' ? v : null;

/** Join a provider base (path or absolute URL) to the env URL. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header that carries a provider's key. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function directoryProviderOf(v: unknown): DirectoryProvider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire: Wire | null =
    v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth,
    base,
    key_env: asString(v.key_env),
    key: asString(v.key),
  };
}

function ratingRowOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumberOrNull(v.rating_milli) ?? 0,
    context_length: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestRequestOf(v: unknown): ManifestRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const p = directoryProviderOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = ratingRowOf(v);
          return r === null ? [] : [r];
        });
  const best: { id: string; rating_milli: number }[] =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? (ratingsDoc.best as unknown[]).flatMap((v) => {
          if (!isRecord(v) || asString(v.id) === '') return [];
          return [{ id: asString(v.id), rating_milli: asNumberOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc === null
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const w = manifestRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, manifest, error: null };
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModelOf(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context =
    asNumberOrNull(v.context_length) ??
    asNumberOrNull(v.context_window) ??
    asNumberOrNull(v.inputTokenLimit);
  // A listing that prices everything at zero lists a free model; one that
  // says nothing about price leaves the flag unknown (the table decides).
  let free: boolean | null = asBoolOrNull(v.free);
  if (free === null && isRecord(v.pricing)) {
    const values = Object.values(v.pricing).flatMap((x) => {
      if (typeof x === 'number') return [x];
      if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
      return [];
    });
    if (values.length > 0) free = values.every((n) => Number.isFinite(n) && n === 0);
  }
  return { id, free, context_length: context };
}

/** Probe one provider's catalog route, in its own wire, tolerating failure. */
export async function probeProvider(
  envUrl: string,
  provider: DirectoryProvider,
  timeoutMs = 5000,
): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [], error: `GET ${url} answered no JSON object` };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [], error: `GET ${url} listed no models` };
    const models = list.flatMap((v) => {
      const m = listedModelOf(provider.wire, v);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, models, error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byId = (a: { id: string }, b: { id: string }): number =>
  a.id < b.id ? -1 : a.id > b.id ? 1 : 0;

/**
 * The catalog: every provider in the directory, every model the quality
 * table or the probe lists for it, sorted by id. The table decides `free`
 * and `context_length` where it has a row; the probe's word fills the gaps.
 * A provider whose probe failed still appears with its table rows, so the
 * catalog never loses a provider to one refused probe.
 */
export function buildCatalog(brief: Brief, probes: readonly Probe[]): Catalog {
  const probeById = new Map(probes.map((p) => [p.provider.id, p]));
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const rowById = new Map<string, RatingRow>();
    for (const row of brief.ratings) {
      if (row.provider === provider.id) rowById.set(row.model, row);
    }
    const probe = probeById.get(provider.id);
    const listed = probe !== undefined && probe.ok ? probe.models : [];
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const add = (id: string, free: boolean, context: number | null): void => {
      if (seen.has(id)) return;
      seen.add(id);
      models.push({ id, free, context_length: context });
    };
    for (const row of [...rowById.values()].sort(byId)) {
      add(row.model, row.free, row.context_length);
    }
    for (const model of [...listed].sort(byId)) {
      const row = rowById.get(model.id);
      if (row !== undefined) continue; // already added from the table
      add(model.id, model.free ?? false, model.context_length);
    }
    models.sort(byId);
    providers.push({
      id: provider.id,
      base_url: provider.base,
      key_env: provider.key_env,
      wire: provider.wire,
      models,
    });
  }
  return { providers };
}
