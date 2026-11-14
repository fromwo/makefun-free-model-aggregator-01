// lib/translate.ts — the three wires, and the answers that cross them.
//
// The entry always calls a provider with a plain JSON request (never a
// stream) and renders the client's answer itself, in JSON or in server-sent
// events, from the provider's one answer. That keeps the usage counts and
// the stop reason exact: the provider's JSON answer carries both, and this
// file copies them into the client's wire instead of relaying an event
// stream whose tail (the usage chunk) a wire may or may not send.
//
// Translations forward the request untouched in meaning: the system prompt,
// the user text (images included), the tool definitions, the stop sequences,
// max_tokens and JSON mode all arrive at the provider exactly as the client
// sent them, because the provider echoes what it parsed and the client
// compares that echo with what it sent. A field the provider's wire cannot
// carry (Anthropic has no JSON mode) is never sent there: the router keeps
// JSON-mode requests off the Anthropic wire.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so a replay renders the same bytes.

import { isRecord, type Json } from './brief.ts';

/** A provider's one answer, in no wire's shape. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** Stop family: 0 stop, 1 length, 2 tool calls, 3 refusal. */
  stopFamily: number;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** How many characters of answer text one streamed delta carries. */
const STREAM_PIECE = 64;

const numOrZero = (v: unknown): number =>
  typeof v === 'number' && Number.isFinite(v) ? v : 0;

/** The text of a string or of a block array's text blocks. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** OpenAI's finish_reason / Anthropic's stop_reason / Gemini's finishReason, as a family. */
function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

const openAIStop = (family: number): string =>
  ['stop', 'length', 'tool_calls', 'content_filter'][family] ?? 'stop';
const anthropicStop = (family: number): string =>
  ['end_turn', 'max_tokens', 'tool_use', 'refusal'][family] ?? 'end_turn';

/* ------------------------------------------------------------------ */
/* Client request -> provider request, one builder per provider wire.  */
/* ------------------------------------------------------------------ */

/** max_tokens, reading either spelling the OpenAI wire allows. */
function openAIMaxTokens(body: Json): number | undefined {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return undefined;
}

/** An OpenAI Chat Completions request that names `model`, never streamed. */
export function openAIToOpenAI(body: Json, model: string): Json {
  const out: Json = { model, messages: body.messages ?? [], stream: false };
  const max = openAIMaxTokens(body);
  if (max !== undefined) out.max_tokens = max;
  if (typeof body.max_completion_tokens === 'number') {
    out.max_completion_tokens = body.max_completion_tokens;
  }
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (body.stop !== undefined) out.stop = body.stop;
  if (Array.isArray(body.tools)) out.tools = body.tools;
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  if (isRecord(body.response_format)) out.response_format = body.response_format;
  return out;
}

/** One Anthropic content block as an OpenAI content part, or null. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic Messages request as an OpenAI Chat Completions request. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages, stream: false };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    out.stop = body.stop_sequences;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  return out;
}

/** An OpenAI content part as an Anthropic content block, or null. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') {
    return { type: 'text', text: part.text };
  }
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return {
        type: 'image',
        source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] },
      };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      const content = raw.content;
      const text = textOf(content);
      const blocks: Json[] = [];
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray((raw as Json).tool_calls)) {
        for (const call of (raw as Json).tool_calls as unknown[]) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) input = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          } else if (isRecord(call.function.arguments)) {
            input = call.function.arguments;
          }
          toolCalls.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input,
          });
        }
      }
      messages.push({ role: 'assistant', content: [...blocks, ...toolCalls] });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    // user (or anything else): content parts.
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks = content.flatMap((p) => {
        if (!isRecord(p)) return [];
        const block = openAIPartToBlock(p);
        return block === null ? [] : [block];
      });
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages, stream: false };
  if (system !== '') out.system = system;
  const max = openAIMaxTokens(body);
  if (max !== undefined) out.max_tokens = max;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters)
            ? raw.function.parameters
            : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  // Anthropic carries no JSON mode: the router never sends a JSON-mode
  // request here, so there is nothing to translate.
  return out;
}

/** An Anthropic Messages request that names `model`, never streamed. */
export function anthropicToAnthropic(body: Json, model: string): Json {
  const out: Json = { model, messages: body.messages ?? [], stream: false };
  if (body.system !== undefined) out.system = body.system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences)) out.stop_sequences = body.stop_sequences;
  if (Array.isArray(body.tools)) out.tools = body.tools;
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  return out;
}

/* ------------------------------------------------------------------ */
/* Gemini: the third wire.                                             */
/* ------------------------------------------------------------------ */

/** One message's content as Gemini parts. */
function partsOf(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ text: block.text });
    } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) {
        parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      } else {
        parts.push({ fileData: { fileUri: url } });
      }
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({
          inlineData: {
            mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
            data: source.data,
          },
        });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { fileUri: source.url } });
      }
    } else if (block.type === 'tool_use') {
      parts.push({
        functionCall: {
          name: typeof block.name === 'string' ? block.name : '',
          args: isRecord(block.input) ? block.input : {},
        },
      });
    } else if (block.type === 'tool_result') {
      parts.push({
        functionResponse: {
          name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          response: { result: toolResultText(block.content) },
        },
      });
    }
  }
  return parts;
}

/** Tool definitions, in either client wire's spelling, as Gemini declarations. */
function geminiDeclarations(tools: unknown): Json[] | null {
  if (!Array.isArray(tools)) return null;
  const declarations: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw)) continue;
    if (isRecord(raw.function) && typeof raw.function.name === 'string') {
      declarations.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    } else if (typeof raw.name === 'string') {
      declarations.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return declarations.length > 0 ? declarations : null;
}

/** Any client request (OpenAI or Anthropic shape) as a Gemini generateContent body. */
export function toGemini(body: Json, clientApi: 'openai' | 'anthropic'): Json {
  const contents: Json[] = [];
  let systemText = '';
  if (clientApi === 'anthropic') {
    systemText = textOf(body.system);
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      const parts = partsOf(raw.content);
      if (parts.length === 0) continue;
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
  } else {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
        continue;
      }
      if (raw.role === 'tool') {
        contents.push({
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
                response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) },
              },
            },
          ],
        });
        continue;
      }
      const parts = partsOf(raw.content);
      if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls as unknown[]) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let args: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) args = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          parts.push({
            functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args },
          });
        }
      }
      if (parts.length === 0) continue;
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = geminiDeclarations(body.tools);
  if (declarations !== null) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const max =
    clientApi === 'anthropic'
      ? typeof body.max_tokens === 'number'
        ? body.max_tokens
        : undefined
      : openAIMaxTokens(body);
  if (max !== undefined) config.maxOutputTokens = max;
  const stops =
    clientApi === 'anthropic'
      ? Array.isArray(body.stop_sequences)
        ? body.stop_sequences
        : []
      : Array.isArray(body.stop)
        ? body.stop
        : typeof body.stop === 'string'
          ? [body.stop]
          : [];
  if (stops.length > 0) config.stopSequences = stops;
  if (clientApi === 'openai' && isRecord(body.response_format)) {
    const format = body.response_format.type;
    if (format === 'json_object' || format === 'json_schema') {
      config.responseMimeType = 'application/json';
    }
  }
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/* ------------------------------------------------------------------ */
/* Provider answers -> one normalized answer.                          */
/* ------------------------------------------------------------------ */

/** An OpenAI Chat Completions JSON answer, normalized. */
export function parseOpenAIAnswer(body: Json): NormalizedAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numOrZero(usage.prompt_tokens),
    outputTokens: numOrZero(usage.completion_tokens),
    stopFamily: stopFamilyOf(choice.finish_reason),
  };
}

/** An Anthropic Messages JSON answer, normalized. */
export function parseAnthropicAnswer(body: Json): NormalizedAnswer | null {
  if (!Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    inputTokens: numOrZero(usage.input_tokens),
    outputTokens: numOrZero(usage.output_tokens),
    stopFamily: stopFamilyOf(body.stop_reason),
  };
}

/** A Gemini generateContent JSON answer, normalized. */
export function parseGeminiAnswer(body: Json): NormalizedAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('\n');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numOrZero(usage.promptTokenCount),
    outputTokens: numOrZero(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(candidate.finishReason),
  };
}

/* ------------------------------------------------------------------ */
/* One normalized answer -> the client's wire, JSON or SSE.            */
/* ------------------------------------------------------------------ */

/** A Chat Completions JSON object. */
export function openAIJsonAnswer(model: string, answer: NormalizedAnswer): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStop(answer.stopFamily),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A Messages JSON object. */
export function anthropicJsonAnswer(model: string, answer: NormalizedAnswer): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE) pieces.push(text.slice(at, at + STREAM_PIECE));
  return pieces.length === 0 ? [''] : pieces;
}

/** A Chat Completions event stream: role, deltas, finish, usage, [DONE]. */
export function openAIStreamAnswer(model: string, answer: NormalizedAnswer): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent(null, chunk({ content: piece }, null)));
  }
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stopFamily))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A Messages event stream: start, one text block, delta, stop. */
export function anthropicStreamAnswer(model: string, answer: NormalizedAnswer): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/* ------------------------------------------------------------------ */
/* Errors and model lists, in the client's wire.                       */
/* ------------------------------------------------------------------ */

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
}

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: m.provider,
      free: true,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.id,
      created_at: FIXED_CREATED_AT,
      free: true,
    })),
    has_more: false,
  };
}
