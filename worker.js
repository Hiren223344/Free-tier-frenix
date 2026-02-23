/**
 * OpenAI-Compatible Bridge — Cloudflare Worker
 * Provider: api.evolvex.gg
 *
 * Deploy: paste this file into Cloudflare Workers dashboard editor
 *
 * Endpoints:
 *   GET  /v1/models
 *   GET  /v1/models/:model
 *   POST /v1/chat/completions   (streaming + JSON mode)
 *   POST /v1/completions        (legacy)
 *   POST /v1/responses          (OpenAI Responses API)
 *   POST /v1/images/generations
 *   POST /v1/embeddings
 *   POST /v1/moderations        (stub fallback)
 *   GET  /v1/                   (endpoint discovery)
 *   GET  /health
 *   GET  /ping
 *   GET  /stats
 */

// ── Config ────────────────────────────────────────────────────────────────────

const PROVIDER_BASE = "https://api.evolvex.gg/v1";
const FALLBACK_KEY  = env.PROVIDER_API_KEY || "evx-sk-1ddbea7a2d47d0b89c78840f0924e94592bb07f4c9a93734";

const DUMMY_TOKENS = new Set([
  "any-key", "dummy", "placeholder", "none", "sk-xxx", "",
  "your_real_api_key", "bearer", "token",
]);

// In-memory stats — resets when isolate recycles (~few hours on free plan)
const STATS = {
  started_at:   Date.now(),
  requests:     0,
  errors:       0,
  tokens:       0,
  latency_ms:   0,
  by_endpoint:  {},
};

function recordStats(endpoint, { tokens = 0, latency = 0, error = false } = {}) {
  STATS.requests  += 1;
  STATS.latency_ms += latency;
  STATS.tokens    += tokens;
  if (error) STATS.errors += 1;
  const ep = STATS.by_endpoint[endpoint] ||= { requests: 0, errors: 0, tokens: 0, latency_ms: 0 };
  ep.requests  += 1;
  ep.latency_ms += latency;
  ep.tokens    += tokens;
  if (error) ep.errors += 1;
}

// ── Header helpers ────────────────────────────────────────────────────────────

function getProviderHeaders(req) {
  const auth  = req.headers.get("Authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
  const key   = (!token || DUMMY_TOKENS.has(token.toLowerCase())) ? FALLBACK_KEY : token;
  return {
    "Authorization": `Bearer ${key}`,
    "Content-Type":  "application/json",
  };
}

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
  };
}

// ── Response helpers ──────────────────────────────────────────────────────────

function jsonResp(data, status = 200, extra = {}) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders(), ...extra },
  });
}

function oaiError(message, type = "invalid_request_error", code = null, status = 400, param = null) {
  recordStats("error", { error: true });
  return jsonResp({ error: { message, type, param, code } }, status);
}

function providerError(err) {
  const status = err.status || 502;
  return oaiError(err.message || "Provider error", "provider_error", null, status);
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────

async function providerFetch(url, options, maxRetries = 3) {
  let delay = 1000;
  const retryStatuses = new Set([429, 500, 502, 503, 504]);

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (e) {
      if (attempt === maxRetries) throw { status: 502, message: `Connection error: ${e.message}` };
      await sleep(delay); delay *= 2; continue;
    }

    if (retryStatuses.has(res.status) && attempt < maxRetries && options.method !== "stream") {
      const retryAfter = parseInt(res.headers.get("Retry-After") || "0") * 1000 || delay;
      await sleep(retryAfter); delay *= 2; continue;
    }

    if (!res.ok) {
      let msg = `Provider error ${res.status}`;
      try { const d = await res.json(); msg = d?.error?.message || d?.message || msg; } catch {}
      throw { status: res.status, message: msg };
    }

    return res;
  }
}

async function providerPost(url, body, headers, stream = false) {
  return providerFetch(url, {
    method:  "POST",
    headers,
    body:    JSON.stringify(body),
    ...(stream ? {} : {}),
  });
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ── Validation ────────────────────────────────────────────────────────────────

const VALID_ROLES = new Set(["system", "user", "assistant", "tool", "function"]);

function validateMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0)
    return "Field 'messages' must be a non-empty array.";
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (!m || typeof m !== "object") return `messages[${i}] must be an object.`;
    if (!m.role) return `messages[${i}] is missing required field 'role'.`;
    if (!VALID_ROLES.has(m.role)) return `messages[${i}].role '${m.role}' is not valid. Must be one of: ${[...VALID_ROLES].sort().join(", ")}.`;
    if (m.content == null && !m.tool_calls) return `messages[${i}] must have 'content' or 'tool_calls'.`;
  }
  return null;
}

function validateNumeric(body, param, lo, hi) {
  if (!(param in body)) return null;
  const v = body[param];
  if (typeof v !== "number") return `'${param}' must be a number.`;
  if (v < lo || v > hi) return `'${param}' must be between ${lo} and ${hi}.`;
  return null;
}

function validateResponseFormat(rf) {
  if (!rf) return null;
  if (typeof rf !== "object") return "'response_format' must be an object.";
  const allowed = new Set(["text", "json_object", "json_schema"]);
  if (!allowed.has(rf.type))
    return `'response_format.type' must be one of: text, json_object, json_schema. Got: '${rf.type}'.`;
  if (rf.type === "json_schema") {
    if (!rf.json_schema) return "'response_format.json_schema' is required when type is 'json_schema'.";
    if (!rf.json_schema.name) return "'response_format.json_schema.name' is required.";
  }
  return null;
}

// ── JSON mode ──────────────────────────────────────────────────────────────────

function injectJsonSystemMsg(messages, rf) {
  if (!rf || !["json_object", "json_schema"].includes(rf.type)) return messages;
  let instr = "You must respond with valid JSON only. No explanation, markdown, or code fences.";
  if (rf.type === "json_schema" && rf.json_schema?.schema)
    instr += ` Conform to this JSON schema: ${JSON.stringify(rf.json_schema.schema)}`;
  const hasJsonSystem = messages.some(m => m.role === "system" && (m.content || "").toLowerCase().includes("json"));
  if (hasJsonSystem) return messages;
  const hasSystem = messages.some(m => m.role === "system");
  if (hasSystem) return messages.map(m => m.role === "system" ? { ...m, content: m.content + "\n\n" + instr } : m);
  return [{ role: "system", content: instr }, ...messages];
}

function handleJsonMode(body, data) {
  const rf = body?.response_format;
  if (!rf || !["json_object", "json_schema"].includes(rf.type)) return data;
  for (const choice of (data.choices || [])) {
    const raw = choice?.message?.content || "";
    const stripped = raw.trim().replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "").trim();
    try {
      JSON.parse(stripped);
      choice.message.content = stripped;
    } catch {
      choice.message.content = JSON.stringify({ error: "Provider did not return valid JSON.", raw });
    }
  }
  return data;
}

// ── /v1/models ────────────────────────────────────────────────────────────────

async function handleModels(req, modelId) {
  const headers = getProviderHeaders(req);
  const url     = modelId ? `${PROVIDER_BASE}/models/${modelId}` : `${PROVIDER_BASE}/models`;
  const t0      = Date.now();
  try {
    const res  = await providerFetch(url, { method: "GET", headers });
    const data = await res.json();
    recordStats("/v1/models", { latency: Date.now() - t0 });
    return jsonResp(data);
  } catch (e) {
    recordStats("/v1/models", { error: true, latency: Date.now() - t0 });
    if (modelId) return oaiError(`The model '${modelId}' does not exist.`, "invalid_request_error", "model_not_found", 404);
    return jsonResp({ object: "list", data: [] });
  }
}

// ── /v1/chat/completions ──────────────────────────────────────────────────────

async function handleChatCompletions(req) {
  let body;
  try { body = await req.json(); } catch { return oaiError("Could not parse request body as JSON."); }

  const msgErr = validateMessages(body.messages);
  if (msgErr) return oaiError(msgErr, "invalid_request_error", null, 400, "messages");

  for (const [param, lo, hi] of [["temperature",0,2],["top_p",0,1],["presence_penalty",-2,2],["frequency_penalty",-2,2],["n",1,128]]) {
    const err = validateNumeric(body, param, lo, hi);
    if (err) return oaiError(err, "invalid_request_error", null, 400, param);
  }

  const rfErr = validateResponseFormat(body.response_format);
  if (rfErr) return oaiError(rfErr, "invalid_request_error", null, 400, "response_format");

  const model    = body.model || "gpt-4";
  const stream   = !!body.stream;
  const messages = injectJsonSystemMsg(body.messages, body.response_format);
  const headers  = getProviderHeaders(req);
  const payload  = { ...body, messages };
  const t0       = Date.now();

  try {
    const res = await providerPost(`${PROVIDER_BASE}/chat/completions`, payload, headers, stream);

    if (stream) {
      recordStats("/v1/chat/completions", { latency: Date.now() - t0 });
      return new Response(res.body, {
        headers: {
          "Content-Type":  "text/event-stream",
          "Cache-Control": "no-cache",
          ...corsHeaders(),
        },
      });
    }

    let data = await res.json();
    data = handleJsonMode(body, data);
    const tokens = data.usage?.total_tokens || 0;
    recordStats("/v1/chat/completions", { tokens, latency: Date.now() - t0 });
    return jsonResp(data);
  } catch (e) {
    recordStats("/v1/chat/completions", { error: true, latency: Date.now() - t0 });
    return providerError(e);
  }
}

// ── /v1/completions (legacy) ──────────────────────────────────────────────────

async function handleLegacyCompletions(req) {
  let body;
  try { body = await req.json(); } catch { return oaiError("Could not parse request body as JSON."); }
  if (!body.prompt) return oaiError("Missing required field: prompt.", "invalid_request_error", null, 400, "prompt");

  const model   = body.model || "gpt-3.5-turbo";
  const stream  = !!body.stream;
  const prompt  = typeof body.prompt === "string" ? body.prompt : body.prompt.join("\n");
  const messages = [{ role: "user", content: prompt }];
  const extra   = {};
  for (const k of ["temperature","top_p","n","max_tokens","stop","presence_penalty","frequency_penalty","user","seed"])
    if (k in body) extra[k] = body[k];

  const headers = getProviderHeaders(req);
  const t0      = Date.now();

  try {
    const res = await providerPost(`${PROVIDER_BASE}/chat/completions`, { model, messages, stream, ...extra }, headers, stream);

    if (stream) {
      // Transform chat chunks → legacy completions chunks
      const { readable, writable } = new TransformStream();
      const writer = writable.getWriter();
      const encoder = new TextEncoder();
      const cmplId  = "cmpl-" + crypto.randomUUID().replace(/-/g, "");
      const created = Math.floor(Date.now() / 1000);

      (async () => {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const lines = decoder.decode(value).split("\n");
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (raw === "[DONE]") { await writer.write(encoder.encode("data: [DONE]\n\n")); break; }
            try {
              const chunk = JSON.parse(raw);
              const delta = chunk.choices?.[0]?.delta || {};
              const text  = delta.content || "";
              const out   = { id: cmplId, object: "text_completion", created, model,
                              choices: [{ text, index: 0, logprobs: null, finish_reason: chunk.choices?.[0]?.finish_reason || null }] };
              await writer.write(encoder.encode(`data: ${JSON.stringify(out)}\n\n`));
            } catch {}
          }
        }
        await writer.close();
      })();

      recordStats("/v1/completions", { latency: Date.now() - t0 });
      return new Response(readable, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", ...corsHeaders() } });
    }

    const chat = await res.json();
    const text = chat.choices?.[0]?.message?.content || "";
    const usage = chat.usage || {};
    recordStats("/v1/completions", { tokens: usage.total_tokens || 0, latency: Date.now() - t0 });
    return jsonResp({
      id:      chat.id || "cmpl-" + crypto.randomUUID().replace(/-/g,""),
      object:  "text_completion",
      created: chat.created || Math.floor(Date.now()/1000),
      model:   chat.model || model,
      choices: [{ text, index: 0, logprobs: null, finish_reason: chat.choices?.[0]?.finish_reason || "stop" }],
      usage,
    });
  } catch (e) {
    recordStats("/v1/completions", { error: true, latency: Date.now() - t0 });
    return providerError(e);
  }
}

// ── /v1/responses ─────────────────────────────────────────────────────────────

async function handleResponses(req) {
  let body;
  try { body = await req.json(); } catch { return oaiError("Could not parse request body as JSON."); }
  if (!("input" in body)) return oaiError("Missing required field: input.", "invalid_request_error", null, 400, "input");

  const model   = body.model || "gpt-4";
  const stream  = !!body.stream;
  const headers = getProviderHeaders(req);

  const messages = [];
  if (body.instructions) messages.push({ role: "system", content: body.instructions });

  const inp = body.input;
  if (typeof inp === "string") {
    messages.push({ role: "user", content: inp });
  } else if (Array.isArray(inp)) {
    for (const item of inp) {
      if (typeof item === "string") messages.push({ role: "user", content: item });
      else if (item && typeof item === "object") messages.push({ role: item.role || "user", content: item.content || "" });
    }
  }

  for (const prev of (body.conversation || []))
    messages.push({ role: prev.role || "user", content: prev.content || "" });

  const extra = {};
  if ("max_output_tokens" in body) extra.max_tokens = body.max_output_tokens;
  for (const k of ["temperature","top_p","stop"]) if (k in body) extra[k] = body[k];

  const t0 = Date.now();

  try {
    const res = await providerPost(`${PROVIDER_BASE}/chat/completions`, { model, messages, stream, ...extra }, headers, stream);

    const respId  = "resp_" + crypto.randomUUID().replace(/-/g,"");
    const itemId  = "msg_"  + crypto.randomUUID().replace(/-/g,"");
    const created = Math.floor(Date.now()/1000);

    if (stream) {
      const { readable, writable } = new TransformStream();
      const writer  = writable.getWriter();
      const encoder = new TextEncoder();

      const ev = d => encoder.encode(`data: ${JSON.stringify(d)}\n\n`);

      (async () => {
        await writer.write(ev({ type: "response.created", response: { id: respId, object: "response", created_at: created, model, status: "in_progress", output: [] } }));
        await writer.write(ev({ type: "response.output_item.added", output_index: 0, item: { id: itemId, type: "message", role: "assistant", content: [], status: "in_progress" } }));
        await writer.write(ev({ type: "response.content_part.added", item_id: itemId, output_index: 0, content_index: 0, part: { type: "output_text", text: "" } }));

        let fullText = "";
        const reader  = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const lines = decoder.decode(value).split("\n");
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (raw === "[DONE]") break;
            try {
              const chunk = JSON.parse(raw);
              const text  = chunk.choices?.[0]?.delta?.content || "";
              if (text) {
                fullText += text;
                await writer.write(ev({ type: "response.output_text.delta", item_id: itemId, output_index: 0, content_index: 0, delta: text }));
              }
            } catch {}
          }
        }

        await writer.write(ev({ type: "response.content_part.done", item_id: itemId, output_index: 0, content_index: 0, part: { type: "output_text", text: fullText } }));
        await writer.write(ev({ type: "response.output_item.done", output_index: 0, item: { id: itemId, type: "message", role: "assistant", content: [{ type: "output_text", text: fullText }], status: "completed" } }));
        await writer.write(ev({ type: "response.done", response: { id: respId, object: "response", created_at: created, model, status: "completed", output: [{ id: itemId, type: "message", role: "assistant", content: [{ type: "output_text", text: fullText }], status: "completed" }] } }));
        await writer.write(encoder.encode("data: [DONE]\n\n"));
        await writer.close();
      })();

      recordStats("/v1/responses", { latency: Date.now() - t0 });
      return new Response(readable, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", ...corsHeaders() } });
    }

    const chat    = await res.json();
    const content = chat.choices?.[0]?.message?.content || "";
    const usage   = chat.usage || {};
    recordStats("/v1/responses", { tokens: usage.total_tokens || 0, latency: Date.now() - t0 });
    return jsonResp({
      id: respId, object: "response", created_at: created, model: chat.model || model,
      status: "completed",
      output: [{ id: itemId, type: "message", role: "assistant", content: [{ type: "output_text", text: content }], status: "completed" }],
      usage: { input_tokens: usage.prompt_tokens || 0, output_tokens: usage.completion_tokens || 0, total_tokens: usage.total_tokens || 0 },
      finish_reason: chat.choices?.[0]?.finish_reason || "stop",
    });
  } catch (e) {
    recordStats("/v1/responses", { error: true, latency: Date.now() - t0 });
    return providerError(e);
  }
}

// ── /v1/images/generations ────────────────────────────────────────────────────

async function handleImageGenerations(req) {
  let body;
  try { body = await req.json(); } catch { return oaiError("Could not parse request body as JSON."); }
  if (!body.prompt) return oaiError("Missing required field: prompt.", "invalid_request_error", null, 400, "prompt");

  const t0 = Date.now();
  try {
    const res  = await providerPost(`${PROVIDER_BASE}/images/generations`, body, getProviderHeaders(req));
    const data = await res.json();
    recordStats("/v1/images/generations", { latency: Date.now() - t0 });
    return jsonResp(data);
  } catch (e) {
    recordStats("/v1/images/generations", { error: true, latency: Date.now() - t0 });
    return providerError(e);
  }
}

// ── /v1/embeddings ────────────────────────────────────────────────────────────

async function handleEmbeddings(req) {
  let body;
  try { body = await req.json(); } catch { return oaiError("Could not parse request body as JSON."); }
  if (!("input" in body)) return oaiError("Missing required field: input.", "invalid_request_error", null, 400, "input");

  const t0 = Date.now();
  try {
    const res  = await providerPost(`${PROVIDER_BASE}/embeddings`, body, getProviderHeaders(req));
    const data = await res.json();
    recordStats("/v1/embeddings", { tokens: data.usage?.total_tokens || 0, latency: Date.now() - t0 });
    return jsonResp(data);
  } catch (e) {
    recordStats("/v1/embeddings", { error: true, latency: Date.now() - t0 });
    return oaiError("Embeddings are not supported by this provider.", "invalid_request_error", "unsupported_endpoint", 400);
  }
}

// ── /v1/moderations ──────────────────────────────────────────────────────────

async function handleModerations(req) {
  let body;
  try { body = await req.json(); } catch { return oaiError("Could not parse request body as JSON."); }
  if (!("input" in body)) return oaiError("Missing required field: input.", "invalid_request_error", null, 400, "input");

  try {
    const res  = await providerPost(`${PROVIDER_BASE}/moderations`, body, getProviderHeaders(req));
    return jsonResp(await res.json());
  } catch {}

  const inputs = Array.isArray(body.input) ? body.input : [body.input];
  const cats   = ["sexual","hate","harassment","self-harm","sexual/minors","hate/threatening","violence/graphic","self-harm/intent","self-harm/instructions","harassment/threatening","violence"];
  return jsonResp({
    id:      "modr-" + crypto.randomUUID().replace(/-/g,""),
    model:   body.model || "text-moderation-latest",
    results: inputs.map(() => ({ flagged: false, categories: Object.fromEntries(cats.map(k=>[k,false])), category_scores: Object.fromEntries(cats.map(k=>[k,0.0])) })),
  });
}

// ── /stats ────────────────────────────────────────────────────────────────────

function handleStats() {
  const uptime = Date.now() - STATS.started_at;
  const byEp   = {};
  for (const [ep, d] of Object.entries(STATS.by_endpoint)) {
    byEp[ep] = {
      ...d,
      avg_latency_ms: d.requests ? Math.round(d.latency_ms / d.requests) : 0,
      error_rate:     d.requests ? +(d.errors / d.requests).toFixed(4) : 0,
    };
  }
  return jsonResp({
    uptime_ms:       uptime,
    uptime_s:        Math.floor(uptime / 1000),
    total_requests:  STATS.requests,
    total_errors:    STATS.errors,
    error_rate:      STATS.requests ? +(STATS.errors / STATS.requests).toFixed(4) : 0,
    total_tokens:    STATS.tokens,
    avg_latency_ms:  STATS.requests ? Math.round(STATS.latency_ms / STATS.requests) : 0,
    by_endpoint:     byEp,
  });
}

// ── /v1/ discovery ─────────────────────────────────────────────────────────────

function handleDiscovery() {
  return jsonResp({
    name:     "OpenAI-compatible bridge (Cloudflare Worker)",
    version:  "1.0.0",
    provider: PROVIDER_BASE,
    runtime:  "cloudflare-workers",
    notes: {
      auth:      "Pass your real API key as 'Authorization: Bearer <key>'. Dummy values fall back to provider default.",
      json_mode: "response_format={type:'json_object'} and 'json_schema' fully supported with system message injection and response post-processing.",
      streaming: "True streaming — provider SSE is piped directly to client via ReadableStream.",
    },
    endpoints: [
      { method:"GET",  path:"/v1/models",             description:"List models from provider." },
      { method:"GET",  path:"/v1/models/:model",       description:"Retrieve a specific model." },
      { method:"POST", path:"/v1/chat/completions",    description:"Chat completion. Supports streaming, JSON mode, tools." },
      { method:"POST", path:"/v1/completions",         description:"Legacy text completions (converted to chat internally)." },
      { method:"POST", path:"/v1/responses",           description:"OpenAI Responses API (converted to chat internally)." },
      { method:"POST", path:"/v1/images/generations",  description:"Image generation." },
      { method:"POST", path:"/v1/embeddings",          description:"Text embeddings." },
      { method:"POST", path:"/v1/moderations",         description:"Content moderation (stub fallback)." },
      { method:"GET",  path:"/stats",                  description:"Bridge usage stats." },
      { method:"GET",  path:"/health",                 description:"Health check." },
      { method:"GET",  path:"/ping",                   description:"Liveness probe." },
    ],
  });
}

// ── /health ───────────────────────────────────────────────────────────────────

async function handleHealth() {
  let providerOk  = false;
  let providerMs  = null;
  const t0 = Date.now();
  try {
    const res = await fetch(`${PROVIDER_BASE}/models`, {
      headers: { "Authorization": `Bearer ${FALLBACK_KEY}` },
      signal: AbortSignal.timeout(5000),
    });
    providerOk = res.ok;
    providerMs = Date.now() - t0;
  } catch {}

  return jsonResp({
    status:    providerOk ? "healthy" : "degraded",
    uptime_s:  Math.floor((Date.now() - STATS.started_at) / 1000),
    requests:  STATS.requests,
    provider:  { reachable: providerOk, latency_ms: providerMs, url: PROVIDER_BASE },
    runtime:   "cloudflare-workers",
    timestamp: Math.floor(Date.now() / 1000),
  }, providerOk ? 200 : 207);
}

// ── Router ────────────────────────────────────────────────────────────────────

export default {
  async fetch(request, env, ctx) {
    STATS.requests++;
    const FALLBACK_KEY = env.PROVIDER_API_KEY || "evx-sk-1ddbea7a2d47d0b89c78840f0924e94592bb07f4c9a93734";

    const url    = new URL(request.url);
    const path   = url.pathname;
    const method = request.method.toUpperCase();

    // CORS preflight
    if (method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    // ── GET routes ──
    if (method === "GET") {
      if (path === "/" || path === "/health")        return handleHealth();
      if (path === "/ping")                          return jsonResp({ pong: true, timestamp: Math.floor(Date.now()/1000) });
      if (path === "/stats")                         return handleStats();
      if (path === "/v1/" || path === "/v1")         return handleDiscovery();
      if (path === "/v1/models")                     return handleModels(request, null);
      if (path.startsWith("/v1/models/"))            return handleModels(request, decodeURIComponent(path.slice("/v1/models/".length)));
    }

    // ── POST routes ──
    if (method === "POST") {
      if (path === "/v1/chat/completions")                      return handleChatCompletions(request);
      if (path === "/v1/completions")                           return handleLegacyCompletions(request);
      if (path === "/v1/responses")                             return handleResponses(request);
      if (path === "/v1/images/generations" || path === "/v1/image/generations") return handleImageGenerations(request);
      if (path === "/v1/embeddings")                            return handleEmbeddings(request);
      if (path === "/v1/moderations")                           return handleModerations(request);
    }

    // 404
    return jsonResp({
      error: { message: `Route not found: ${method} ${path}`, type: "invalid_request_error", code: "route_not_found", param: null }
    }, 404);
  },
};
