/**
 * llama.cpp server integration with LoRA adapter support.
 *
 * Supports dual backends (Russian + English) with per-request
 * LoRA adapter selection and mixing. Routes to the correct
 * llama-server based on which adapters are active.
 */

// Dual backend URLs
const LLAMA_URL_RU = process.env.LLAMA_URL_RU || process.env.LLAMA_URL || "http://localhost:8080";
const LLAMA_URL_EN = process.env.LLAMA_URL_EN || null;

function parseAdapterList(raw, lang) {
  if (!raw) return [];
  return raw.split(",").map((entry, index) => {
    const [id, ...rest] = entry.split(":");
    return { id: id.trim(), label: rest.join(":").trim() || id.trim(), index, lang };
  });
}

const adaptersRu = parseAdapterList(process.env.ADAPTERS_RU || process.env.ADAPTERS, "ru");
const adaptersEn = parseAdapterList(process.env.ADAPTERS_EN, "en");

// Combined adapter list (UI sees all of them)
export const adapters = [...adaptersRu, ...adaptersEn];

/**
 * Determine which backend to use based on active adapter weights.
 * Can't mix cross-language — picks the language with highest total weight.
 */
function resolveBackend(adapterWeights) {
  let ruWeight = 0, enWeight = 0;
  for (const [id, scale] of Object.entries(adapterWeights || {})) {
    if (scale <= 0) continue;
    if (adaptersRu.find(a => a.id === id)) ruWeight += scale;
    if (adaptersEn.find(a => a.id === id)) enWeight += scale;
  }

  if (enWeight > 0 && LLAMA_URL_EN) {
    return { url: LLAMA_URL_EN, adapters: adaptersEn, lang: "en" };
  }
  return { url: LLAMA_URL_RU, adapters: adaptersRu, lang: "ru" };
}

export async function generate({ adapterWeights, messages, options, signal, onStatus, onToken, onDone, onError }) {
  try {
    if (onStatus) onStatus("loading");

    const backend = resolveBackend(adapterWeights);

    // Build lora_scaled array — indices are per-backend
    const loraScaled = [];
    if (adapterWeights && backend.adapters.length > 0) {
      for (const [id, scale] of Object.entries(adapterWeights)) {
        const adapter = backend.adapters.find(a => a.id === id);
        if (adapter && scale > 0) {
          loraScaled.push({ id: adapter.index, scale });
        }
      }
    }

    const body = {
      messages,
      stream: true,
      ...(options?.temperature && { temperature: options.temperature }),
      ...(options?.max_tokens && { max_tokens: options.max_tokens }),
      ...(options?.top_p && { top_p: options.top_p }),
      repeat_penalty: options?.repeat_penalty ?? 1.1,
      ...(options?.frequency_penalty && { frequency_penalty: options.frequency_penalty }),
    };

    if (loraScaled.length > 0) {
      body.lora = loraScaled;
    }

    const resp = await fetch(`${backend.url}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });

    if (!resp.ok) {
      const errBody = await resp.text();
      onError(`llama.cpp error: ${resp.status} ${errBody}`);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";
    let firstToken = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") continue;

        try {
          const chunk = JSON.parse(data);
          const content = chunk.choices?.[0]?.delta?.content;
          if (content) {
            if (firstToken) {
              if (onStatus) onStatus("generating");
              firstToken = false;
            }
            fullText += content;
            onToken(content);
          }
        } catch {
          // skip malformed chunks
        }
      }
    }

    onDone(fullText);
  } catch (err) {
    if (err.name === "AbortError") {
      onDone("");
    } else {
      onError(err.message);
    }
  }
}

export async function checkHealth() {
  try {
    const resp = await fetch(`${LLAMA_URL_RU}/health`);
    if (!resp.ok) return false;
    if (LLAMA_URL_EN) {
      const resp2 = await fetch(`${LLAMA_URL_EN}/health`);
      return resp2.ok;
    }
    return true;
  } catch {
    return false;
  }
}
