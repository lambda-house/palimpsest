/**
 * Conversation summarization via Claude API.
 *
 * Compresses old conversation turns into a running summary,
 * keeping recent messages verbatim. The summary is injected
 * as a system message before the recent history.
 *
 * This allows the small llama model to "remember" long conversations
 * without filling its limited context window with raw history.
 */

import Anthropic from "@anthropic-ai/sdk";

const apiKey = process.env.ANTHROPIC_API_KEY;
const client = apiKey ? new Anthropic({ apiKey }) : null;

const SUMMARIZE_MODEL = process.env.SUMMARIZE_MODEL || "claude-haiku-4-5-20251001";

// Keep this many recent messages verbatim (the rest gets summarized)
const KEEP_RECENT = 6;
// Trigger summarization when history exceeds this many messages
const SUMMARIZE_THRESHOLD = 10;

/**
 * Build messages array for llama, with summarized history if needed.
 *
 * @param {Array} history — full conversation history from UserStore
 * @param {string|null} existingSummary — previous summary if any
 * @returns {Promise<{messages: Array, summary: string|null}>}
 */
export async function buildContextWithSummary(history, existingSummary) {
  const hasSystem = history[0]?.role === "system";
  const systemMsg = hasSystem ? history[0] : null;
  const messages = hasSystem ? history.slice(1) : [...history];

  // Not enough history to summarize — return as-is
  if (messages.length <= SUMMARIZE_THRESHOLD || !client) {
    return { messages: history, summary: existingSummary };
  }

  // Split: old messages to summarize, recent to keep verbatim
  const toSummarize = messages.slice(0, -KEEP_RECENT);
  const recent = messages.slice(-KEEP_RECENT);

  // Build text of old messages for summarization
  const oldText = toSummarize
    .map((m) => `${m.role === "user" ? "Пользователь" : "Ассистент"}: ${m.content.slice(0, 500)}`)
    .join("\n\n");

  // Include existing summary for continuity
  const prevContext = existingSummary
    ? `Предыдущее краткое содержание: ${existingSummary}\n\nНовые сообщения:\n${oldText}`
    : oldText;

  try {
    const response = await client.messages.create({
      model: SUMMARIZE_MODEL,
      max_tokens: 512,
      system: `Ты суммаризатор литературного диалога. Сожми предыдущие сообщения в краткое содержание (3-5 предложений), сохраняя:
- Имена персонажей и ключевые сюжетные точки
- Установленный сеттинг и атмосферу
- Незавершённые линии повествования
- Тон и настроение разговора
Пиши кратко, по-русски. Только краткое содержание, без пояснений.`,
      messages: [{ role: "user", content: prevContext }],
    });

    const summary = response.content[0]?.text || existingSummary;

    // Reconstruct messages: system + summary injection + recent
    const result = [];
    if (systemMsg) result.push(systemMsg);
    if (summary) {
      result.push({
        role: "system",
        content: `Краткое содержание предыдущей беседы: ${summary}`,
      });
    }
    result.push(...recent);

    return { messages: result, summary };
  } catch (err) {
    console.error("Summarization failed:", err.message);
    // Fall back to raw history
    return { messages: history, summary: existingSummary };
  }
}
