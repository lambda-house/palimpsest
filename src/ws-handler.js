import { generate, adapters } from "./llama.js";
import { enhance } from "./enhance.js";
import { buildContextWithSummary } from "./summarize.js";

// Per-user conversation summaries (in-memory, lost on restart)
const userSummaries = new Map();

export function handleConnection(ws, queue, userStore) {
  let userId = null;
  let currentCancel = null;

  ws.on("message", (data) => {
    let msg;
    try {
      msg = JSON.parse(data);
    } catch {
      ws.send(JSON.stringify({ type: "error", text: "invalid json" }));
      return;
    }

    if (msg.type === "hello") {
      const user = userStore.getOrCreate(msg.userId || null, msg.systemPrompt);
      userId = user.userId;
      const history = user.history
        .filter((m) => m.role !== "system")
        .map(({ role, content }) => ({ role, content }));
      const systemPrompt =
        user.history[0]?.role === "system" ? user.history[0].content : null;
      ws.send(JSON.stringify({
        type: "session",
        userId,
        adapters,
        systemPrompt,
        history,
      }));
      return;
    }

    if (msg.type === "clear") {
      if (userId) {
        userStore.clearHistory(userId, msg.systemPrompt);
        userSummaries.delete(userId);
      }
      ws.send(JSON.stringify({ type: "cleared" }));
      return;
    }

    if (msg.type === "abort") {
      if (currentCancel) {
        currentCancel();
        currentCancel = null;
      }
      return;
    }

    if (msg.type !== "chat") return;

    if (currentCancel) {
      currentCancel();
      currentCancel = null;
    }

    if (!userId) {
      const user = userStore.getOrCreate(null, msg.systemPrompt);
      userId = user.userId;
      ws.send(JSON.stringify({ type: "session", userId }));
    }

    const text = msg.text;
    if (!text) return;

    // adapterWeights: { "pelevin": 0.8, "lovecraft": 0.2 }
    const adapterWeights = msg.adapterWeights || {};
    const options = msg.options || undefined;

    userStore.appendMessage(userId, "user", text);

    // Get system prompt from history for enhance context
    const history = userStore.getHistory(userId);
    const systemPrompt =
      history[0]?.role === "system" ? history[0].content : null;

    const { position, cancel } = queue.enqueue({
      onPosition(pos) {
        if (ws.readyState === ws.OPEN) {
          ws.send(JSON.stringify({ type: "queue", position: pos }));
        }
      },
      async execute(signal) {
        // Enhance prompt via Claude before sending to llama
        let enhancedText = text;
        try {
          if (ws.readyState === ws.OPEN) {
            ws.send(JSON.stringify({ type: "status", status: "enhancing" }));
          }
          enhancedText = await enhance({
            userPrompt: text,
            adapterWeights,
            adapterList: adapters,
            systemPrompt,
          });
        } catch (err) {
          console.error("Enhancement failed, using original prompt:", err.message);
        }

        // Build messages with summarized history + enhanced prompt
        const rawHistory = userStore.getHistory(userId);
        const existingSummary = userSummaries.get(userId) || null;
        const { messages: summarizedMessages, summary: newSummary } =
          await buildContextWithSummary(rawHistory, existingSummary);
        if (newSummary) userSummaries.set(userId, newSummary);

        // Replace last user message with enhanced version
        const llamaMessages = summarizedMessages.map((m, i) =>
          i === summarizedMessages.length - 1 && m.role === "user"
            ? { ...m, content: enhancedText }
            : m
        );

        await generate({
          adapterWeights,
          messages: llamaMessages,
          options,
          signal,
          onStatus(status) {
            if (ws.readyState === ws.OPEN) {
              ws.send(JSON.stringify({ type: "status", status }));
            }
          },
          onToken(token) {
            if (ws.readyState === ws.OPEN) {
              ws.send(JSON.stringify({ type: "token", text: token }));
            }
          },
          onDone(fullText) {
            if (fullText) {
              userStore.appendMessage(userId, "assistant", fullText);
            }
            if (ws.readyState === ws.OPEN) {
              ws.send(JSON.stringify({
                type: "done",
                meta: {
                  adapters: adapterWeights,
                  options: { temperature: options?.temperature, max_tokens: options?.max_tokens },
                  enhanced: enhancedText !== text ? enhancedText : undefined,
                  timestamp: new Date().toISOString(),
                },
              }));
            }
          },
          onError(errText) {
            if (ws.readyState === ws.OPEN) {
              ws.send(JSON.stringify({ type: "error", text: errText }));
            }
          },
        });
      },
    });

    currentCancel = cancel;

    if (position > 0) {
      ws.send(JSON.stringify({ type: "queue", position }));
    }
  });

  ws.on("close", () => {
    if (currentCancel) {
      currentCancel();
      currentCancel = null;
    }
  });
}
