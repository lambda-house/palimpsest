/**
 * Prompt enhancement via Claude API.
 *
 * Takes the user's raw prompt and LoRA adapter mix, asks Opus to craft
 * an optimal prompt that will get the best literary output from the small
 * fine-tuned model. Passthrough if ANTHROPIC_API_KEY is not set.
 */

import Anthropic from "@anthropic-ai/sdk";

const apiKey = process.env.ANTHROPIC_API_KEY;
const client = apiKey ? new Anthropic({ apiKey }) : null;

const ENHANCE_MODEL = process.env.ENHANCE_MODEL || "claude-sonnet-4-20250514";

/**
 * @param {string} userPrompt — the user's original message
 * @param {object} adapterWeights — e.g. { pelevin: 0.7, lovecraft: 0.3 }
 * @param {Array} adapterList — full adapter list with { id, label }
 * @param {string} systemPrompt — the current system prompt for context
 * @returns {Promise<string>} — enhanced prompt, or original if no API key
 */
export async function enhance({ userPrompt, adapterWeights, adapterList, systemPrompt }) {
  if (!client) return userPrompt;

  const active = Object.entries(adapterWeights)
    .filter(([, v]) => v > 0)
    .map(([id, v]) => {
      const adapter = adapterList.find((a) => a.id === id);
      return { id, label: adapter?.label || id, weight: v };
    });

  if (active.length === 0) return userPrompt;

  const total = active.reduce((s, a) => s + a.weight, 0);
  const mix = active
    .map((a) => `${a.label} (${a.id}) — ${Math.round((a.weight / total) * 100)}%`)
    .join(", ");

  const response = await client.messages.create({
    model: ENHANCE_MODEL,
    max_tokens: 1024,
    system: `You are a literary prompt engineer. Your job is to take a user's creative writing request and rewrite it into an optimal prompt for a small (12B parameter) Russian-language literary model that has LoRA adapters fine-tuned on specific authors.

The model will receive your rewritten prompt as the user message. The LoRA adapter mix is applied at inference time — you do NOT need to instruct the model to "write like author X". Instead, focus on:

1. Enriching the creative brief with vivid specifics: setting details, mood, sensory anchors, narrative tension
2. Priming the themes, motifs, and narrative devices characteristic of the authors in the mix — without naming them
3. Suggesting a narrative structure or opening direction that plays to the strengths of the style blend
4. Keeping instructions in Russian, matching the language of the literary model
5. Being concise — the model has a 4096 token context, so the prompt should leave room for generation

The current system prompt is: "${systemPrompt || "Ты — писатель художественной прозы."}"

Output ONLY the rewritten prompt in Russian. No explanations, no meta-commentary.`,
    messages: [
      {
        role: "user",
        content: `Author mix: ${mix}

User's request: ${userPrompt}`,
      },
    ],
  });

  const enhanced = response.content[0]?.text;
  return enhanced || userPrompt;
}
