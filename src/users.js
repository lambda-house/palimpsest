import crypto from "node:crypto";

const MAX_HISTORY = 50;

export class UserStore {
  constructor() {
    this.users = new Map();
  }

  getOrCreate(userId, systemPrompt) {
    if (userId && this.users.has(userId)) {
      const user = this.users.get(userId);
      // update system prompt if provided
      if (systemPrompt && user.history[0]?.role === "system") {
        user.history[0].content = systemPrompt;
      }
      return user;
    }
    const id = userId || crypto.randomUUID();
    const history = systemPrompt
      ? [{ role: "system", content: systemPrompt }]
      : [];
    const user = { userId: id, history };
    this.users.set(id, user);
    return user;
  }

  appendMessage(userId, role, content) {
    const user = this.getOrCreate(userId);
    user.history.push({ role, content });
    // trim keeping system prompt (if present) + last MAX_HISTORY messages
    const hasSystem = user.history[0]?.role === "system";
    const offset = hasSystem ? 1 : 0;
    if (user.history.length > MAX_HISTORY + offset) {
      const prefix = hasSystem ? [user.history[0]] : [];
      user.history = [...prefix, ...user.history.slice(-(MAX_HISTORY))];
    }
  }

  getHistory(userId) {
    return this.getOrCreate(userId).history;
  }

  clearHistory(userId, systemPrompt) {
    const user = this.users.get(userId);
    if (user) {
      const oldSystem = user.history[0]?.role === "system" ? user.history[0].content : null;
      const prompt = systemPrompt || oldSystem;
      user.history = prompt ? [{ role: "system", content: prompt }] : [];
    }
  }
}
