import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { WebSocketServer } from "ws";
import { RequestQueue } from "./src/queue.js";
import { UserStore } from "./src/users.js";
import { checkHealth, adapters, generate } from "./src/llama.js";
import { handleConnection } from "./src/ws-handler.js";

const PORT = process.env.PORT || 3000;

const indexHtml = fs.readFileSync(
  path.join(import.meta.dirname, "index.html"),
  "utf-8"
);
const sourcesHtml = fs.readFileSync(
  path.join(import.meta.dirname, "sources.html"),
  "utf-8"
);

const MODELS_DIR = process.env.MODELS_DIR || "";

const CONCURRENCY = parseInt(process.env.CONCURRENCY || "2", 10);
const queue = new RequestQueue(CONCURRENCY);
const userStore = new UserStore();

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString()));
    req.on("error", reject);
  });
}

function json(res, status, data) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

const server = http.createServer(async (req, res) => {
  if (req.url === "/" || req.url === "/index.html") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(indexHtml);
    return;
  }

  if (req.url === "/sources") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(sourcesHtml);
    return;
  }

  if (req.url.startsWith("/download/") && req.method === "GET") {
    if (!MODELS_DIR) {
      res.writeHead(404);
      res.end("MODELS_DIR not configured");
      return;
    }
    const fileName = path.basename(decodeURIComponent(req.url.slice("/download/".length)));
    if (!fileName || fileName.startsWith(".")) {
      res.writeHead(400);
      res.end("invalid filename");
      return;
    }
    const filePath = path.join(MODELS_DIR, fileName);
    if (!filePath.startsWith(path.resolve(MODELS_DIR))) {
      res.writeHead(403);
      res.end("forbidden");
      return;
    }
    try {
      const stat = fs.statSync(filePath);
      res.writeHead(200, {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": `attachment; filename="${fileName}"`,
        "Content-Length": stat.size,
      });
      fs.createReadStream(filePath).pipe(res);
    } catch {
      res.writeHead(404);
      res.end("file not found");
    }
    return;
  }

  if (req.url === "/health") {
    try {
      if (await checkHealth()) {
        res.writeHead(200);
        res.end("ok");
      } else {
        res.writeHead(503);
        res.end("llama.cpp unreachable");
      }
    } catch {
      res.writeHead(503);
      res.end("llama.cpp unreachable");
    }
    return;
  }

  if (req.url === "/api/adapters" && req.method === "GET") {
    json(res, 200, { adapters });
    return;
  }

  if (req.url === "/api/generate" && req.method === "POST") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch {
      json(res, 400, { error: "invalid json" });
      return;
    }

    const { adapterWeights, messages, prompt, options } = body;
    if (!messages && !prompt) {
      json(res, 400, { error: "messages or prompt required" });
      return;
    }

    const msgs = messages || [{ role: "user", content: prompt }];

    const result = await new Promise((resolve) => {
      const { cancel } = queue.enqueue({
        onPosition() {},
        async execute(signal) {
          await generate({
            adapterWeights: adapterWeights || {},
            messages: msgs,
            options,
            signal,
            onToken() {},
            onDone(fullText) {
              resolve({
                text: fullText,
                meta: {
                  adapters: adapterWeights || {},
                  options: { temperature: options?.temperature, max_tokens: options?.max_tokens },
                  timestamp: new Date().toISOString(),
                },
              });
            },
            onError(errText) {
              resolve({ error: errText });
            },
          });
        },
      });

      setTimeout(() => {
        cancel();
        resolve({ error: "timeout" });
      }, 300_000);
    });

    if (result.error) {
      json(res, 502, { error: result.error });
    } else if (result.text !== undefined) {
      json(res, 200, { text: result.text, meta: result.meta });
    }
    return;
  }

  res.writeHead(404);
  res.end("not found");
});

const wss = new WebSocketServer({ server });
wss.on("connection", (ws) => handleConnection(ws, queue, userStore));

server.listen(PORT, () => {
  console.log(`palimpsest listening on :${PORT}`);
  console.log(`llama.cpp: ${process.env.LLAMA_URL || "http://localhost:8080"}`);
  console.log(`adapters: ${adapters.map(a => a.id).join(", ") || "none (base model only)"}`);
});
