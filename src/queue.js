import { EventEmitter } from "node:events";

export class RequestQueue extends EventEmitter {
  constructor(concurrency = 1) {
    super();
    this.concurrency = concurrency;
    this.queue = [];
    this.active = new Set();
  }

  get length() {
    return this.queue.length + this.active.size;
  }

  enqueue(request) {
    const ac = new AbortController();
    const entry = {
      id: crypto.randomUUID(),
      request,
      signal: ac.signal,
      cancelled: false,
    };

    const cancel = () => {
      entry.cancelled = true;
      if (this.active.has(entry)) {
        ac.abort();
      } else {
        const idx = this.queue.indexOf(entry);
        if (idx !== -1) {
          this.queue.splice(idx, 1);
          this._notifyPositions();
        }
      }
    };

    this.queue.push(entry);

    if (this.active.size < this.concurrency) {
      this._processNext();
    }

    return { position: this.active.has(entry) ? 0 : this.queue.length, cancel };
  }

  async _processNext() {
    if (this.queue.length === 0 || this.active.size >= this.concurrency) {
      return;
    }

    const entry = this.queue.shift();
    if (entry.cancelled) {
      this._processNext();
      return;
    }

    this.active.add(entry);
    this._notifyPositions();

    const { request, signal } = entry;

    try {
      await request.execute(signal);
    } catch {
      // errors handled by the request itself
    } finally {
      this.active.delete(entry);
      this._processNext();
    }
  }

  _notifyPositions() {
    for (let i = 0; i < this.queue.length; i++) {
      const entry = this.queue[i];
      if (!entry.cancelled && entry.request.onPosition) {
        entry.request.onPosition(i + 1);
      }
    }
    for (const entry of this.active) {
      if (entry.request.onPosition) {
        entry.request.onPosition(0);
      }
    }
  }
}
