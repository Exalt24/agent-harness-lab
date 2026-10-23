// A clock the loop reads instead of Date.now and setTimeout, so tests can move time without waiting.
export interface Clock {
  now(): number;
  sleep(ms: number): Promise<void>;
}

export const realClock: Clock = {
  now: () => Date.now(),
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
};

// Time only moves when something advances it. sleep() advances it and records the request.
export class FakeClock implements Clock {
  private t: number;
  readonly sleeps: number[] = [];
  constructor(start = 0) {
    this.t = start;
  }
  now(): number {
    return this.t;
  }
  advance(ms: number): void {
    this.t += ms;
  }
  async sleep(ms: number): Promise<void> {
    this.sleeps.push(ms);
    this.t += ms;
  }
}
