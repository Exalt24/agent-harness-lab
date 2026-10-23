export interface Interval {
  low: number;
  high: number;
}

// Wilson score interval for a binomial proportion. It stays inside [0, 1] and behaves at 0 of n
// and n of n, which the plain normal interval does not. z = 1.96 gives 95 percent.
export function wilson(successes: number, n: number, z = 1.96): Interval | null {
  if (n <= 0) return null;
  if (successes < 0 || successes > n) throw new RangeError(`successes ${successes} outside 0..${n}`);
  const p = successes / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return { low: Math.max(0, center - half), high: Math.min(1, center + half) };
}

export const pct = (x: number): string => `${(x * 100).toFixed(0)}%`;

// A rate is never shown without its interval and its counts. No scored trials means no rate.
export function formatRate(pass: number, scored: number, errors: number): string {
  const ci = wilson(pass, scored);
  const err = errors > 0 ? `, ${errors} ERROR` : "";
  if (!ci) return `no scored trials${err}`;
  return `${pass}/${scored} = ${pct(pass / scored)} (95% CI ${pct(ci.low)} to ${pct(ci.high)})${err}`;
}

export function overlaps(a: Interval | null, b: Interval | null): boolean | null {
  if (!a || !b) return null;
  return a.low <= b.high && b.low <= a.high;
}
