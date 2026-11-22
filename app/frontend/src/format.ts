/** Station-local (California) time formatting. */

export const STATION_TZ = 'America/Los_Angeles';

const dateFmt = new Intl.DateTimeFormat('en-GB', {
  timeZone: STATION_TZ,
  weekday: 'short',
  day: '2-digit',
  month: 'short',
  year: 'numeric',
});
// en-US: short zone names are PDT/PST rather than GMT-7.
const timeFmt = new Intl.DateTimeFormat('en-US', {
  timeZone: STATION_TZ,
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
  timeZoneName: 'short',
});
const shortFmt = new Intl.DateTimeFormat('en-GB', {
  timeZone: STATION_TZ,
  day: '2-digit',
  month: 'short',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});
const partsFmt = new Intl.DateTimeFormat('en-CA', {
  timeZone: STATION_TZ,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});

export function formatDate(ms: number): string {
  return dateFmt.format(ms);
}

export function formatTime(ms: number): string {
  return timeFmt.format(ms);
}

export function formatSim(iso: string | number): string {
  return shortFmt.format(typeof iso === 'number' ? iso : Date.parse(iso));
}

/** Value for <input type="datetime-local">, in station time. */
export function toLocalInput(ms: number): string {
  const p = Object.fromEntries(partsFmt.formatToParts(ms).map((x) => [x.type, x.value]));
  const hour = p.hour === '24' ? '00' : p.hour;
  return `${p.year}-${p.month}-${p.day}T${hour}:${p.minute}`;
}

export function speedLabel(speed: number): string {
  const perSec = speed; // simulated seconds per wall second
  if (perSec < 60) return `${perSec.toFixed(0)} s/s`;
  if (perSec < 3600) return `${(perSec / 60).toFixed(perSec % 60 ? 1 : 0)} min/s`;
  return `${(perSec / 3600).toFixed(1)} h/s`;
}

export function ago(ms: number | null, now: number): string {
  if (ms == null) return '—';
  const s = Math.max(0, Math.round((now - ms) / 1000));
  if (s < 60) return `${s} s ago`;
  return `${Math.floor(s / 60)} min ago`;
}
