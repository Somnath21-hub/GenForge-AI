/**
 * Scientific and Engineering Formatting Utilities for GenForge AI
 */

export function formatPercentage(val, decimals = 2) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  // If the number is already between 0 and 1, convert to 0-100 if appropriate,
  // but if it's already e.g. 98.68, keep as is
  const num = Number(val);
  const formatted = num <= 1.0 && num > 0.0 ? num * 100 : num;
  return `${formatted.toFixed(decimals)}%`;
}

export function formatPercentagePoint(delta, decimals = 2) {
  if (delta === null || delta === undefined || isNaN(delta)) return '—';
  const num = Number(delta);
  // If delta is e.g. 0.0109, convert to 1.09 if needed
  const val = Math.abs(num) < 0.2 && Math.abs(num) > 0 ? num * 100 : num;
  const sign = val > 0 ? '+' : '';
  return `${sign}${val.toFixed(decimals)} pp`;
}

export function formatNumber(num) {
  if (num === null || num === undefined || isNaN(num)) return '—';
  return Number(num).toLocaleString('en-US');
}

export function formatDate(dateString) {
  if (!dateString) return '—';
  try {
    const d = new Date(dateString);
    if (isNaN(d.getTime())) return dateString;
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
  } catch {
    return dateString;
  }
}

export function formatDecision(decision) {
  if (!decision) return 'UNKNOWN';
  const clean = String(decision).toUpperCase().trim();
  return clean;
}
