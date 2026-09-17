import React from 'react';

export function MetricPill({
  label,
  value,
  unit = '',
  delta = null,
  deltaType = 'positive', // 'positive' | 'negative' | 'neutral'
  subtitle = null,
  icon: Icon = null,
  highlight = false,
  className = '',
}) {
  const getDeltaColor = () => {
    if (deltaType === 'positive') return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30 shadow-glow-emerald';
    if (deltaType === 'negative') return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
    return 'text-slate-400 bg-slate-800 border-slate-700';
  };

  return (
    <div
      className={`p-3.5 rounded-lg border transition-all flex flex-col justify-between ${
        highlight
          ? 'border-accent/40 bg-gradient-to-b from-accent/10 to-surface-200 shadow-glow-sm'
          : 'border-border-subtle/80 bg-surface-200/90 hover:border-slate-600 hover:bg-surface-100/50'
      } ${className}`}
    >
      <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
        <span className="font-medium text-[11px] uppercase tracking-wider text-slate-400 font-mono">
          {label}
        </span>
        {Icon && (
          <div className="p-1 rounded bg-surface-100 border border-border-subtle text-slate-400">
            <Icon className="w-3.5 h-3.5" />
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-1.5 my-1">
        <span className="text-xl font-bold font-mono tracking-tight text-white">
          {value}
        </span>
        {unit && <span className="text-xs text-slate-400 font-mono">{unit}</span>}
      </div>

      <div className="flex items-center justify-between mt-1 text-[11px]">
        {delta !== null ? (
          <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-bold border ${getDeltaColor()}`}>
            {delta}
          </span>
        ) : <span />}
        {subtitle && <span className="text-slate-400 text-[10px] font-sans truncate">{subtitle}</span>}
      </div>
    </div>
  );
}
