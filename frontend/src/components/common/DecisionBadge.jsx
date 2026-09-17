import React from 'react';
import { DECISION_CONFIG } from '../../utils/constants';

export function DecisionBadge({ decision, size = 'md', showDot = true }) {
  const clean = (decision || 'UNKNOWN').toUpperCase().trim();
  const config = DECISION_CONFIG[clean] || {
    label: clean,
    bg: 'bg-slate-700/20',
    text: 'text-slate-300',
    border: 'border-slate-600/40',
    dot: 'bg-slate-400',
  };

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-[10px]',
    md: 'px-2.5 py-1 text-xs',
    lg: 'px-3 py-1.5 text-sm font-semibold',
  }[size] || 'px-2.5 py-1 text-xs';

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-mono font-medium rounded border ${config.bg} ${config.text} ${config.border} ${sizeClasses}`}
      title={config.description || clean}
    >
      {showDot && (
        <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      )}
      {config.label}
    </span>
  );
}
