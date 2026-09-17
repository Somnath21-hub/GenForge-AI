import React from 'react';
import { Database, AlertCircle } from 'lucide-react';

export function EmptyState({
  title = 'No Data Available',
  description = 'No experiments or datasets have been recorded yet.',
  action = null,
  icon: Icon = Database,
}) {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center tech-card-subtle my-4">
      <div className="w-12 h-12 rounded-full bg-slate-800/80 border border-slate-700/60 flex items-center justify-center mb-3">
        <Icon className="w-5 h-5 text-slate-400" />
      </div>
      <h4 className="text-sm font-semibold text-slate-200">{title}</h4>
      <p className="text-xs text-slate-400 max-w-sm mt-1 mb-4">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
}
