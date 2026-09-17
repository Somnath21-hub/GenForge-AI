import React from 'react';
import { Loader2 } from 'lucide-react';

export function LoadingState({ message = 'Loading experiment data...', className = '' }) {
  return (
    <div className={`flex flex-col items-center justify-center p-12 text-slate-400 ${className}`}>
      <Loader2 className="w-7 h-7 animate-spin text-accent mb-3" />
      <p className="text-xs font-mono tracking-wide">{message}</p>
    </div>
  );
}
