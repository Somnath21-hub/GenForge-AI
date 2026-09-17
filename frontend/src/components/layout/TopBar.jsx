import React from 'react';
import { useSystem } from '../../context/SystemContext';
import { RefreshCw, Play, Cpu, Activity, Server, Zap, ShieldCheck } from 'lucide-react';

export function TopBar({ title, subtitle, onOpenRunModal }) {
  const { isBackendConnected, checkHealth, systemInfo, isRunningExperiment } = useSystem();

  return (
    <header className="h-14 bg-surface-300/90 backdrop-blur-md border-b border-border-subtle px-6 flex items-center justify-between flex-shrink-0 z-10">
      {/* Current Page Title */}
      <div className="flex flex-col justify-center">
        <h1 className="text-sm font-semibold text-slate-100 tracking-tight font-sans">
          {title}
        </h1>
        {subtitle && <p className="text-[11px] text-slate-400 font-sans">{subtitle}</p>}
      </div>

      {/* Right Actions & Status Badges */}
      <div className="flex items-center gap-3">
        {/* GPU Accelerator Status */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-surface-100/80 border border-border-subtle rounded-md text-[11px] font-mono text-slate-300 shadow-sm">
          <Cpu className="w-3.5 h-3.5 text-accent" />
          <span className="text-slate-400">GPU:</span>
          <span className="font-semibold text-slate-100">
            {systemInfo.gpu_name?.includes('RTX') ? 'RTX 2050 (CUDA 12)' : 'CUDA Device'}
          </span>
        </div>

        {/* Backend Daemon Status Pill */}
        <div
          className={`flex items-center gap-2 px-3 py-1 rounded-md text-[11px] font-mono border shadow-sm transition-all ${
            isBackendConnected
              ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
              : 'bg-rose-500/10 text-rose-300 border-rose-500/30'
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isBackendConnected ? 'bg-emerald-400 shadow-glow-emerald animate-pulse' : 'bg-rose-500'
            }`}
          />
          <span className="font-semibold">
            {isBackendConnected ? 'FastAPI Connected' : 'Daemon Offline'}
          </span>
        </div>

        {/* Refresh Ping Button */}
        <button
          onClick={checkHealth}
          title="Refresh Backend Health & Telemetry"
          className="p-1.5 rounded-md text-slate-400 hover:text-slate-200 hover:bg-surface-100 border border-border-subtle transition-all"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>

        {/* Start Experiment Action */}
        <button
          onClick={onOpenRunModal}
          className="px-3.5 py-1.5 bg-gradient-to-r from-accent to-brand-600 hover:from-accent-hover hover:to-brand-700 text-white rounded-md text-xs font-semibold flex items-center gap-2 transition-all shadow-glow-sm hover:shadow-glow-md"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>Run Experiment</span>
        </button>
      </div>
    </header>
  );
}
