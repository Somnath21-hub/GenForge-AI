import React from 'react';
import { NavLink } from 'react-router-dom';
import { useSystem } from '../../context/SystemContext';
import {
  LayoutDashboard,
  FlaskConical,
  Sparkles,
  CheckCircle2,
  GitBranch,
  Database,
  LineChart,
  Cpu,
  Radio,
} from 'lucide-react';

const MAIN_NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical },
  { to: '/generation', label: 'Synthetic Data', icon: Sparkles },
  { to: '/evaluation', label: 'Evaluation', icon: CheckCircle2 },
  { to: '/optimizer', label: 'Optimization', icon: GitBranch },
];

const ANALYTICS_NAV = [
  { to: '/dataset', label: 'Dataset Analysis', icon: Database },
  { to: '/research', label: 'Research Results', icon: LineChart },
  { to: '/system', label: 'System Diagnostics', icon: Cpu },
];

export function Sidebar() {
  const { systemInfo, isBackendConnected } = useSystem();

  return (
    <aside className="w-64 bg-surface-300 border-r border-border-subtle flex flex-col justify-between flex-shrink-0 select-none z-20">
      {/* Top Section */}
      <div>
        {/* Brand Header - Exactly h-14 to match TopBar */}
        <div className="h-14 px-4 border-b border-border-subtle flex items-center bg-surface-300">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg overflow-hidden flex items-center justify-center flex-shrink-0 bg-surface-100 border border-border-subtle shadow-glow-sm">
              <img
                src="/logo.png"
                alt="GenForge AI Logo"
                className="w-full h-full object-contain p-0.5"
              />
            </div>
            <div className="flex flex-col justify-center">
              <div className="flex items-center gap-1.5 leading-none">
                <span className="font-bold text-slate-100 text-sm tracking-tight font-sans">
                  GenForge
                </span>
                <span className="text-[10px] font-mono font-bold text-cyan-400">
                  AI
                </span>
              </div>
              <span className="text-[10px] text-slate-400 font-sans tracking-tight mt-1 leading-none">
                Autonomous ML Platform
              </span>
            </div>
          </div>
        </div>

        {/* Navigation Sections */}
        <div className="p-3 space-y-5">
          {/* Main Pipeline Navigation */}
          <div>
            <div className="px-3 mb-1.5 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500">
              Pipeline Workspace
            </div>
            <nav className="space-y-0.5">
              {MAIN_NAV.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === '/'}
                    className={({ isActive }) =>
                      `group relative flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-all ${
                        isActive
                          ? 'bg-gradient-to-r from-accent/20 to-indigo-500/10 text-white font-semibold border border-accent/40 shadow-sm'
                          : 'text-slate-400 hover:text-slate-100 hover:bg-surface-100/60'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <Icon className={`w-4 h-4 transition-colors ${isActive ? 'text-accent' : 'text-slate-400 group-hover:text-slate-200'}`} />
                        <span>{item.label}</span>
                        {isActive && (
                          <span className="ml-auto w-1.5 h-1.5 rounded-full bg-accent shadow-glow-sm" />
                        )}
                      </>
                    )}
                  </NavLink>
                );
              })}
            </nav>
          </div>

          {/* Research & Analytics Navigation */}
          <div>
            <div className="px-3 mb-1.5 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500">
              Research & Platform
            </div>
            <nav className="space-y-0.5">
              {ANALYTICS_NAV.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={({ isActive }) =>
                      `group relative flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-all ${
                        isActive
                          ? 'bg-gradient-to-r from-accent/20 to-indigo-500/10 text-white font-semibold border border-accent/40 shadow-sm'
                          : 'text-slate-400 hover:text-slate-100 hover:bg-surface-100/60'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <Icon className={`w-4 h-4 transition-colors ${isActive ? 'text-accent' : 'text-slate-400 group-hover:text-slate-200'}`} />
                        <span>{item.label}</span>
                        {isActive && (
                          <span className="ml-auto w-1.5 h-1.5 rounded-full bg-accent shadow-glow-sm" />
                        )}
                      </>
                    )}
                  </NavLink>
                );
              })}
            </nav>
          </div>
        </div>
      </div>

      {/* Hardware Telemetry Card */}
      <div className="p-3 m-2.5 bg-gradient-to-b from-surface-200 to-surface-300 rounded-lg border border-border-subtle text-[11px] font-mono space-y-2.5 shadow-lg">
        <div className="flex items-center justify-between pb-2 border-b border-border-subtle/80">
          <span className="flex items-center gap-1.5 text-slate-300 font-semibold">
            <Radio className={`w-3 h-3 ${isBackendConnected ? 'text-emerald-400 animate-pulse' : 'text-rose-500'}`} />
            Engine Telemetry
          </span>
          <span className="text-[10px] px-1.5 py-0.2 rounded bg-surface-100 text-slate-400 border border-border-subtle">
            v1.0.0
          </span>
        </div>

        <div className="flex items-center justify-between text-slate-400">
          <span className="text-slate-500">Accelerator</span>
          <span className="text-slate-200 font-semibold truncate max-w-[120px]" title={systemInfo.gpu_name}>
            {systemInfo.gpu_name?.includes('RTX') ? 'RTX 2050' : 'PyTorch CUDA'}
          </span>
        </div>

        <div className="flex items-center justify-between text-slate-400">
          <span className="text-slate-500">FastAPI Daemon</span>
          <span className="flex items-center gap-1.5 font-semibold">
            <span className={`w-1.5 h-1.5 rounded-full ${isBackendConnected ? 'bg-emerald-400 shadow-glow-emerald' : 'bg-rose-500'}`} />
            <span className={isBackendConnected ? 'text-emerald-400' : 'text-rose-400'}>
              {isBackendConnected ? 'Active :8000' : 'Offline'}
            </span>
          </span>
        </div>
      </div>
    </aside>
  );
}
