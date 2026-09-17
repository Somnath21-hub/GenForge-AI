import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

export function StrategyArenaChart({ height = 260 }) {
  const arenaData = [
    {
      strategy: 'RANDOM PRIOR',
      acceptance: 1.94,
      accuracy: 11.50,
      recall: 78.70,
      f1: 88.08,
      selected: false,
    },
    {
      strategy: 'LATENT 0.10',
      acceptance: 77.61,
      accuracy: 89.76,
      recall: 91.59,
      f1: 95.61,
      selected: false,
    },
    {
      strategy: 'LATENT 0.15',
      acceptance: 76.92,
      accuracy: 89.57,
      recall: 93.95,
      f1: 96.88,
      selected: true,
    },
    {
      strategy: 'LATENT 0.20',
      acceptance: 75.79,
      accuracy: 89.05,
      recall: 91.82,
      f1: 95.73,
      selected: false,
    },
    {
      strategy: 'ADAPTIVE 0.80',
      acceptance: 25.67,
      accuracy: 47.80,
      recall: 84.30,
      f1: 91.48,
      selected: false,
    },
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-surface-300/95 backdrop-blur-md border border-border p-3 rounded-lg shadow-2xl text-xs font-mono space-y-1.5 min-w-[200px]">
          <div className="flex items-center justify-between gap-4 pb-1.5 border-b border-border-subtle">
            <span className="font-bold text-slate-100">{d.strategy}</span>
            {d.selected && (
              <span className="bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-1.5 py-0.2 rounded text-[9px] font-bold">
                WINNER
              </span>
            )}
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span>Critic Acceptance:</span>
            <span className="text-cyan-400 font-bold">{d.acceptance.toFixed(2)}%</span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span>Cond. Accuracy:</span>
            <span className="text-purple-400 font-bold">{d.accuracy.toFixed(2)}%</span>
          </div>
          <div className="flex items-center justify-between text-slate-400 pt-1 border-t border-border-subtle">
            <span>Downstream Recall:</span>
            <span className="text-emerald-400 font-bold">{d.recall.toFixed(2)}%</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={arenaData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
          <defs>
            <linearGradient id="cyanGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.9} />
              <stop offset="100%" stopColor="#0891b2" stopOpacity={0.4} />
            </linearGradient>
            <linearGradient id="purpleGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#a855f7" stopOpacity={0.9} />
              <stop offset="100%" stopColor="#7e22ce" stopOpacity={0.4} />
            </linearGradient>
            <linearGradient id="emeraldGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.9} />
              <stop offset="100%" stopColor="#047857" stopOpacity={0.4} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#182235" vertical={false} />
          <XAxis
            dataKey="strategy"
            stroke="#64748B"
            fontSize={10}
            tickLine={false}
          />
          <YAxis
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
            formatter={(val) => {
              if (val === 'acceptance') return 'Critic Acceptance %';
              if (val === 'accuracy') return 'Synthetic Cond. Accuracy %';
              if (val === 'recall') return 'Downstream Class 5 Recall %';
              return val;
            }}
          />
          <Bar dataKey="acceptance" fill="url(#cyanGrad)" radius={[3, 3, 0, 0]} />
          <Bar dataKey="accuracy" fill="url(#purpleGrad)" radius={[3, 3, 0, 0]} />
          <Bar dataKey="recall" fill="url(#emeraldGrad)" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
