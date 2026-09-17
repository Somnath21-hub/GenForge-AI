import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

export function DownstreamComparisonChart({
  height = 240,
  metrics = null,
}) {
  const data = metrics || [
    {
      name: 'Class 5 Recall',
      baseline: 77.47,
      augmented: 93.95,
      delta: 16.48,
    },
    {
      name: 'Class 5 F1 Score',
      baseline: 87.30,
      augmented: 96.88,
      delta: 9.58,
    },
    {
      name: 'Overall Accuracy',
      baseline: 97.35,
      augmented: 98.90,
      delta: 1.55,
    },
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-surface-300/95 backdrop-blur-md border border-border p-3 rounded-lg shadow-2xl text-xs font-mono space-y-1.5 min-w-[210px]">
          <div className="font-bold text-slate-100 pb-1 border-b border-border-subtle">{d.name}</div>
          <div className="flex items-center justify-between text-slate-400">
            <span>Baseline (Real Scarcity):</span>
            <span className="text-slate-200 font-bold">{d.baseline.toFixed(2)}%</span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span>Augmented (GenForge):</span>
            <span className="text-emerald-400 font-bold">{d.augmented.toFixed(2)}%</span>
          </div>
          <div className="flex items-center justify-between text-emerald-400 pt-1 border-t border-border-subtle font-bold">
            <span>Delta:</span>
            <span>+{d.delta.toFixed(2)} pp</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="baselineGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#64748b" stopOpacity={0.8} />
              <stop offset="100%" stopColor="#475569" stopOpacity={0.4} />
            </linearGradient>
            <linearGradient id="augmentedGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.95} />
              <stop offset="100%" stopColor="#059669" stopOpacity={0.5} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#182235" vertical={false} />
          <XAxis
            dataKey="name"
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
          />
          <YAxis
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
            domain={[60, 100]}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
            formatter={(val) => (val === 'baseline' ? 'Baseline (Real Data Only)' : 'Augmented (GenForge Latent Manifold)')}
          />
          <Bar dataKey="baseline" fill="url(#baselineGrad)" radius={[3, 3, 0, 0]} />
          <Bar dataKey="augmented" fill="url(#augmentedGrad)" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
