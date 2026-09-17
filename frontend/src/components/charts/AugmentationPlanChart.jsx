import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

export function AugmentationPlanChart({
  height = 240,
}) {
  const planData = [
    { class_id: 0, real: 5923, synthetic: 819, target: 6742 },
    { class_id: 1, real: 6742, synthetic: 0, target: 6742 },
    { class_id: 2, real: 5958, synthetic: 784, target: 6742 },
    { class_id: 3, real: 6131, synthetic: 611, target: 6742 },
    { class_id: 4, real: 5842, synthetic: 900, target: 6742 },
    { class_id: 5, real: 100, synthetic: 6642, target: 6742 },
    { class_id: 6, real: 5918, synthetic: 824, target: 6742 },
    { class_id: 7, real: 6265, synthetic: 477, target: 6742 },
    { class_id: 8, real: 5851, synthetic: 891, target: 6742 },
    { class_id: 9, real: 5949, synthetic: 793, target: 6742 },
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-surface-300 border border-border p-2.5 rounded shadow-lg text-xs font-mono space-y-1">
          <div className="font-semibold text-slate-200">Class {d.class_id} Augmentation</div>
          <div className="text-slate-400">Real Samples: <span className="text-blue-400 font-bold">{d.real.toLocaleString()}</span></div>
          <div className="text-slate-400">Synthetic Target: <span className="text-emerald-400 font-bold">+{d.synthetic.toLocaleString()}</span></div>
          <div className="text-slate-400 border-t border-border-subtle pt-1">Balanced Target: <span className="text-slate-200 font-bold">{d.target.toLocaleString()}</span></div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={planData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1F293D" vertical={false} />
          <XAxis
            dataKey="class_id"
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
            tickFormatter={(val) => `D${val}`}
          />
          <YAxis
            stroke="#64748B"
            fontSize={11}
            tickLine={false}
            tickFormatter={(val) => (val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }}
            formatter={(val) => (val === 'real' ? 'Real Samples' : 'Synthetic Augmentation')}
          />
          <Bar dataKey="real" stackId="a" fill="#3B82F6" radius={[0, 0, 0, 0]} />
          <Bar dataKey="synthetic" stackId="a" fill="#10B981" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
