import React, { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine } from 'recharts';

export function MultiSeedGroupedChart({ height = 280 }) {
  const [activeMetric, setActiveMetric] = useState('recall'); // 'recall' | 'f1' | 'overall'

  const rawSeedData = [
    {
      seed: 'Seed 42',
      baselineRecall: 77.47,
      augmentedRecall: 93.95,
      deltaRecall: 16.48,

      baselineF1: 87.30,
      augmentedF1: 96.88,
      deltaF1: 9.58,

      baselineOverall: 97.35,
      augmentedOverall: 98.90,
      deltaOverall: 1.55,
    },
    {
      seed: 'Seed 123',
      baselineRecall: 77.47,
      augmentedRecall: 91.59,
      deltaRecall: 14.12,

      baselineF1: 87.30,
      augmentedF1: 95.61,
      deltaF1: 8.31,

      baselineOverall: 97.35,
      augmentedOverall: 98.68,
      deltaOverall: 1.33,
    },
    {
      seed: 'Seed 456',
      baselineRecall: 77.47,
      augmentedRecall: 84.98,
      deltaRecall: 7.51,

      baselineF1: 87.30,
      augmentedF1: 91.48,
      deltaF1: 4.18,

      baselineOverall: 97.35,
      augmentedOverall: 97.74,
      deltaOverall: 0.39,
    },
  ];

  const chartData = rawSeedData.map((d) => {
    if (activeMetric === 'recall') {
      return {
        name: d.seed,
        baseline: d.baselineRecall,
        augmented: d.augmentedRecall,
        delta: d.deltaRecall,
      };
    }
    if (activeMetric === 'f1') {
      return {
        name: d.seed,
        baseline: d.baselineF1,
        augmented: d.augmentedF1,
        delta: d.deltaF1,
      };
    }
    return {
      name: d.seed,
      baseline: d.baselineOverall,
      augmented: d.augmentedOverall,
      delta: d.deltaOverall,
    };
  });

  const getMetricTitle = () => {
    if (activeMetric === 'recall') return 'Class 5 Recall';
    if (activeMetric === 'f1') return 'Class 5 F1-Score';
    return 'Overall Accuracy';
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-surface-300 border border-border p-3 rounded shadow-xl text-xs font-mono space-y-1">
          <div className="font-bold text-slate-100">{d.name} — {getMetricTitle()}</div>
          <div className="text-slate-400">Baseline: <span className="text-slate-200 font-bold">{d.baseline.toFixed(2)}%</span></div>
          <div className="text-slate-400">Augmented: <span className="text-emerald-400 font-bold">{d.augmented.toFixed(2)}%</span></div>
          <div className="text-emerald-400 border-t border-border-subtle pt-1 font-semibold">
            Gain: +{d.delta.toFixed(2)} pp
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400 font-mono">Metric Dimension:</span>
        <div className="flex items-center gap-1.5 bg-surface-100 p-1 border border-border-subtle rounded text-xs">
          <button
            onClick={() => setActiveMetric('recall')}
            className={`px-2.5 py-1 rounded transition-colors ${
              activeMetric === 'recall' ? 'bg-accent text-white font-medium' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Class 5 Recall
          </button>
          <button
            onClick={() => setActiveMetric('f1')}
            className={`px-2.5 py-1 rounded transition-colors ${
              activeMetric === 'f1' ? 'bg-accent text-white font-medium' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Class 5 F1
          </button>
          <button
            onClick={() => setActiveMetric('overall')}
            className={`px-2.5 py-1 rounded transition-colors ${
              activeMetric === 'overall' ? 'bg-accent text-white font-medium' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Overall Accuracy
          </button>
        </div>
      </div>

      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1F293D" vertical={false} />
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
              domain={[70, 100]}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }}
              formatter={(val) => (val === 'baseline' ? 'Baseline Run' : 'GenForge Augmented Run')}
            />
            <Bar dataKey="baseline" fill="#64748B" radius={[2, 2, 0, 0]} />
            <Bar dataKey="augmented" fill="#10B981" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
