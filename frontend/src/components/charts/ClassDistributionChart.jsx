import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from 'recharts';

export function ClassDistributionChart({
  data = null,
  minorityClass = 5,
  height = 240,
}) {
  // Default MNIST Imbalanced distribution if data is not provided
  const chartData = data || [
    { class_id: 0, count: 5923, name: 'Digit 0' },
    { class_id: 1, count: 6742, name: 'Digit 1' },
    { class_id: 2, count: 5958, name: 'Digit 2' },
    { class_id: 3, count: 6131, name: 'Digit 3' },
    { class_id: 4, count: 5842, name: 'Digit 4' },
    { class_id: 5, count: 100, name: 'Digit 5 (Minority)' },
    { class_id: 6, count: 5918, name: 'Digit 6' },
    { class_id: 7, count: 6265, name: 'Digit 7' },
    { class_id: 8, count: 5851, name: 'Digit 8' },
    { class_id: 9, count: 5949, name: 'Digit 9' },
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      const isMinority = d.class_id === minorityClass;
      return (
        <div className="bg-surface-300 border border-border p-2.5 rounded shadow-lg text-xs font-mono">
          <div className="font-semibold text-slate-200">{d.name}</div>
          <div className="text-slate-400 mt-1">
            Samples: <span className="text-slate-100 font-bold">{d.count.toLocaleString()}</span>
          </div>
          {isMinority && (
            <div className="text-amber-400 text-[10px] mt-1 font-semibold">
              Severe Scarcity (67.4x Imbalance)
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
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
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {chartData.map((entry) => (
              <Cell
                key={`cell-${entry.class_id}`}
                fill={entry.class_id === minorityClass ? '#F59E0B' : '#3B82F6'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
