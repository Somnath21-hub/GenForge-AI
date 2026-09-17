import React, { useState } from 'react';

export function ConfusionMatrixHeatmap({ baselineMatrix, augmentedMatrix }) {
  const [activeView, setActiveView] = useState('augmented'); // 'augmented' | 'baseline' | 'diff'

  // Default matrices from severe imbalance experiment (Seed 42)
  const defaultBaseline = [
    [977, 0, 0, 0, 0, 0, 0, 1, 2, 0],
    [1, 1129, 1, 2, 0, 0, 1, 0, 1, 0],
    [0, 0, 1031, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1008, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 971, 0, 3, 0, 4, 4],
    [1, 3, 1, 70, 0, 691, 12, 1, 97, 16], // Class 5 row: 691 TP, 201 FN!
    [3, 1, 0, 0, 2, 0, 952, 0, 0, 0],
    [0, 5, 5, 0, 1, 0, 0, 1015, 2, 0],
    [4, 0, 1, 1, 0, 0, 0, 0, 966, 2],
    [1, 0, 0, 2, 5, 0, 0, 1, 0, 1000],
  ];

  const defaultAugmented = [
    [979, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1132, 0, 3, 0, 0, 0, 0, 0, 0],
    [1, 0, 1023, 1, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 1009, 0, 0, 0, 0, 1, 0],
    [0, 2, 0, 0, 972, 0, 1, 0, 1, 6],
    [1, 2, 0, 45, 0, 817, 10, 3, 11, 3], // Class 5 row: 817 TP (up from 691!), only 75 FN!
    [6, 2, 0, 0, 1, 0, 948, 0, 1, 0],
    [0, 5, 3, 0, 1, 0, 0, 1016, 3, 0],
    [2, 0, 1, 1, 0, 0, 0, 0, 970, 0],
    [0, 0, 0, 1, 4, 0, 0, 0, 0, 1004],
  ];

  const currentMatrix = activeView === 'baseline'
    ? (baselineMatrix || defaultBaseline)
    : (augmentedMatrix || defaultAugmented);

  const getCellColor = (rowIdx, colIdx, val) => {
    const isDiagonal = rowIdx === colIdx;
    const isClass5 = rowIdx === 5 || colIdx === 5;

    if (isDiagonal) {
      if (val >= 900) return 'bg-emerald-600/80 text-white font-bold';
      if (val >= 750) return 'bg-emerald-600/60 text-slate-100 font-bold';
      if (val >= 600) return 'bg-emerald-700/50 text-slate-200 font-semibold';
      return 'bg-emerald-800/40 text-slate-300';
    }

    // Off-diagonal errors
    if (val === 0) return 'bg-surface-300/40 text-slate-600';
    if (val <= 2) return 'bg-surface-100 text-slate-400';
    if (val <= 10) return 'bg-amber-500/20 text-amber-300 font-medium';
    if (val <= 50) return 'bg-rose-500/30 text-rose-300 font-semibold';
    return 'bg-rose-600/50 text-white font-bold ring-1 ring-rose-500';
  };

  return (
    <div className="space-y-4">
      {/* Controls & Legend */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-1.5 bg-surface-100 p-1 border border-border-subtle rounded">
          <button
            onClick={() => setActiveView('augmented')}
            className={`px-3 py-1 rounded transition-colors font-medium ${
              activeView === 'augmented' ? 'bg-accent text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Augmented Matrix (GenForge)
          </button>
          <button
            onClick={() => setActiveView('baseline')}
            className={`px-3 py-1 rounded transition-colors font-medium ${
              activeView === 'baseline' ? 'bg-accent text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Baseline Matrix (Real Scarcity)
          </button>
        </div>

        <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-emerald-600 inline-block" />
            <span>High True Positives</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-rose-500/40 inline-block" />
            <span>High Misclassifications</span>
          </div>
        </div>
      </div>

      {/* Matrix Heatmap */}
      <div className="overflow-x-auto pb-2">
        <div className="min-w-[500px]">
          {/* Column Header (Predicted) */}
          <div className="text-center text-xs font-mono font-semibold text-slate-400 mb-1">
            Predicted Class Label →
          </div>

          <div className="grid grid-cols-[40px_repeat(10,1fr)] gap-1 text-center text-xs font-mono">
            {/* Header row */}
            <div className="h-7 flex items-center justify-center text-[10px] text-slate-500 font-bold">
              T \ P
            </div>
            {Array.from({ length: 10 }).map((_, i) => (
              <div
                key={`hdr-${i}`}
                className={`h-7 flex items-center justify-center font-bold text-slate-300 rounded ${
                  i === 5 ? 'bg-amber-500/10 text-amber-300 border border-amber-500/30' : 'bg-surface-100'
                }`}
              >
                {i}
              </div>
            ))}

            {/* Matrix rows */}
            {currentMatrix.map((row, rowIdx) => (
              <React.Fragment key={`row-${rowIdx}`}>
                {/* Row label (True class) */}
                <div
                  className={`h-8 flex items-center justify-center font-bold text-slate-300 rounded ${
                    rowIdx === 5 ? 'bg-amber-500/10 text-amber-300 border border-amber-500/30' : 'bg-surface-100'
                  }`}
                  title={`True Class ${rowIdx}`}
                >
                  {rowIdx}
                </div>

                {/* Row cells */}
                {row.map((val, colIdx) => (
                  <div
                    key={`cell-${rowIdx}-${colIdx}`}
                    title={`True: ${rowIdx}, Predicted: ${colIdx} → ${val} samples`}
                    className={`h-8 flex items-center justify-center text-[11px] rounded transition-all cursor-default ${getCellColor(
                      rowIdx,
                      colIdx,
                      val
                    )} ${rowIdx === 5 && colIdx === 5 ? 'ring-2 ring-emerald-400' : ''}`}
                  >
                    {val}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono mt-3">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              Row/Col 5 corresponds to the minority class (100 real samples).
            </span>
            <span>
              {activeView === 'augmented' ? (
                <strong className="text-emerald-400 font-bold">Class 5 True Positives: 817 (+126 over baseline)</strong>
              ) : (
                <strong className="text-rose-400 font-bold">Class 5 True Positives: 691 (201 False Negatives)</strong>
              )}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
