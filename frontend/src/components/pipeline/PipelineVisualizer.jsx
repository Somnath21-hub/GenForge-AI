import React from 'react';
import { PIPELINE_STAGES } from '../../utils/constants';
import { CheckCircle2, Loader2, Circle, ArrowRight, ChevronRight } from 'lucide-react';

export function PipelineVisualizer({
  currentStage = 'completed', // 'idle' | 'dataset' | 'analyze' | 'plan' | 'generate' | 'critic' | 'evaluate' | 'optimize' | 'completed'
  onSelectStage = null,
  activeSelectedStage = null,
  compact = false,
}) {
  const stageOrder = ['dataset', 'analyze', 'plan', 'generate', 'critic', 'evaluate', 'optimize'];
  const activeIdx = currentStage === 'completed'
    ? stageOrder.length
    : currentStage === 'idle'
      ? -1
      : stageOrder.indexOf(currentStage);

  return (
    <div className="w-full">
      <div className="flex items-center justify-between gap-1.5 overflow-x-auto pb-1 scrollbar-none">
        {PIPELINE_STAGES.map((stage, idx) => {
          const isCompleted = activeIdx > idx;
          const isCurrent = activeIdx === idx;
          const isPending = activeIdx < idx;
          const isSelected = activeSelectedStage === stage.id;

          let statusBg = 'bg-surface-200/80 border-border-subtle text-slate-400';
          let icon = <Circle className="w-3.5 h-3.5 text-slate-600" />;

          if (isCompleted) {
            statusBg = 'bg-gradient-to-b from-emerald-500/15 to-emerald-950/20 border-emerald-500/40 text-emerald-300 shadow-sm';
            icon = <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
          } else if (isCurrent) {
            statusBg = 'bg-gradient-to-b from-blue-500/25 to-indigo-950/40 border-accent text-blue-200 shadow-glow-md ring-1 ring-accent/50 animate-pulse';
            icon = <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />;
          }

          if (isSelected) {
            statusBg += ' ring-2 ring-accent';
          }

          return (
            <React.Fragment key={stage.id}>
              <div
                onClick={() => onSelectStage && onSelectStage(stage.id)}
                className={`flex-1 min-w-[125px] p-3 rounded-lg border transition-all ${statusBg} ${
                  onSelectStage ? 'cursor-pointer hover:border-slate-400' : ''
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] font-mono font-bold text-slate-500">
                    0{idx + 1}
                  </span>
                  {icon}
                </div>
                <div className="text-xs font-bold text-slate-100 tracking-wide font-sans">
                  {stage.label}
                </div>
                {!compact && (
                  <div className="text-[10px] text-slate-400 truncate mt-0.5 font-sans" title={stage.desc}>
                    {stage.desc}
                  </div>
                )}
              </div>

              {idx < PIPELINE_STAGES.length - 1 && (
                <div className="flex-shrink-0 text-slate-600 px-0.5">
                  <ChevronRight className="w-4 h-4 text-slate-600" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
