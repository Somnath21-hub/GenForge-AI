import React from 'react';
import { Card } from '../components/common/Card';
import { DecisionBadge } from '../components/common/DecisionBadge';
import { MetricPill } from '../components/common/MetricPill';
import { GitBranch, CheckCircle2, RefreshCw, AlertCircle, ArrowRight, Sliders, ShieldCheck } from 'lucide-react';

export function Optimizer() {
  const timelineSteps = [
    {
      iter: 1,
      strategy: 'RANDOM_PRIOR (z ~ N(0, I))',
      quality: '11.50% Cond. Accuracy',
      acceptance: '1.94% (Critic Threshold 0.90)',
      downstreamDelta: '+0.12 pp (Recall: 78.70%)',
      decision: 'RESAMPLE',
      reason: 'Prior mismatch detected. Gaussian prior lands in void latent space. Critic rejects 98% of samples.',
      status: 'warning',
    },
    {
      iter: 2,
      strategy: 'ADAPTIVE_LATENT (scale 0.80)',
      quality: '47.80% Cond. Accuracy',
      acceptance: '25.67%',
      downstreamDelta: '+0.88 pp (Recall: 84.30%)',
      decision: 'RESAMPLE',
      reason: 'Empirical class covariance improves accuracy, but acceptance remains below optimal target.',
      status: 'info',
    },
    {
      iter: 3,
      strategy: 'LATENT_MANIFOLD (scale 0.15)',
      quality: '89.57% Cond. Accuracy',
      acceptance: '76.92%',
      downstreamDelta: '+1.55 pp Overall, +16.48 pp Recall',
      decision: 'KEEP',
      reason: 'Statistically significant positive downstream utility (95% CI strictly positive). Augmented model saved.',
      status: 'success',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-slate-100">Autonomous Strategy Optimizer</h2>
        <p className="text-xs text-slate-400">
          Stateful LangGraph decision engine validating downstream utility and guiding generative adaptation
        </p>
      </div>

      {/* Decision Flow Strip */}
      <Card
        title="Autonomous Optimization State Machine"
        subtitle="Dynamic cycle executed by the LangGraph orchestrator"
      >
        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 text-xs font-mono">
          <div className="p-3 tech-card-subtle rounded border border-border-subtle">
            <div className="text-[10px] text-slate-500 font-bold">01. OBSERVE</div>
            <div className="font-semibold text-slate-200 mt-1">Dataset Profile</div>
            <div className="text-[11px] text-slate-400 mt-0.5">Detects minority scarcity (67.4x)</div>
          </div>

          <div className="p-3 tech-card-subtle rounded border border-border-subtle">
            <div className="text-[10px] text-slate-500 font-bold">02. EVALUATE</div>
            <div className="font-semibold text-slate-200 mt-1">Critic Gate</div>
            <div className="text-[11px] text-slate-400 mt-0.5">Filters confidence ≥ 0.90</div>
          </div>

          <div className="p-3 tech-card-subtle rounded border border-border-subtle">
            <div className="text-[10px] text-slate-500 font-bold">03. COMPARE</div>
            <div className="font-semibold text-slate-200 mt-1">Holdout CNN</div>
            <div className="text-[11px] text-slate-400 mt-0.5">Measures recall delta on 10k set</div>
          </div>

          <div className="p-3 tech-card-subtle rounded border border-accent/40 bg-accent/5">
            <div className="text-[10px] text-accent font-bold">04. DECIDE</div>
            <div className="font-semibold text-slate-100 mt-1">Statistical Engine</div>
            <div className="text-[11px] text-slate-300 mt-0.5">Evaluates 95% Confidence Interval</div>
          </div>

          <div className="p-3 tech-card-subtle rounded border border-emerald-500/30 bg-emerald-500/5">
            <div className="text-[10px] text-emerald-400 font-bold">05. ROUTE</div>
            <div className="font-semibold text-emerald-300 mt-1">Action Routing</div>
            <div className="text-[11px] text-slate-300 mt-0.5">KEEP, RESAMPLE, RETRAIN, STOP</div>
          </div>
        </div>
      </Card>

      {/* Autonomous Iteration Timeline */}
      <Card
        title="Optimization Iteration History & Decision Trace"
        subtitle="Chronological trail of adaptive strategies explored by the optimizer"
      >
        <div className="space-y-4">
          {timelineSteps.map((step) => (
            <div
              key={`step-${step.iter}`}
              className={`p-4 rounded-lg border transition-all text-xs font-mono ${
                step.decision === 'KEEP'
                  ? 'bg-emerald-500/5 border-emerald-500/30'
                  : 'bg-surface-100 border-border-subtle'
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2.5 border-b border-border-subtle/60">
                <div className="flex items-center gap-2.5">
                  <span className="w-6 h-6 rounded bg-surface-300 border border-border-subtle flex items-center justify-center font-bold text-slate-300 text-xs">
                    0{step.iter}
                  </span>
                  <span className="font-bold text-slate-200 text-sm">{step.strategy}</span>
                </div>
                <DecisionBadge decision={step.decision} size="md" />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 my-3 text-[11px]">
                <div>
                  <span className="text-slate-500">Critic Acceptance:</span>
                  <span className="text-slate-200 font-semibold ml-1.5">{step.acceptance}</span>
                </div>
                <div>
                  <span className="text-slate-500">Quality Score:</span>
                  <span className="text-slate-200 font-semibold ml-1.5">{step.quality}</span>
                </div>
                <div>
                  <span className="text-slate-500">Downstream Delta:</span>
                  <span className="text-emerald-400 font-bold ml-1.5">{step.downstreamDelta}</span>
                </div>
              </div>

              <div className="text-slate-400 text-xs bg-surface-300 p-2.5 rounded border border-border-subtle">
                <span className="text-slate-500 font-bold">Optimizer Rationale: </span>
                {step.reason}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Decision Rules Reference */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Decision Thresholds" subtitle="Mathematical criteria for optimizer branches">
          <div className="space-y-2.5 text-xs font-mono">
            <div className="flex items-center justify-between p-2 tech-card-subtle rounded">
              <span className="text-emerald-400 font-bold">KEEP</span>
              <span className="text-slate-300">Mean Δ &gt; 0 and 95% CI strictly positive</span>
            </div>
            <div className="flex items-center justify-between p-2 tech-card-subtle rounded">
              <span className="text-amber-400 font-bold">UNCERTAIN</span>
              <span className="text-slate-300">Mean Δ &gt; 0 but 95% CI crosses zero (noise)</span>
            </div>
            <div className="flex items-center justify-between p-2 tech-card-subtle rounded">
              <span className="text-purple-400 font-bold">RESAMPLE</span>
              <span className="text-slate-300">Critic Acc &lt; 70% or Recall gain sub-optimal</span>
            </div>
            <div className="flex items-center justify-between p-2 tech-card-subtle rounded">
              <span className="text-blue-400 font-bold">RETRAIN</span>
              <span className="text-slate-300">Recon MSE &gt; 0.08 (encoder structural flaw)</span>
            </div>
          </div>
        </Card>

        <Card title="Guaranteed Convergence" subtitle="Stopping criteria & safety guards">
          <div className="space-y-3 text-xs text-slate-300">
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
              <div>
                <strong>Maximum Iteration Guard:</strong> Bounded at $K \le 5$ iterations to avoid infinite feedback loops.
              </div>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
              <div>
                <strong>Checkpoint Immutability:</strong> Base models are never overwritten in-place during exploration.
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
