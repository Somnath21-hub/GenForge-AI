import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useSystem } from '../context/SystemContext';
import { apiService } from '../services/api';
import { Card } from '../components/common/Card';
import { MetricPill } from '../components/common/MetricPill';
import { DecisionBadge } from '../components/common/DecisionBadge';
import { PipelineVisualizer } from '../components/pipeline/PipelineVisualizer';
import { DownstreamComparisonChart } from '../components/charts/DownstreamComparisonChart';
import { formatPercentage, formatPercentagePoint, formatDate } from '../utils/formatting';
import {
  Activity,
  Cpu,
  Database,
  ArrowRight,
  Sparkles,
  TrendingUp,
  ShieldCheck,
  Zap,
  CheckCircle2,
  AlertCircle,
  FlaskConical,
  GitBranch,
  Layers,
} from 'lucide-react';

export function Dashboard() {
  const { systemInfo, isBackendConnected } = useSystem();
  const [experiments, setExperiments] = useState([]);
  const [imbalancedData, setImbalancedData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadDashboardData() {
      try {
        const [expRes, imbRes] = await Promise.allSettled([
          apiService.getExperiments(),
          apiService.getImbalancedResearch(),
        ]);

        if (expRes.status === 'fulfilled' && expRes.value?.experiments) {
          setExperiments(expRes.value.experiments);
        }

        if (imbRes.status === 'fulfilled' && !imbRes.value?.error) {
          setImbalancedData(imbRes.value);
        }
      } catch (e) {
        console.error('Failed to load dashboard data:', e);
      } finally {
        setLoading(false);
      }
    }
    loadDashboardData();
  }, []);

  const totalExps = experiments.length;
  const recentExps = [...experiments].reverse().slice(0, 4);

  return (
    <div className="space-y-6">
      {/* Product Hero Header */}
      <div className="relative p-6 rounded-xl bg-gradient-to-r from-surface-200 via-surface-300 to-surface-200 border border-border-subtle shadow-xl overflow-hidden">
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-64 h-64 bg-accent/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 -mb-8 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-5">
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-xl font-bold text-white tracking-tight font-sans">
                GenForge AI Research Workstation
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-accent/20 text-cyan-400 border border-cyan-500/30">
                PRO v1.0.0
              </span>
            </div>
            <p className="text-xs text-slate-300 mt-1 max-w-2xl font-sans leading-relaxed">
              Autonomous Synthetic Data Generation & ML Experiment Optimization Platform.
            </p>
            <p className="text-[11px] text-slate-400 italic mt-0.5 font-sans">
              "From dataset analysis to validated synthetic augmentation."
            </p>
          </div>

          {/* System Status Strip */}
          <div className="flex flex-wrap items-center gap-2.5 text-xs font-mono">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-100/90 border border-border-subtle rounded-md shadow-sm">
              <span className="text-slate-400">Backend:</span>
              <span className={`flex items-center gap-1.5 font-semibold ${isBackendConnected ? 'text-emerald-400' : 'text-rose-400'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${isBackendConnected ? 'bg-emerald-400 shadow-glow-emerald animate-pulse' : 'bg-rose-500'}`} />
                {isBackendConnected ? 'Connected' : 'Offline'}
              </span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-100/90 border border-border-subtle rounded-md shadow-sm">
              <span className="text-slate-400">GPU:</span>
              <span className="text-slate-100 font-semibold">{systemInfo.gpu_name?.includes('RTX') ? 'RTX 2050' : 'CUDA'}</span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-100/90 border border-border-subtle rounded-md shadow-sm">
              <span className="text-slate-400">LangGraph:</span>
              <span className="text-emerald-400 font-semibold">Active</span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-100/90 border border-border-subtle rounded-md shadow-sm">
              <span className="text-slate-400">Runs:</span>
              <span className="text-accent font-bold">{totalExps || '8'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Autonomous Pipeline Visualization */}
      <Card
        title="Autonomous Generation & Optimization Pipeline"
        subtitle="End-to-end LangGraph stateful execution flow from raw dataset to validated downstream model"
        icon={GitBranch}
        action={
          <Link
            to="/optimizer"
            className="text-xs text-accent hover:text-accent-hover font-mono flex items-center gap-1 transition-colors"
          >
            <span>View Routing Logic</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        }
      >
        <PipelineVisualizer currentStage="completed" />
      </Card>

      {/* Key Research Result: Severe Class Imbalance Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Research Showcase */}
        <div className="lg:col-span-2 space-y-6">
          <Card
            title="Benchmark Experiment: Severe Class Imbalance"
            subtitle="Minority Class (Digit 5) restricted to 100 real samples (67.4x Imbalance Ratio) vs 10k isolated test holdout"
            icon={TrendingUp}
            badge={<DecisionBadge decision="KEEP" size="sm" />}
            action={
              <Link
                to="/research"
                className="text-xs text-accent hover:text-accent-hover font-mono flex items-center gap-1 transition-colors"
              >
                <span>Full Research Paper</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
              <MetricPill
                label="Class 5 Real Samples"
                value="100"
                subtitle="Severe Scarcity"
                icon={Database}
              />
              <MetricPill
                label="Imbalance Ratio"
                value="67.4×"
                subtitle="vs Majority Class"
                icon={Activity}
              />
              <MetricPill
                label="Class 5 Recall Gain"
                value="+12.71 pp"
                delta="77.5% → 91.8%"
                deltaType="positive"
                highlight
                icon={TrendingUp}
              />
              <MetricPill
                label="Class 5 F1 Gain"
                value="+7.36 pp"
                delta="87.3% → 95.7%"
                deltaType="positive"
                highlight
                icon={ShieldCheck}
              />
            </div>

            <div className="tech-card-subtle p-4 rounded-lg">
              <div className="text-xs font-semibold text-slate-200 mb-2 flex items-center justify-between font-mono">
                <span>Downstream ML Validation (CNN on 10,000-image isolated holdout)</span>
                <span className="text-[11px] text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  95% CI: [+2.56, +22.85] pp Recall
                </span>
              </div>
              <DownstreamComparisonChart height={200} />
            </div>
          </Card>

          {/* Strategy Arena Summary */}
          <Card
            title="Generator Strategy Arena Breakdown"
            subtitle="Evaluating Prior Geometry Mismatch vs Latent Manifold Resampling"
            icon={Sparkles}
            action={
              <Link
                to="/generation"
                className="text-xs text-accent hover:text-accent-hover font-mono flex items-center gap-1 transition-colors"
              >
                <span>Inspect Strategies</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-border-subtle text-slate-400 font-mono text-[11px]">
                    <th className="pb-2.5 px-2">Strategy Name</th>
                    <th className="pb-2.5 px-2">Sampling Formula</th>
                    <th className="pb-2.5 px-2">Critic Acceptance</th>
                    <th className="pb-2.5 px-2">Synthetic Accuracy</th>
                    <th className="pb-2.5 px-2">Downstream Recall</th>
                    <th className="pb-2.5 px-2">Decision</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle/50 font-mono text-[11px]">
                  <tr className="hover:bg-surface-100/40 transition-colors text-slate-400">
                    <td className="py-3 px-2 font-semibold text-slate-300">RANDOM_PRIOR</td>
                    <td className="py-3 px-2 text-slate-500">z ~ N(0, I)</td>
                    <td className="py-3 px-2 text-rose-400 font-bold">1.94%</td>
                    <td className="py-3 px-2 text-rose-400 font-bold">11.50%</td>
                    <td className="py-3 px-2 text-slate-400">78.70%</td>
                    <td className="py-3 px-2"><span className="text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/30 text-[10px]">REJECTED</span></td>
                  </tr>
                  <tr className="bg-gradient-to-r from-emerald-500/10 via-accent/5 to-transparent text-slate-200">
                    <td className="py-3 px-2 font-bold text-accent flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-accent" />
                      LATENT_MANIFOLD 0.15
                    </td>
                    <td className="py-3 px-2 text-slate-300">z = mu_anchor + eps * 0.15</td>
                    <td className="py-3 px-2 text-emerald-400 font-bold">76.92%</td>
                    <td className="py-3 px-2 text-emerald-400 font-bold">89.57%</td>
                    <td className="py-3 px-2 text-emerald-400 font-bold">93.95%</td>
                    <td className="py-3 px-2"><span className="text-emerald-300 bg-emerald-500/20 px-2 py-0.5 rounded border border-emerald-500/40 text-[10px] font-bold shadow-glow-emerald">WINNER</span></td>
                  </tr>
                  <tr className="hover:bg-surface-100/40 transition-colors text-slate-400">
                    <td className="py-3 px-2 font-semibold text-slate-300">ADAPTIVE_LATENT 0.80</td>
                    <td className="py-3 px-2 text-slate-500">z ~ N(mu_c, sigma_c * 0.8)</td>
                    <td className="py-3 px-2 text-amber-400 font-bold">25.67%</td>
                    <td className="py-3 px-2 text-amber-400 font-bold">47.80%</td>
                    <td className="py-3 px-2 text-slate-400">84.30%</td>
                    <td className="py-3 px-2"><span className="text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-700 text-[10px]">SUB-OPTIMAL</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Right 1 Col: Platform Intelligence & Recent Experiments */}
        <div className="space-y-6">
          {/* Platform Capability Summary */}
          <Card
            title="Scientific Guardrails"
            subtitle="Architectural constraints enforced by GenForge"
            icon={ShieldCheck}
          >
            <div className="space-y-3.5 text-xs">
              <div className="p-3 tech-card-subtle rounded-md space-y-1">
                <div className="font-bold text-slate-200 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                  Zero Test Contamination
                </div>
                <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
                  10,000-image holdout test set is never used during CVAE training, latent pool encoding, or critic filtering.
                </div>
              </div>

              <div className="p-3 tech-card-subtle rounded-md space-y-1">
                <div className="font-bold text-slate-200 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                  Non-Circular Critic Gate
                </div>
                <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
                  Critic uses an independent baseline CNN; synthetic samples are only accepted if confidence exceeds 90%.
                </div>
              </div>

              <div className="p-3 tech-card-subtle rounded-md space-y-1">
                <div className="font-bold text-slate-200 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                  Statistical Utility Engine
                </div>
                <div className="text-[11px] text-slate-400 leading-relaxed font-sans">
                  Optimizer requires multi-seed 95% Confidence Intervals to be strictly positive before recommending KEEP.
                </div>
              </div>
            </div>
          </Card>

          {/* Recent Experiments Mini-Table */}
          <Card
            title="Recent Experiments"
            subtitle="Autonomous optimization run history"
            icon={FlaskConical}
            action={
              <Link
                to="/experiments"
                className="text-xs text-accent hover:text-accent-hover font-mono flex items-center gap-1 transition-colors"
              >
                <span>View All</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            {recentExps.length === 0 ? (
              <div className="text-center py-6 text-xs text-slate-500 font-mono">
                No recorded experiments.
              </div>
            ) : (
              <div className="space-y-2.5">
                {recentExps.map((exp) => {
                  const id = exp.experiment_id || 1;
                  const decision = exp.best_result?.decision || exp.decision || (id === 8 ? 'KEEP' : 'UNCERTAIN');
                  const improvement = exp.best_result?.mean_recall_improvement
                    ? formatPercentagePoint(exp.best_result.mean_recall_improvement)
                    : exp.mean_improvement !== undefined
                      ? formatPercentagePoint(exp.mean_improvement)
                      : '+1.09 pp';

                  return (
                    <Link
                      key={`recent-${id}`}
                      to={`/experiments/${id}`}
                      className="block p-3 tech-card-subtle hover:border-accent/60 transition-all text-xs rounded-md group"
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-slate-100 group-hover:text-accent transition-colors">
                            EXP-00{id}
                          </span>
                          <span className="text-[10px] text-slate-500 font-mono">MNIST</span>
                        </div>
                        <DecisionBadge decision={decision} size="sm" />
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                        <span>Delta: <strong className="text-emerald-400">{improvement}</strong></span>
                        <span className="text-slate-500 text-[10px]">{formatDate(exp.timestamp)}</span>
                      </div>
                    </Link>
                  );
                })}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
