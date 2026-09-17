import React, { useState } from 'react';
import { Card } from '../components/common/Card';
import { MetricPill } from '../components/common/MetricPill';
import { StrategyArenaChart } from '../components/charts/StrategyArenaChart';
import { Sparkles, Brain, Cpu, ShieldCheck, CheckCircle2, AlertTriangle, Layers, Sliders } from 'lucide-react';

export function Generation() {
  const [selectedScale, setSelectedScale] = useState(0.15);

  const scaleGrid = [
    { scale: 0.05, acceptance: '82.4%', accuracy: '91.2%', diversity: 0.21, desc: 'Ultra-tight manifold radius; highest fidelity, lower variance.' },
    { scale: 0.10, acceptance: '77.6%', accuracy: '89.8%', diversity: 0.28, desc: 'Balanced manifold exploration.' },
    { scale: 0.15, acceptance: '76.9%', accuracy: '89.6%', diversity: 0.33, desc: 'Optimal trade-off: high acceptance with maximum diversity (Recommended).' },
    { scale: 0.20, acceptance: '75.8%', accuracy: '89.1%', diversity: 0.36, desc: 'Higher variance; boundary artifacts start appearing.' },
    { scale: 0.30, acceptance: '62.4%', accuracy: '78.5%', diversity: 0.42, desc: 'Manifold escape; significant critic rejections.' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-slate-100">Synthetic Data Generation Strategies</h2>
        <p className="text-xs text-slate-400">
          Prior geometry analysis, Latent Manifold resampling mechanics, and Critic confidence filtering
        </p>
      </div>

      {/* Core Architectural Insight: Prior Mismatch */}
      <Card
        title="Key Research Discovery: Latent Manifold Sampling vs Prior Mismatch"
        subtitle="Sampling geometry resolution for Conditional Variational Autoencoders"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          {/* Left: Random Prior Problem */}
          <div className="p-4 tech-card-subtle rounded-lg border-rose-500/20 bg-rose-500/5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-mono font-bold text-rose-300 flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-rose-400" />
                RANDOM GAUSSIAN PRIOR [z ~ N(0, I)]
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/20 text-rose-300 border border-rose-500/30">
                26% – 33% Accuracy
              </span>
            </div>
            <p className="text-slate-300 leading-relaxed">
              Standard CVAE generation draws random vectors from a standard normal distribution N(0, I) independently of class condition c. Because the encoder maps empirical data to dense, class-specific sub-manifolds with non-zero centroids, sampling from the standard Gaussian prior lands in void regions between manifolds.
            </p>
            <div className="p-2.5 bg-surface-300 rounded font-mono text-[11px] text-slate-400 space-y-1">
              <div>• V1 Conditional Accuracy: <span className="text-rose-400 font-bold">26.1%</span></div>
              <div>• V2 Conditional Accuracy: <span className="text-rose-400 font-bold">29.4%</span></div>
              <div>• V3 Conditional Accuracy: <span className="text-rose-400 font-bold">33.2%</span></div>
              <div className="text-slate-500 text-[10px] mt-1 pt-1 border-t border-border-subtle">
                Diagnosis: Encoder reconstruction capability is &gt;98.5%, proving the decoder is fine. The failure is purely sampling geometry.
              </div>
            </div>
          </div>

          {/* Right: Latent Manifold Solution */}
          <div className="p-4 tech-card-subtle rounded-lg border-emerald-500/30 bg-emerald-500/5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-mono font-bold text-emerald-300 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                LATENT MANIFOLD RESAMPLING (GenForge)
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-bold">
                98.3% – 99.0% Accuracy (+72.4 pp)
              </span>
            </div>
            <p className="text-slate-300 leading-relaxed">
              GenForge samples z = mu_anchor + epsilon * scale, anchoring vectors directly on the empirical manifold of available training instances, perturbed with controlled scale sigma in [0.05, 0.20]. Zero test leakage is guaranteed by encoding training samples only.
            </p>
            <div className="p-2.5 bg-surface-300 rounded font-mono text-[11px] text-slate-400 space-y-1">
              <div>• V1 Latent Manifold: <span className="text-emerald-400 font-bold">98.30%</span></div>
              <div>• V2 Latent Manifold: <span className="text-emerald-400 font-bold">98.90%</span></div>
              <div>• V3 Latent Manifold: <span className="text-emerald-400 font-bold">99.00%</span></div>
              <div className="text-emerald-400 text-[10px] mt-1 pt-1 border-t border-border-subtle font-semibold">
                Result: Unlocks full generative fidelity without retraining or altering generator weights.
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Perturbation Scale Sensitivity Grid */}
      <Card
        title="Latent Perturbation Scale Sensitivity Grid"
        subtitle="Trade-off between Critic acceptance rate and synthetic diversity"
      >
        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 mb-6">
          {scaleGrid.map((item) => (
            <div
              key={`scale-${item.scale}`}
              onClick={() => setSelectedScale(item.scale)}
              className={`p-3.5 tech-card-subtle rounded-lg cursor-pointer transition-all border ${
                selectedScale === item.scale
                  ? 'border-accent bg-accent/10 ring-1 ring-accent'
                  : 'hover:border-slate-500'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono font-bold mb-2">
                <span className="text-slate-200">Scale σ = {item.scale}</span>
                {item.scale === 0.15 && (
                  <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-1 py-0.2 rounded font-semibold">
                    BEST
                  </span>
                )}
              </div>
              <div className="text-[11px] font-mono text-slate-400 space-y-1">
                <div>Acceptance: <strong className="text-slate-200">{item.acceptance}</strong></div>
                <div>Cond. Acc: <strong className="text-slate-200">{item.accuracy}</strong></div>
                <div>Diversity: <strong className="text-slate-200">{item.diversity}</strong></div>
              </div>
              <p className="text-[10px] text-slate-400 mt-2 pt-2 border-t border-border-subtle leading-tight">
                {item.desc}
              </p>
            </div>
          ))}
        </div>

        <StrategyArenaChart height={220} />
      </Card>

      {/* Critic Confidence Filtering Process */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card
          title="Critic Confidence Filtering"
          subtitle="Non-circular verification pipeline using independent baseline CNN"
        >
          <div className="space-y-3 text-xs font-mono">
            <div className="p-3 bg-surface-100 rounded border border-border-subtle flex items-start gap-3">
              <div className="w-6 h-6 rounded bg-blue-500/15 text-blue-400 flex items-center justify-center font-bold flex-shrink-0">
                1
              </div>
              <div>
                <div className="font-semibold text-slate-200">Raw Generation</div>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  Generator outputs batch of synthetic images conditioned on target class label.
                </div>
              </div>
            </div>

            <div className="p-3 bg-surface-100 rounded border border-border-subtle flex items-start gap-3">
              <div className="w-6 h-6 rounded bg-purple-500/15 text-purple-400 flex items-center justify-center font-bold flex-shrink-0">
                2
              </div>
              <div>
                <div className="font-semibold text-slate-200">Confidence Scoring</div>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  Independent Critic computes softmax probabilities P(y = c | x_synth).
                </div>
              </div>
            </div>

            <div className="p-3 bg-surface-100 rounded border border-border-subtle flex items-start gap-3">
              <div className="w-6 h-6 rounded bg-emerald-500/15 text-emerald-400 flex items-center justify-center font-bold flex-shrink-0">
                3
              </div>
              <div>
                <div className="font-semibold text-slate-200">Threshold Gate (Confidence ≥ 0.90)</div>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  Only high-fidelity samples pass into the augmented training partition; low-confidence outliers are pruned.
                </div>
              </div>
            </div>
          </div>
        </Card>

        <Card
          title="Synthetic Quality Metrics"
          subtitle="Statistical properties of generated minority samples"
        >
          <div className="grid grid-cols-2 gap-3 mb-4">
            <MetricPill
              label="Critic Acceptance"
              value="76.92%"
              subtitle="Threshold = 0.90"
              icon={CheckCircle2}
              highlight
            />
            <MetricPill
              label="Synthetic Accuracy"
              value="89.57%"
              subtitle="Conditional Match"
              icon={ShieldCheck}
            />
            <MetricPill
              label="Diversity Score"
              value="0.331"
              subtitle="Pairwise Distance"
              icon={Layers}
            />
            <MetricPill
              label="Duplicate Rate"
              value="0.00%"
              subtitle="Zero Memorization"
              icon={Cpu}
            />
          </div>
          <div className="p-3 tech-card-subtle rounded text-xs text-slate-300 font-mono">
            GenForge verifies that generated samples are diverse novel interpolations on the digit manifold rather than direct copy-pastes of the 100 anchor training instances.
          </div>
        </Card>
      </div>
    </div>
  );
}
