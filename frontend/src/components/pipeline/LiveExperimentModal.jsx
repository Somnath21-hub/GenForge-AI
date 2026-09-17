import React, { useState } from 'react';
import { useSystem } from '../../context/SystemContext';
import { X, Play, Loader2, CheckCircle2, AlertCircle, Cpu, Sliders, Database } from 'lucide-react';
import { PipelineVisualizer } from './PipelineVisualizer';

export function LiveExperimentModal({ isOpen, onClose }) {
  const { launchExperiment, isRunningExperiment, experimentProgressStage } = useSystem();

  const [config, setConfig] = useState({
    dataset: 'MNIST (Isolated Holdout Protocol)',
    generator: 'Conditional VAE (Latent Size 32)',
    strategy: 'LATENT_MANIFOLD',
    critic_threshold: 0.90,
    max_iterations: 3,
    development_class_5_limit: 100,
  });

  const [statusMessage, setStatusMessage] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  if (!isOpen) return null;

  const handleStart = async () => {
    setErrorMessage(null);
    setStatusMessage('Initiating LangGraph autonomous execution pipeline...');
    try {
      await launchExperiment({
        max_iterations: Number(config.max_iterations),
        critic_threshold: Number(config.critic_threshold),
        development_class_5_limit: Number(config.development_class_5_limit),
      });
      setStatusMessage('Autonomous experiment completed successfully.');
    } catch (err) {
      setErrorMessage(err.message || 'Failed to start experiment. Check backend.');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
      <div className="bg-surface-200 border border-border rounded-lg shadow-2xl w-full max-w-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-surface-300">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded bg-accent/15 border border-accent/30 flex items-center justify-center text-accent">
              <Play className="w-4 h-4 fill-current" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-slate-100">Run New ML Experiment</h3>
              <p className="text-xs text-slate-400">Configure parameters for the autonomous GenForge pipeline</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isRunningExperiment}
            className="text-slate-400 hover:text-slate-200 p-1 rounded hover:bg-slate-800 disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 overflow-y-auto space-y-5 text-xs">
          {/* Configuration Form */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block font-medium text-slate-300 mb-1.5 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-slate-400" />
                Target Dataset
              </label>
              <input
                type="text"
                disabled
                value={config.dataset}
                className="w-full px-3 py-2 bg-surface-100 border border-border-subtle rounded text-slate-300 font-mono text-xs cursor-not-allowed"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">
                10,000 holdout test samples remain strictly isolated
              </span>
            </div>

            <div>
              <label className="block font-medium text-slate-300 mb-1.5 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
                Generator Architecture
              </label>
              <input
                type="text"
                disabled
                value={config.generator}
                className="w-full px-3 py-2 bg-surface-100 border border-border-subtle rounded text-slate-300 font-mono text-xs cursor-not-allowed"
              />
              <span className="text-[10px] text-slate-500 mt-1 block">
                Checkpoint: models/conditional_vae.pth
              </span>
            </div>

            <div>
              <label className="block font-medium text-slate-300 mb-1.5 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-slate-400" />
                Critic Acceptance Threshold
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0.75"
                  max="0.98"
                  step="0.01"
                  value={config.critic_threshold}
                  disabled={isRunningExperiment}
                  onChange={(e) => setConfig({ ...config, critic_threshold: parseFloat(e.target.value) })}
                  className="flex-1 accent-accent"
                />
                <span className="font-mono text-slate-200 font-semibold bg-surface-100 px-2 py-1 border border-border-subtle rounded">
                  {(config.critic_threshold * 100).toFixed(0)}%
                </span>
              </div>
              <span className="text-[10px] text-slate-500 mt-1 block">
                Samples with confidence below this threshold are rejected
              </span>
            </div>

            <div>
              <label className="block font-medium text-slate-300 mb-1.5">
                Minority Class 5 Real Sample Limit
              </label>
              <select
                value={config.development_class_5_limit}
                disabled={isRunningExperiment}
                onChange={(e) => setConfig({ ...config, development_class_5_limit: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-surface-100 border border-border-subtle rounded text-slate-200 font-mono text-xs"
              >
                <option value={100}>100 samples (Severe Imbalance 67.4x)</option>
                <option value={500}>500 samples (Moderate Imbalance 13.5x)</option>
                <option value={1000}>1000 samples (Standard Development Partition)</option>
              </select>
              <span className="text-[10px] text-slate-500 mt-1 block">
                Tests generator efficacy under severe data scarcity
              </span>
            </div>

            <div>
              <label className="block font-medium text-slate-300 mb-1.5">
                Maximum LangGraph Iterations
              </label>
              <input
                type="number"
                min="1"
                max="5"
                value={config.max_iterations}
                disabled={isRunningExperiment}
                onChange={(e) => setConfig({ ...config, max_iterations: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-surface-100 border border-border-subtle rounded text-slate-200 font-mono text-xs"
              />
            </div>

            <div>
              <label className="block font-medium text-slate-300 mb-1.5">
                Generation Strategy Selection
              </label>
              <select
                value={config.strategy}
                disabled={isRunningExperiment}
                onChange={(e) => setConfig({ ...config, strategy: e.target.value })}
                className="w-full px-3 py-2 bg-surface-100 border border-border-subtle rounded text-slate-200 font-mono text-xs"
              >
                <option value="LATENT_MANIFOLD">LATENT_MANIFOLD (scale 0.15 - Recommended)</option>
                <option value="ADAPTIVE_LATENT">ADAPTIVE_LATENT (scale 0.80)</option>
                <option value="RANDOM_PRIOR">RANDOM_PRIOR N(0, I) Baseline</option>
              </select>
            </div>
          </div>

          {/* Live Progress Section */}
          {(isRunningExperiment || experimentProgressStage !== 'idle') && (
            <div className="p-4 bg-surface-300 border border-border-subtle rounded-lg space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200 flex items-center gap-2">
                  {isRunningExperiment ? (
                    <Loader2 className="w-4 h-4 text-accent animate-spin" />
                  ) : experimentProgressStage === 'completed' ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-amber-400" />
                  )}
                  Pipeline Execution: <span className="font-mono uppercase text-accent">{experimentProgressStage}</span>
                </span>
                <span className="text-slate-400 text-[11px] font-mono">
                  {isRunningExperiment ? 'Active Background Process' : 'Finished'}
                </span>
              </div>

              <PipelineVisualizer currentStage={experimentProgressStage} compact />

              {statusMessage && (
                <div className="p-2.5 bg-surface-100 rounded text-[11px] font-mono text-slate-300 border border-border-subtle">
                  &gt; {statusMessage}
                </div>
              )}

              {errorMessage && (
                <div className="p-2.5 bg-rose-500/10 text-rose-300 rounded text-[11px] font-mono border border-rose-500/30">
                  Error: {errorMessage}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3.5 border-t border-border bg-surface-300 flex items-center justify-between">
          <div className="text-[11px] text-slate-500 font-mono">
            Engine: PyTorch CUDA (RTX 2050)
          </div>
          <div className="flex items-center gap-2.5">
            <button
              onClick={onClose}
              disabled={isRunningExperiment}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded transition-colors disabled:opacity-50"
            >
              Close
            </button>
            <button
              onClick={handleStart}
              disabled={isRunningExperiment}
              className="px-4 py-2 bg-accent hover:bg-accent-hover text-white text-xs font-medium rounded flex items-center gap-2 transition-colors disabled:opacity-50 glow-blue"
            >
              {isRunningExperiment ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Executing Pipeline...
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  Start Experiment
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
