export const PIPELINE_STAGES = [
  { id: 'dataset', label: 'Dataset', desc: 'MNIST Development Partition' },
  { id: 'analyze', label: 'Analyze', desc: 'Imbalance & Weakness Detection' },
  { id: 'plan', label: 'Plan', desc: 'Target Synthetic Distribution' },
  { id: 'generate', label: 'Generate', desc: 'CVAE Latent Manifold Resampling' },
  { id: 'critic', label: 'Critic', desc: 'Confidence & Quality Filtering' },
  { id: 'evaluate', label: 'Evaluate', desc: 'Independent Holdout Validation' },
  { id: 'optimize', label: 'Optimize', desc: 'Statistical Decision Routing' },
];

export const DECISION_CONFIG = {
  KEEP: {
    label: 'KEEP',
    bg: 'bg-emerald-500/15',
    text: 'text-emerald-400',
    border: 'border-emerald-500/30',
    dot: 'bg-emerald-400',
    description: 'Statistically significant positive downstream utility',
  },
  RETRAIN: {
    label: 'RETRAIN',
    bg: 'bg-blue-500/15',
    text: 'text-blue-400',
    border: 'border-blue-500/30',
    dot: 'bg-blue-400',
    description: 'Reconstructive failure; model retrained with tuned hyperparameters',
  },
  ADAPT_LATENT: {
    label: 'ADAPT_LATENT',
    bg: 'bg-purple-500/15',
    text: 'text-purple-400',
    border: 'border-purple-500/30',
    dot: 'bg-purple-400',
    description: 'Empirical class-conditioned covariance shift adaptation',
  },
  RESAMPLE: {
    label: 'RESAMPLE',
    bg: 'bg-amber-500/15',
    text: 'text-amber-400',
    border: 'border-amber-500/30',
    dot: 'bg-amber-400',
    description: 'Perturbation scale adjusted around manifold anchors',
  },
  STOP: {
    label: 'STOP',
    bg: 'bg-slate-500/15',
    text: 'text-slate-400',
    border: 'border-slate-500/30',
    dot: 'bg-slate-400',
    description: 'Max iterations reached or no viable strategy identified',
  },
  UNCERTAIN: {
    label: 'UNCERTAIN',
    bg: 'bg-amber-500/15',
    text: 'text-amber-300',
    border: 'border-amber-500/30',
    dot: 'bg-amber-300',
    description: '95% Confidence Interval crosses zero (near-ceiling noise)',
  }
};
