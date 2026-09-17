import React, { useState } from 'react';
import { Card } from '../components/common/Card';
import { MetricPill } from '../components/common/MetricPill';
import { ClassDistributionChart } from '../components/charts/ClassDistributionChart';
import { AugmentationPlanChart } from '../components/charts/AugmentationPlanChart';
import {
  Database,
  AlertTriangle,
  Layers,
  Activity,
  CheckCircle2,
  UploadCloud,
  FolderOpen,
  FileCode,
  FileCheck,
  Check,
} from 'lucide-react';

export function Dataset() {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
  };

  const processFiles = (fileList) => {
    const files = Array.from(fileList).map((f) => ({
      name: f.name,
      size: (f.size / (1024 * 1024)).toFixed(2) + ' MB',
      type: f.type || 'Dataset Archive/Array',
    }));
    setUploadedFiles(files);
    setUploadStatus('Ready for Ingestion & Imbalance Profiling');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-slate-100">Dataset Intelligence & Scarcity Analysis</h2>
        <p className="text-xs text-slate-400">
          Automated pre-generation audit of class balance, minority data scarcity, and dataset ingestion
        </p>
      </div>

      {/* Dataset Ingestion & Upload Zone Card */}
      <Card
        title="Dataset Ingestion & Upload Hub"
        subtitle="Where to place or upload your raw training partitions"
        badge={
          <span className="text-[11px] font-mono text-accent bg-accent/15 px-2 py-0.5 rounded border border-accent/30 flex items-center gap-1">
            <FolderOpen className="w-3 h-3" />
            Path: GenForge-AI/data/
          </span>
        }
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Drag-and-Drop Area */}
          <div className="lg:col-span-7">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleFileDrop}
              className={`p-6 border-2 border-dashed rounded-lg text-center transition-all ${
                isDragging
                  ? 'border-accent bg-accent/10'
                  : 'border-border-subtle bg-surface-100/60 hover:border-slate-500'
              }`}
            >
              <div className="w-12 h-12 rounded-full bg-accent/15 border border-accent/30 text-accent flex items-center justify-center mx-auto mb-3">
                <UploadCloud className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-semibold text-slate-200">
                Drag & Drop Dataset Files Here
              </h4>
              <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                Supports TorchVision datasets, image archives (<code className="text-accent font-mono">.zip</code>, <code className="text-accent font-mono">.tar.gz</code>), NumPy arrays (<code className="text-accent font-mono">.npy</code>), and tabular <code className="text-accent font-mono">.csv</code>.
              </p>

              <label className="inline-block mt-4">
                <span className="px-3.5 py-1.5 bg-accent hover:bg-accent-hover text-white rounded text-xs font-semibold cursor-pointer transition-colors shadow-sm font-mono">
                  Browse Local Files
                </span>
                <input
                  type="file"
                  multiple
                  onChange={handleFileInput}
                  className="hidden"
                  accept=".zip,.tar.gz,.tar,.csv,.npy,.png,.jpg,.jpeg,.pth"
                />
              </label>

              {uploadedFiles.length > 0 && (
                <div className="mt-4 text-left border-t border-border-subtle pt-3 space-y-1.5 font-mono text-xs">
                  <div className="text-[11px] text-emerald-400 font-semibold flex items-center gap-1">
                    <FileCheck className="w-3.5 h-3.5" />
                    {uploadStatus}
                  </div>
                  {uploadedFiles.map((file, idx) => (
                    <div
                      key={`file-${idx}`}
                      className="flex items-center justify-between p-2 bg-surface-200 rounded text-slate-300"
                    >
                      <span className="truncate max-w-[240px]">{file.name}</span>
                      <span className="text-slate-500 text-[11px]">{file.size}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right: Filesystem Location Guide */}
          <div className="lg:col-span-5 space-y-3 text-xs font-mono">
            <div className="p-3.5 tech-card-subtle rounded border border-border-subtle space-y-2">
              <div className="font-bold text-slate-200 flex items-center gap-1.5">
                <FolderOpen className="w-4 h-4 text-accent" />
                Local Filesystem Location
              </div>
              <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
                You can directly drop your dataset folders into the project directory at:
              </p>
              <div className="p-2 bg-surface-300 rounded text-emerald-400 border border-emerald-500/20 text-[11px] select-all">
                GenForge-AI/data/
              </div>
            </div>

            <div className="p-3.5 tech-card-subtle rounded border border-border-subtle space-y-1.5">
              <div className="font-bold text-slate-200 text-xs">Directory Structure Example:</div>
              <pre className="text-[11px] text-slate-400 leading-tight">
{`GenForge-AI/
├── data/
│   ├── MNIST/               <-- Active Benchmark
│   ├── custom_dataset/      <-- Custom Images
│   │   ├── train/
│   │   │   ├── class_0/
│   │   │   └── class_1/
│   │   └── test/            <-- Isolated Holdout
└── models/`}
              </pre>
            </div>
          </div>
        </div>
      </Card>

      {/* Metric Tiles */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <MetricPill
          label="Total Development Images"
          value="54,057"
          subtitle="Training Partition"
          icon={Database}
        />
        <MetricPill
          label="Isolated Test Set"
          value="10,000"
          subtitle="Untouched Holdout"
          icon={CheckCircle2}
        />
        <MetricPill
          label="Imbalance Ratio"
          value="67.42×"
          subtitle="Class 1 vs Class 5"
          highlight
          icon={AlertTriangle}
        />
        <MetricPill
          label="Minority Class (Digit 5)"
          value="100"
          subtitle="Real Samples Only"
          highlight
          icon={Activity}
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card
          title="Class Frequency Distribution"
          subtitle="Digit 5 constrained to 100 samples to benchmark data scarcity"
        >
          <ClassDistributionChart height={240} />
          <div className="p-3 tech-card-subtle rounded mt-4 text-xs font-mono text-slate-300">
            <span className="text-amber-400 font-bold">Scarcity Diagnosis: </span>
            Class 5 contains only 100 development samples, creating a 67.4× imbalance relative to majority class 1 (6,742 samples).
          </div>
        </Card>

        <Card
          title="Target Augmentation Plan"
          subtitle="Synthetic sample quota required to achieve uniform distribution"
        >
          <AugmentationPlanChart height={240} />
          <div className="p-3 tech-card-subtle rounded mt-4 text-xs font-mono text-slate-300">
            <span className="text-emerald-400 font-bold">Augmentation Quota: </span>
            GenForge allocates +6,642 synthetic samples to Class 5 and balances all classes to the 6,742 ceiling.
          </div>
        </Card>
      </div>

      {/* Dataset Partitions & Leakage Protection */}
      <Card
        title="Partitioning & Zero-Contamination Architecture"
        subtitle="How GenForge ensures scientific validity during synthetic experimentation"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-3.5 tech-card-subtle rounded border border-border-subtle">
            <div className="text-slate-400 font-bold text-sm text-slate-200">1. Real Train Split</div>
            <div className="text-slate-400 mt-1">54,057 samples</div>
            <p className="text-slate-500 text-[11px] mt-2 font-sans">
              Used for initial CVAE training, latent manifold anchor encoding, and baseline CNN training.
            </p>
          </div>

          <div className="p-3.5 tech-card-subtle rounded border border-accent/30 bg-accent/5">
            <div className="text-slate-400 font-bold text-sm text-accent">2. Synthetic Augmentation</div>
            <div className="text-slate-400 mt-1">13,361 planned samples</div>
            <p className="text-slate-400 text-[11px] mt-2 font-sans">
              Generated via Latent Manifold Resampling and filtered through the Critic before concatenation with the real training split.
            </p>
          </div>

          <div className="p-3.5 tech-card-subtle rounded border border-emerald-500/30 bg-emerald-500/5">
            <div className="text-slate-400 font-bold text-sm text-emerald-300">3. Test Holdout</div>
            <div className="text-slate-400 mt-1">10,000 samples (Isolated)</div>
            <p className="text-slate-400 text-[11px] mt-2 font-sans">
              Strictly isolated. Never accessed by any generative component, encoder, or critic filter.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
