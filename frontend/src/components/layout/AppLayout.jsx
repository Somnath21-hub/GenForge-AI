import React, { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { LiveExperimentModal } from '../pipeline/LiveExperimentModal';

const ROUTE_TITLES = {
  '/': { title: 'Overview & Intelligence Pipeline', subtitle: 'Autonomous synthetic data generation & ML optimization overview' },
  '/experiments': { title: 'Experiment History & Registry', subtitle: 'Detailed record of autonomous optimization runs' },
  '/generation': { title: 'Synthetic Generation Strategies', subtitle: 'Prior distribution vs Latent Manifold geometry and Critic filtering' },
  '/evaluation': { title: 'Independent Holdout Evaluation', subtitle: 'Isolated 10,000-sample test set verification & Critic benchmark' },
  '/optimizer': { title: 'Autonomous Strategy Optimizer', subtitle: 'Statistical utility verification and LangGraph routing' },
  '/dataset': { title: 'Dataset Intelligence & Imbalance Analysis', subtitle: 'Minority-class detection, scarcity ratios, and augmentation planning' },
  '/research': { title: 'Empirical Research Findings', subtitle: 'Statistical confidence intervals under severe class imbalance' },
  '/system': { title: 'System Architecture & Checkpoints', subtitle: 'Hardware status, active models, and FastAPI endpoint catalog' },
};

export function AppLayout() {
  const location = useLocation();
  const [isRunModalOpen, setIsRunModalOpen] = useState(false);

  // Extract base route for title lookup (e.g. /experiments/1 -> /experiments)
  const pathname = location.pathname;
  let currentMeta = ROUTE_TITLES[pathname];
  if (!currentMeta) {
    if (pathname.startsWith('/experiments/')) {
      currentMeta = {
        title: 'Experiment Report',
        subtitle: 'In-depth breakdown of configuration, quality, and downstream metrics',
      };
    } else {
      currentMeta = { title: 'GenForge AI', subtitle: 'Autonomous ML Platform' };
    }
  }

  return (
    <div className="flex h-screen w-screen bg-background text-slate-100 overflow-hidden">
      {/* Persistent Left Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopBar
          title={currentMeta.title}
          subtitle={currentMeta.subtitle}
          onOpenRunModal={() => setIsRunModalOpen(true)}
        />

        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Global Run Experiment Modal */}
      <LiveExperimentModal
        isOpen={isRunModalOpen}
        onClose={() => setIsRunModalOpen(false)}
      />
    </div>
  );
}
