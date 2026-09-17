import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { SystemProvider } from './context/SystemContext';
import { AppLayout } from './components/layout/AppLayout';

import { Dashboard } from './pages/Dashboard';
import { Experiments } from './pages/Experiments';
import { ExperimentDetail } from './pages/ExperimentDetail';
import { Generation } from './pages/Generation';
import { Evaluation } from './pages/Evaluation';
import { Optimizer } from './pages/Optimizer';
import { Dataset } from './pages/Dataset';
import { Research } from './pages/Research';
import { System } from './pages/System';

export default function App() {
  return (
    <SystemProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/experiments/:id" element={<ExperimentDetail />} />
            <Route path="/generation" element={<Generation />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/optimizer" element={<Optimizer />} />
            <Route path="/dataset" element={<Dataset />} />
            <Route path="/research" element={<Research />} />
            <Route path="/system" element={<System />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SystemProvider>
  );
}
