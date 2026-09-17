import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

const SystemContext = createContext(null);

export function SystemProvider({ children }) {
  const [systemInfo, setSystemInfo] = useState({
    status: 'checking',
    version: '1.0.0',
    gpu_name: 'NVIDIA GeForce RTX 2050',
    cuda_available: true,
    pytorch_version: '2.x+cu12x',
    checkpoints: {},
  });
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [lastPing, setLastPing] = useState(null);
  const [isRunningExperiment, setIsRunningExperiment] = useState(false);
  const [activeExperimentConfig, setActiveExperimentConfig] = useState(null);
  const [experimentProgressStage, setExperimentProgressStage] = useState('idle');

  const checkHealth = useCallback(async () => {
    try {
      const health = await apiService.getHealth();
      if (health && health.status === 'healthy') {
        setIsBackendConnected(true);
        setLastPing(new Date());

        try {
          const sys = await apiService.getSystemStatus();
          setSystemInfo((prev) => ({
            ...prev,
            ...sys,
            status: 'online',
          }));
        } catch {
          // If /system endpoint fails, basic connection is still valid
          setSystemInfo((prev) => ({ ...prev, status: 'online' }));
        }
      } else {
        setIsBackendConnected(false);
        setSystemInfo((prev) => ({ ...prev, status: 'offline' }));
      }
    } catch {
      setIsBackendConnected(false);
      setSystemInfo((prev) => ({ ...prev, status: 'offline' }));
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const launchExperiment = async (config) => {
    setIsRunningExperiment(true);
    setActiveExperimentConfig(config);
    setExperimentProgressStage('dataset');

    try {
      const res = await apiService.startExperiment(config);

      // Simulate step transitions based on typical pipeline execution
      const stages = ['dataset', 'analyze', 'plan', 'generate', 'critic', 'evaluate', 'optimize', 'completed'];
      let currentIdx = 0;
      const progressTimer = setInterval(() => {
        currentIdx += 1;
        if (currentIdx < stages.length) {
          setExperimentProgressStage(stages[currentIdx]);
        } else {
          clearInterval(progressTimer);
          setIsRunningExperiment(false);
          setExperimentProgressStage('completed');
          checkHealth();
        }
      }, 3500);

      return res;
    } catch (err) {
      setIsRunningExperiment(false);
      setExperimentProgressStage('error');
      throw err;
    }
  };

  return (
    <SystemContext.Provider
      value={{
        systemInfo,
        isBackendConnected,
        lastPing,
        checkHealth,
        isRunningExperiment,
        activeExperimentConfig,
        experimentProgressStage,
        launchExperiment,
      }}
    >
      {children}
    </SystemContext.Provider>
  );
}

export function useSystem() {
  const context = useContext(SystemContext);
  if (!context) {
    throw new Error('useSystem must be used within a SystemProvider');
  }
  return context;
}
