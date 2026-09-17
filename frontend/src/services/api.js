/**
 * Centralized API service for GenForge AI FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function fetchJson(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorJson;
      try {
        errorJson = JSON.parse(errorText);
      } catch {
        errorJson = { message: errorText || `HTTP error ${response.status}` };
      }
      throw new Error(errorJson.detail || errorJson.message || `Request failed with status ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    console.warn(`[GenForge API] Error on ${endpoint}:`, err.message);
    throw err;
  }
}

export const apiService = {
  // System & Health
  async getRoot() {
    return fetchJson('/');
  },

  async getHealth() {
    return fetchJson('/health');
  },

  async getSystemStatus() {
    return fetchJson('/system');
  },

  // Experiments
  async getExperiments() {
    return fetchJson('/experiments');
  },

  async getExperiment(id) {
    return fetchJson(`/experiments/${id}`);
  },

  async getBestExperiment() {
    return fetchJson('/experiments/best/result');
  },

  async compareExperiments(id1, id2) {
    return fetchJson(`/experiments/compare/${id1}/${id2}`);
  },

  async startExperiment(config = {}) {
    return fetchJson('/experiment', {
      method: 'POST',
      body: JSON.stringify({
        max_iterations: config.max_iterations ?? 3,
        critic_threshold: config.critic_threshold ?? 0.90,
        development_class_5_limit: config.development_class_5_limit ?? 100,
      }),
    });
  },

  // Research Results
  async getImbalancedResearch() {
    return fetchJson('/research/imbalanced');
  },

  async getMultiSeedResearch() {
    return fetchJson('/research/multiseed');
  },
};
