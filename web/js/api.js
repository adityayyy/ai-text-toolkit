/**
 * API client - handles all backend communication
 */

import { getState, setState } from '../state.js';
import { escapeHtml } from '../utils/index.js';

const API_BASE = getState('apiBaseUrl');

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });
  
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      message = err.detail || err.message || message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }
  
  return response.json();
}

export async function detect(text) {
  return request('/detect', {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
}

export async function humanize(text, options = {}) {
  return request('/humanize', {
    method: 'POST',
    body: JSON.stringify({
      text,
      re_detect: true,
      temperature: options.temperature ?? 0.7,
      top_p: options.top_p ?? 0.9,
      max_tokens: options.max_tokens ?? 512,
    }),
  });
}

export async function humanizeExternal(text, provider, apiKey, options = {}) {
  return request('/humanize-external', {
    method: 'POST',
    body: JSON.stringify({
      text,
      provider,
      api_key: apiKey,
      temperature: options.temperature ?? 0.7,
      top_p: options.top_p ?? 0.9,
      max_tokens: options.max_tokens ?? 512,
    }),
  });
}

export async function healthCheck() {
  return request('/health');
}