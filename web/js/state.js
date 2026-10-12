/**
 * State management - centralized application state
 */

import { getLocal, setLocal } from './utils/index.js';

// ─── State ───
const state = {
  // Theme
  theme: 'light', // 'light' | 'dark'
  
  // API configuration
  apiBaseUrl: 'http://localhost:8000',
  
  // API Key (in-memory only, never persisted)
  apiProvider: 'local', // 'local' | 'gemini' | 'openai' | 'nvidia'
  apiKey: '',
  
  // UI state
  isLoading: false,
  loadingText: 'Working…',
  error: null,
  
  // Results
  detectResult: null,
  humanizeResult: null,
  showLlmOpinion: false,
  llmOpinionChecked: false,
  
  // Input
  inputText: '',
  wordCount: 0,
};

// ─── Subscribers ───
const subscribers = new Map();

function notify(key, value) {
  const subs = subscribers.get(key) || [];
  subs.forEach(cb => cb(value));
}

export function getState(key) {
  if (key) return state[key];
  return { ...state };
}

export function setState(key, value) {
  state[key] = value;
  notify(key, value);
}

export function subscribe(key, callback) {
  if (!subscribers.has(key)) {
    subscribers.set(key, []);
  }
  subscribers.get(key).push(callback);
  
  // Return unsubscribe function
  return () => {
    const subs = subscribers.get(key) || [];
    const idx = subs.indexOf(callback);
    if (idx > -1) subs.splice(idx, 1);
  };
}

export function subscribeAll(callback) {
  const keys = Object.keys(state);
  const unsubs = keys.map(key => subscribe(key, callback));
  return () => unsubs.forEach(unsub => unsub());
}

// ─── Theme persistence ───
export function loadTheme() {
  try {
    const stored = localStorage.getItem('copydesk-theme');
    if (stored) {
      state.theme = stored;
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      state.theme = 'dark';
    }
  } catch {
    // file:// or blocked storage — default light
  }
  applyTheme(state.theme);
}

export function setTheme(theme) {
  state.theme = theme;
  applyTheme(theme);
  try {
    localStorage.setItem('copydesk-theme', theme);
  } catch {
    // non-fatal
  }
  notify('theme', theme);
}

export function toggleTheme() {
  setTheme(state.theme === 'dark' ? 'light' : 'dark');
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
}

// ─── API Key sync (in-memory only) ───
export function setApiProvider(provider) {
  state.apiProvider = provider;
  notify('apiProvider', provider);
}

export function setApiKey(key) {
  state.apiKey = key;
  notify('apiKey', key);
}

export function clearApiKey() {
  state.apiKey = '';
  state.apiProvider = 'local';
  notify('apiKey', '');
  notify('apiProvider', 'local');
}

// ─── Loading state ───
export function setLoading(show, text = 'Working…') {
  state.isLoading = show;
  state.loadingText = text;
  notify('isLoading', show);
  notify('loadingText', text);
}

// ─── Error state ───
export function setError(message) {
  state.error = message;
  notify('error', message);
}

export function clearError() {
  state.error = null;
  notify('error', null);
}

// ─── Input state ───
export function setInputText(text) {
  state.inputText = text;
  notify('inputText', text);
}

export function setWordCount(count) {
  state.wordCount = count;
  notify('wordCount', count);
}

// ─── Results state ───
export function setDetectResult(result) {
  state.detectResult = result;
  notify('detectResult', result);
}

export function setHumanizeResult(result) {
  state.humanizeResult = result;
  notify('humanizeResult', result);
}

export function clearResults() {
  state.detectResult = null;
  state.humanizeResult = null;
  state.showLlmOpinion = false;
  state.llmOpinionChecked = false;
  notify('detectResult', null);
  notify('humanizeResult', null);
}

export function setLlmOpinion(show, checked = false) {
  state.showLlmOpinion = show;
  state.llmOpinionChecked = checked;
  notify('showLlmOpinion', show);
  notify('llmOpinionChecked', checked);
}

// Initialize theme on load
loadTheme();