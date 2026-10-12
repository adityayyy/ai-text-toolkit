/**
 * Main application entry point
 * Initializes all components and sets up the app
 */

import { init as initTextarea } from './components/textarea.js';
import { init as initResults } from './components/results.js';
import { init as initApiKey } from './components/api-key.js';
import { init as initTheme } from './components/theme.js';
import { getState } from '../state.js';

// Initialize all components
function init() {
  // Initialize theme first (affects rendering)
  initTheme();
  
  // Initialize components
  initTextarea();
  initResults();
  initApiKey();
  
  // Show API base in footer for debugging
  showApiBase();
  
  console.log('[ai-text-toolkit] Initialized');
  console.log('[ai-text-toolkit] API Base:', getState('apiBaseUrl'));
  console.log('[ai-text-toolkit] Theme:', getState('theme'));
}

function showApiBase() {
  const footer = document.querySelector('.colophon');
  const apiBase = getState('apiBaseUrl');
  if (footer && apiBase) {
    const badge = document.createElement('span');
    badge.className = 'api-badge';
    badge.textContent = `API: ${apiBase}`;
    footer.appendChild(badge);
  }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}