/**
 * API Key component - handles the dropdown panel for API key configuration
 */

import { $ } from '../utils/dom.js';
import { getState, setState, subscribe } from '../state.js';
import { getProviderName } from '../utils/helpers.js';

// DOM elements
let elements = {};

function cacheElements() {
  elements = {
    apikeyToggle: $('#apikeyToggle'),
    apikeyPanel: $('#apikeyPanel'),
    apiProviderSelect: $('#apiProviderSelect'),
    apiKeyInput: $('#apiKeyInput'),
    apikeyFieldRow: $('#apikeyFieldRow'),
    apikeyClear: $('#apikeyClear'),
  };
}

function init() {
  cacheElements();
  bindEvents();
  syncFromState();
}

function bindEvents() {
  // Toggle panel
  elements.apikeyToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    togglePanel();
  });
  
  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!elements.apikeyPanel.hidden && 
        !elements.apikeyPanel.contains(e.target) && 
        e.target !== elements.apikeyToggle) {
      closePanel();
    }
  });
  
  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !elements.apikeyPanel.hidden) {
      closePanel();
    }
  });
  
  // Provider select
  elements.apiProviderSelect.addEventListener('change', () => {
    syncToState();
    updateFieldVisibility();
  });
  
  // API key input
  elements.apiKeyInput.addEventListener('input', syncToState);
  
  // Clear button
  elements.apikeyClear.addEventListener('click', () => {
    elements.apiKeyInput.value = '';
    syncToState();
  });
}

function togglePanel() {
  elements.apikeyPanel.hidden = !elements.apikeyPanel.hidden;
  elements.apikeyToggle.setAttribute('aria-expanded', String(!elements.apikeyPanel.hidden));
}

function closePanel() {
  elements.apikeyPanel.hidden = true;
  elements.apikeyToggle.setAttribute('aria-expanded', 'false');
}

function syncToState() {
  setState('apiProvider', elements.apiProviderSelect.value);
  setState('apiKey', elements.apiKeyInput.value);
  updateFieldVisibility();
}

function syncFromState() {
  elements.apiProviderSelect.value = getState('apiProvider');
  elements.apiKeyInput.value = getState('apiKey');
  updateFieldVisibility();
}

function updateFieldVisibility() {
  const isLocal = getState('apiProvider') === 'local';
  elements.apikeyFieldRow.hidden = isLocal;
}

// Subscribe to external state changes
subscribe('apiProvider', () => {
  elements.apiProviderSelect.value = getState('apiProvider');
  updateFieldVisibility();
});

subscribe('apiKey', () => {
  elements.apiKeyInput.value = getState('apiKey');
});

// Initialize
init();

export { init, togglePanel, closePanel };