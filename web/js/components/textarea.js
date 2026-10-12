/**
 * Textarea component - handles input text, word count, and LLM checkbox
 */

import { $ } from '../utils/dom.js';
import { getState, setState, subscribe } from '../state.js';
import { formatAiScore, getProviderName } from '../utils/helpers.js';
import { escapeHtml } from '../utils/index.js';

// DOM elements (lazy-initialized)
let elements = {};

function cacheElements() {
  elements = {
    inputText: $('#inputText'),
    wordCount: $('#wordCount'),
    btnDetect: $('#btnDetect'),
    btnHumanize: $('#btnHumanize'),
    llmSecondaryToggle: $('#llmSecondaryToggle'),
    llmAsk: $('#llmAsk'),
    llmAskLabel: $('#llmAskLabel'),
    llmAskProvider: $('#llmAskProvider'),
    errorBox: $('#errorBox'),
    loading: $('#loading'),
    loadingText: $('#loadingText'),
  };
}

function init() {
  cacheElements();
  bindEvents();
  updateWordCount();
}

function bindEvents() {
  // Textarea input
  elements.inputText.addEventListener('input', updateWordCount);
  
  // Buttons
  elements.btnDetect.addEventListener('click', handleDetect);
  elements.btnHumanize.addEventListener('click', handleHumanize);
  
  // LLM checkbox
  elements.llmAsk.addEventListener('change', () => {
    // State is managed by the results component
  });
}

function updateWordCount() {
  const text = elements.inputText.value.trim();
  const count = text ? text.split(/\s+/).length : 0;
  setState('inputText', text);
  setState('wordCount', count);
  elements.wordCount.textContent = `${count} word${count !== 1 ? 's' : ''}`;
  
  // Show/hide LLM checkbox based on API key state
  updateLlmCheckbox();
}

function updateLlmCheckbox() {
  const { apiProvider, apiKey } = getState();
  if (apiProvider !== 'local' && apiKey) {
    const providerName = getProviderName(apiProvider);
    elements.llmAskProvider.textContent = providerName;
    elements.llmAskLabel.innerHTML = `Also ask <b>${escapeHtml(providerName)}</b> for a second reading (uses your API key)`;
    elements.llmSecondaryToggle.style.display = 'block';
    elements.llmAsk.checked = false;
    setState('llmOpinionChecked', false);
  } else {
    elements.llmSecondaryToggle.style.display = 'none';
    elements.llmAsk.checked = false;
    setState('llmOpinionChecked', false);
  }
}

async function handleDetect() {
  const text = elements.inputText.value.trim();
  if (!text) {
    showError('Paste some text first — the desk cannot read an empty sheet.');
    return;
  }
  
  clearError();
  hideResults();
  setLoading(true, 'Reading the proof');
  
  try {
    const { detect } = await import('../api.js');
    const data = await detect(text);
    
    // Handle LLM opinion if checked
    let llm = null;
    if (elements.llmAsk.checked && getState('apiProvider') !== 'local' && getState('apiKey')) {
      try {
        const { humanizeExternal } = await import('../api.js');
        const prompt = `Analyze the following text and assess the likelihood it was written by an AI language model versus a human. Respond in this exact format:\nLIKELIHOOD: [0-100]\nREASONING: [one sentence]\n\nText: ${text}`;
        const raw = await humanizeExternal(prompt, getState('apiProvider'), getState('apiKey'));
        const match = raw.match(/LIKELIHOOD:\s*(\d{1,3})/i);
        const reasoning = (raw.match(/REASONING:\s*(.+)/i) || [null, null])[1];
        let likelihood = null;
        if (match) likelihood = Math.min(100, Math.max(0, parseInt(match[1], 10)));
        if (likelihood !== null) {
          llm = { provider: getState('apiProvider'), likelihood, reasoning: (reasoning || '').trim().slice(0, 300) };
        }
      } catch (e) {
        llm = { error: e.message, provider: getState('apiProvider') };
      }
    }
    
    // Import and call render function
    const { renderDetectResult } = await import('./results.js');
    renderDetectResult(data, llm);
    
  } catch (e) {
    showError(e.message);
  } finally {
    setLoading(false);
  }
}

async function handleHumanize() {
  const text = elements.inputText.value.trim();
  if (!text) {
    showError('Paste some text first — the desk cannot rewrite an empty sheet.');
    return;
  }
  
  clearError();
  hideResults();
  setLoading(true, 'Setting the human pass');
  
  try {
    const { humanize } = await import('../api.js');
    const data = await humanize(text);
    const { renderHumanizeResult } = await import('./results.js');
    renderHumanizeResult(data);
  } catch (e) {
    showError(e.message || 'Humanize failed — check your API key and try again');
  } finally {
    setLoading(false);
  }
}

function showError(message) {
  elements.errorBox.textContent = message;
  elements.errorBox.style.display = 'block';
  setState('error', message);
}

function clearError() {
  elements.errorBox.textContent = '';
  elements.errorBox.style.display = 'none';
  setState('error', null);
}

function setLoading(show, text = 'Working…') {
  elements.loadingText.textContent = text;
  elements.loading.style.display = show ? 'block' : 'none';
  elements.btnDetect.disabled = show;
  elements.btnHumanize.disabled = show;
  setState('isLoading', show);
  setState('loadingText', text);
}

function hideResults() {
  document.getElementById('resultsDetect').classList.remove('visible');
  document.getElementById('resultsHumanize').classList.remove('visible');
  document.getElementById('llmOpinion').style.display = 'none';
  setState('detectResult', null);
  setState('humanizeResult', null);
}

// Subscribe to API key changes to update UI
subscribe('apiProvider', updateLlmCheckbox);
subscribe('apiKey', updateLlmCheckbox);

// Export init function
export { init, updateWordCount, updateLlmCheckbox };