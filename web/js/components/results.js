/**
 * Results component - renders detection and humanization results
 */

import { $, $$ } from '../utils/dom.js';
import { getState, subscribe } from '../state.js';
import { formatAiScore, getAiLabel, getInterpretation, formatElapsed, escapeHtml, QUERY_SVG, QUIET_SVG } from '../utils/helpers.js';

// DOM elements (lazy-initialized)
let elements = {};

function cacheElements() {
  elements = {
    resultsDetect: $('#resultsDetect'),
    detectScore: $('#detectScore'),
    detectLabel: $('#detectLabel'),
    detectInterpretation: $('#detectInterpretation'),
    detectEmptyState: $('#detectEmptyState'),
    featuresEl: $('#features'),
    sentencesEl: $('#sentences'),
    llmOpinion: $('#llmOpinion'),
    llmOpinionTitle: $('#llmOpinionTitle'),
    llmOpinionScore: $('#llmOpinionScore'),
    llmOpinionError: $('#llmOpinionError'),
    llmOpinionReasoning: $('#llmOpinionReasoning'),
    resultsHumanize: $('#resultsHumanize'),
    humanizedText: $('#humanizedText'),
    origScore: $('#origScore'),
    newScore: $('#newScore'),
    delta: $('#delta'),
    backendUsed: $('#backendUsed'),
    hzTime: $('#hzTime'),
    humanizeViaBadge: $('#humanizeViaBadge'),
  };
}

function init() {
  cacheElements();
}

function renderDetectResult(data, llm = null) {
  if (!elements.detectScore) cacheElements();
  
  const p = data.ai_probability;
  elements.detectScore.textContent = formatAiScore(p);
  elements.detectScore.className = `verdict-num ${data.is_ai ? 'verdict-num--ai' : 'verdict-num--human'}`;
  elements.detectLabel.textContent = data.label;
  elements.detectLabel.className = `verdict-tag ${data.is_ai ? 'verdict-tag--ai' : 'verdict-tag--human'}`;

  // Interpretation
  elements.detectInterpretation.textContent = getInterpretation(p);
  elements.detectInterpretation.style.display = 'block';
  elements.detectEmptyState.style.display = 'none';

  // Signals table
  renderSignals(data.feature_breakdown);

  // Sentence marks
  renderSentences(data.sentence_scores);

  // Second reader (opt-in)
  renderLlmOpinion(llm);

  // Show detect results
  elements.resultsDetect.classList.add('visible');
  elements.resultsHumanize.classList.remove('visible');
}

function renderSignals(featureBreakdown) {
  const featureOrder = [
    ['perplexity', 'Perplexity'],
    ['burstiness', 'Burstiness'],
    ['avg_sentence_length', 'Avg Sentence Length'],
    ['sentence_length_var', 'Sentence Length Variance'],
    ['type_token_ratio', 'Type-Token Ratio'],
    ['punctuation_ratio', 'Punctuation Ratio'],
    ['bigram_uniqueness', 'Bigram Uniqueness'],
    ['avg_word_length', 'Avg Word Length'],
    ['pos_noun', 'POS: Nouns'],
    ['pos_verb', 'POS: Verbs'],
    ['pos_adj', 'POS: Adjectives'],
    ['pos_adv', 'POS: Adverbs'],
    ['pos_pron', 'POS: Pronouns'],
    ['pos_other', 'POS: Other'],
  ];

  elements.featuresEl.innerHTML = '';
  for (const [key, label] of featureOrder) {
    if (featureBreakdown[key] !== undefined) {
      const row = document.createElement('div');
      row.className = 'signal-row';
      row.innerHTML = `<span class="signal-name">${label}</span><span class="signal-value">${Number(featureBreakdown[key]).toFixed(key.includes('ratio') || key.includes('pos_') ? 4 : 2)}</span>`;
      elements.featuresEl.appendChild(row);
    }
  }
}

function renderSentences(sentenceScores) {
  elements.sentencesEl.innerHTML = '';
  
  sentenceScores.forEach((s, i) => {
    const flagged = s.ai_score >= 70;
    const item = document.createElement('div');
    item.className = `mark-row${flagged ? ' mark-row--flagged' : ''}`;
    item.style.animationDelay = `${Math.min(i, 12) * 35}ms`;
    item.innerHTML =
      `<span class="mark-glyph">${flagged ? QUERY_SVG : QUIET_SVG}</span>` +
      `<span class="mark-text">${escapeHtml(s.sentence)}</span>` +
      `<span class="mark-pct">${formatAiScore(s.ai_score)}</span>`;
    elements.sentencesEl.appendChild(item);
  });
}

function renderLlmOpinion(llm) {
  if (llm) {
    elements.llmOpinionTitle.textContent = 'Second reader — via ' + (llm.provider || 'LLM');
    if (llm.error) {
      elements.llmOpinion.classList.add('is-error');
      elements.llmOpinionScore.textContent = '—';
      elements.llmOpinionError.style.display = 'block';
      elements.llmOpinionError.textContent = llm.error;
      elements.llmOpinionReasoning.textContent = '';
    } else {
      elements.llmOpinion.classList.remove('is-error');
      elements.llmOpinionScore.textContent = formatAiScore(llm.likelihood);
      elements.llmOpinionReasoning.textContent = llm.reasoning || '(no reasoning provided)';
      elements.llmOpinionError.style.display = 'none';
      elements.llmOpinionError.textContent = '';
    }
    elements.llmOpinion.style.display = 'block';
  } else {
    elements.llmOpinion.style.display = 'none';
  }
}

function renderHumanizeResult(data) {
  if (!elements.humanizedText) cacheElements();
  
  elements.humanizedText.textContent = data.humanized_text;
  elements.origScore.textContent = formatAiScore(data.original_ai_probability);
  
  if (data.humanized_ai_probability !== null && data.humanized_ai_probability !== undefined) {
    elements.newScore.textContent = formatAiScore(data.humanized_ai_probability);
    const delta = data.humanized_ai_probability - data.original_ai_probability;
    elements.delta.textContent = `${delta >= 0 ? '+' : ''}${formatAiScore(delta)}`;
    elements.delta.className = 'setpass-delta ' + (delta < 0 ? 'negative' : 'positive');
  } else {
    elements.newScore.textContent = '—%';
    elements.delta.textContent = '';
  }
  
  elements.backendUsed.textContent = data.backend_used;
  elements.hzTime.textContent = formatElapsed(data.elapsed_seconds);

  const badge = elements.humanizeViaBadge;
  if (badge) {
    badge.textContent = 'via ' + (data.backend_used || 'local');
    badge.style.display = 'inline-block';
  }

  elements.resultsHumanize.classList.add('visible');
  elements.resultsDetect.classList.remove('visible');
}

// Initialize on module load
init();

// Export for external use
export { renderDetectResult, renderHumanizeResult };