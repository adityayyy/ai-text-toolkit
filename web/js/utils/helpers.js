/**
 * Helper utilities
 */

export const HTML_ESCAPES = {
  '&': '&',
  '<': '<',
  '>': '>',
  '"': '"',
  "'": ''',
};

export function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, c => HTML_ESCAPES[c]);
}

export function formatAiScore(score) {
  return `${Number(score).toFixed(1)}%`;
}

export function formatElapsed(seconds) {
  return Number(seconds).toFixed(2);
}

export function getAiLabel(probability) {
  if (probability > 70) return 'AI-generated';
  if (probability < 30) return 'Likely human';
  return 'Uncertain';
}

export function getInterpretation(probability) {
  if (probability > 70) {
    return 'Reads like it was written by a model, not a person.';
  }
  if (probability < 30) {
    return 'Reads human to me.';
  }
  return 'Hard to say for sure — could go either way.';
}

export function getProviderName(provider) {
  const names = {
    gemini: 'Gemini',
    openai: 'OpenAI',
    nvidia: 'NVIDIA',
  };
  return names[provider] || provider;
}

export function getThemeIcon(theme) {
  return theme === 'dark' ? '☀️' : '🌙';
}

export function getThemeLabel(theme) {
  return theme === 'dark' ? 'Light' : 'Dark';
}

export const QUERY_SVG = `
  <svg viewBox="0 0 20 20" aria-hidden="true" class="g-query">
    <circle cx="10" cy="10" r="8.2"/>
    <path d="M7.8 7.6a2.2 2.2 0 1 1 3 2.05c-.7.28-.8.75-.8 1.35"/>
    <circle cx="10" cy="13.6" r="0.4" fill="currentColor" stroke="none"/>
  </svg>
`;

export const QUIET_SVG = `
  <svg viewBox="0 0 20 20" aria-hidden="true" class="g-quiet">
    <circle cx="10" cy="10" r="8.2"/>
  </svg>
`;

export const ICON_MOON = `
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M20 13.5A8 8 0 0 1 10.5 4 8 8 0 1 0 20 13.5Z"/>
  </svg>
`;

export const ICON_SUN = `
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <circle cx="12" cy="12" r="4.5"/>
    <path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>
  </svg>
`;

export const ICON_KEY = `
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M14.5 4.5a4.5 4.5 0 0 0-6.4 5.6L3 15.2V18h2.8l1-1v-1.8h1.8l1.2-1.2v-1.6l1.3-1.3a4.5 4.5 0 0 0 3.4-6.6Z"/>
    <circle cx="16" cy="8" r="1.1"/>
  </svg>
`;