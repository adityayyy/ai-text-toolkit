/**
 * Theme component - handles light/dark mode toggle
 */

import { $ } from '../utils/dom.js';
import { getState, setState } from '../state.js';
import { ICON_MOON, ICON_SUN, getThemeLabel } from '../utils/helpers.js';

// DOM elements
let elements = {};

function cacheElements() {
  elements = {
    themeToggle: $('#themeToggle'),
    iconMoon: $('#iconMoon'),
    iconSun: $('#iconSun'),
    themeLabel: $('#themeLabel'),
  };
}

function init() {
  cacheElements();
  applyTheme(getState('theme'));
  bindEvents();
}

function bindEvents() {
  elements.themeToggle.addEventListener('click', () => {
    const next = getState('theme') === 'dark' ? 'light' : 'dark';
    setTheme(next);
  });
}

function setTheme(theme) {
  setState('theme', theme);
  applyTheme(theme);
  try {
    localStorage.setItem('copydesk-theme', theme);
  } catch {
    // non-fatal
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  elements.iconMoon.style.display = theme === 'dark' ? 'none' : '';
  elements.iconSun.style.display = theme === 'dark' ? '' : 'none';
  elements.themeLabel.textContent = getThemeLabel(theme);
}

// Initialize
init();

export { init, setTheme };