/**
 * DOM utilities
 */

export function $(selector, root = document) {
  return root.querySelector(selector);
}

export function $$(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

export function createElement(tag, attributes = {}, ...children) {
  const el = document.createElement(tag);
  Object.entries(attributes).forEach(([key, value]) => {
    if (key === 'class') {
      el.className = value;
    } else if (key === 'style' && typeof value === 'object') {
      Object.assign(el.style, value);
    } else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      el.setAttribute(key, value);
    }
  });
  children.flat().forEach(child => {
    if (typeof child === 'string') {
      el.appendChild(document.createTextNode(child));
    } else if (child instanceof Node) {
      el.appendChild(child);
    }
  });
  return el;
}

export function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

export function setTextContent(element, text) {
  element.textContent = text;
}

export function toggleClass(element, className, force) {
  return element.classList.toggle(className, force);
}

export function addClass(element, ...classNames) {
  element.classList.add(...classNames);
}

export function removeClass(element, ...classNames) {
  element.classList.remove(...classNames);
}

export function hasClass(element, className) {
  return element.classList.contains(className);
}

/**
 * Animation utilities
 */
export function animate(element, keyframes, options) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return Promise.resolve();
  }
  return element.animate(keyframes, options).finished;
}

export function stagger(elements, baseDelay = 35, maxItems = 12) {
  elements.forEach((el, i) => {
    const delay = Math.min(i, maxItems) * baseDelay;
    el.style.animationDelay = `${delay}ms`;
  });
}

/**
 * Event utilities
 */
export function on(event, selector, handler, root = document) {
  root.addEventListener(event, e => {
    const target = e.target.closest(selector);
    if (target && (root === document || root.contains(target))) {
      handler.call(target, e);
    }
  });
}

export function once(event, selector, handler, root = document) {
  const wrapped = e => {
    const target = e.target.closest(selector);
    if (target && (root === document || root.contains(target))) {
      root.removeEventListener(event, wrapped);
      handler.call(target, e);
    }
  };
  root.addEventListener(event, wrapped);
}

/**
 * Local storage utilities
 */
export function getLocal(key, fallback = null) {
  try {
    const value = localStorage.getItem(key);
    return value !== null ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

export function setLocal(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function removeLocal(key) {
  try {
    localStorage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}

/**
 * Debounce utility
 */
export function debounce(fn, delay) {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Format utilities
 */
export function formatNumber(num, decimals = 1) {
  return Number(num).toFixed(decimals);
}

export function pluralize(count, singular, plural = null) {
  return count === 1 ? singular : (plural || singular + 's');
}