"use strict";

(() => {
  try {
    const preference = window.localStorage.getItem("kb-pente-monitor-theme");
    if (preference === "light" || preference === "dark") {
      document.documentElement.dataset.theme = preference;
    }
  } catch {
    // System theme remains active when browser storage is unavailable.
  }
})();
