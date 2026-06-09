(function () {
  const sheets = Array.from(document.querySelectorAll(".hero, main > .section"));

  if (!sheets.length) {
    return;
  }

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  sheets.forEach((sheet, index) => {
    sheet.classList.add("page-sheet");
    sheet.style.setProperty("--sheet-index", index);
    if (prefersReducedMotion) {
      sheet.classList.add("is-visible");
    }
  });

  if (prefersReducedMotion || !("IntersectionObserver" in window)) {
    sheets.forEach((sheet) => sheet.classList.add("is-visible"));
    return;
  }

  let viewportFrame = 0;
  const markViewportSheets = () => {
    viewportFrame = 0;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
    let currentSheet = null;

    sheets.forEach((sheet) => {
      const rect = sheet.getBoundingClientRect();
      const isNearViewport = rect.top < viewportHeight * 0.82 && rect.bottom > viewportHeight * 0.14;
      const crossesFocusLine = rect.top <= viewportHeight * 0.46 && rect.bottom >= viewportHeight * 0.46;

      if (isNearViewport) {
        sheet.classList.add("is-visible");
      }

      if (crossesFocusLine) {
        currentSheet = sheet;
      }
    });

    if (currentSheet) {
      sheets.forEach((sheet) => sheet.classList.toggle("is-current", sheet === currentSheet));
    }
  };

  const queueViewportCheck = () => {
    if (viewportFrame) {
      return;
    }
    viewportFrame = requestAnimationFrame(markViewportSheets);
  };

  const scrollPaddingTop = () => {
    const value = window.getComputedStyle(document.documentElement).scrollPaddingTop;
    return Number.parseFloat(value) || 0;
  };

  const currentSheetIndex = () => {
    const targetTop = scrollPaddingTop();
    let currentIndex = 0;
    let closestDistance = Number.POSITIVE_INFINITY;

    sheets.forEach((sheet, index) => {
      const distance = Math.abs(sheet.getBoundingClientRect().top - targetTop);

      if (distance < closestDistance) {
        closestDistance = distance;
        currentIndex = index;
      }
    });

    return currentIndex;
  };

  const canScrollWithinSheet = (sheet, deltaY) => {
    if (!sheet || sheet.scrollHeight <= sheet.clientHeight + 24) {
      return false;
    }

    if (deltaY > 0) {
      return sheet.scrollTop < sheet.scrollHeight - sheet.clientHeight - 2;
    }

    if (deltaY < 0) {
      return sheet.scrollTop > 2;
    }

    return false;
  };

  const isMediaControlTarget = (target) => (
    target && target.closest && target.closest("video, audio, .video-frame")
  );

  let pageScrollLock = false;
  const pageToSheet = (index) => {
    const targetIndex = Math.max(0, Math.min(sheets.length - 1, index));
    const currentIndex = currentSheetIndex();

    if (targetIndex === currentIndex) {
      pageScrollLock = false;
      return;
    }

    sheets[targetIndex].classList.add("is-visible");
    sheets[targetIndex].scrollIntoView({ block: "start", behavior: "smooth" });
    window.setTimeout(() => {
      pageScrollLock = false;
      queueViewportCheck();
    }, 760);
  };

  const pageByDelta = (deltaY) => {
    const direction = deltaY > 0 ? 1 : -1;
    pageToSheet(currentSheetIndex() + direction);
  };

  window.addEventListener(
    "wheel",
    (event) => {
      if (isMediaControlTarget(event.target)) {
        return;
      }

      if (event.ctrlKey || Math.abs(event.deltaY) < Math.abs(event.deltaX) || Math.abs(event.deltaY) < 8) {
        return;
      }

      const sheet = event.target.closest ? event.target.closest(".page-sheet") : null;

      if (canScrollWithinSheet(sheet, event.deltaY)) {
        return;
      }

      event.preventDefault();

      if (pageScrollLock) {
        return;
      }

      pageScrollLock = true;
      pageByDelta(event.deltaY);
    },
    { passive: false }
  );

  window.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }

    const activeElement = document.activeElement;
    const tagName = activeElement && activeElement.tagName;

    if (
      activeElement
      && (
        activeElement.isContentEditable
        || ["AUDIO", "INPUT", "SELECT", "TEXTAREA", "VIDEO"].includes(tagName)
        || isMediaControlTarget(activeElement)
      )
    ) {
      return;
    }

    const nextKeys = ["ArrowDown", "PageDown", " "];
    const previousKeys = ["ArrowUp", "PageUp"];

    if (!nextKeys.includes(event.key) && !previousKeys.includes(event.key)) {
      return;
    }

    const sheet = activeElement && activeElement.closest ? activeElement.closest(".page-sheet") : null;
    const deltaY = nextKeys.includes(event.key) ? 1 : -1;

    if (canScrollWithinSheet(sheet, deltaY)) {
      return;
    }

    event.preventDefault();

    if (pageScrollLock) {
      return;
    }

    pageScrollLock = true;
    pageByDelta(deltaY);
  });

  const revealHashTarget = () => {
    if (!window.location.hash) {
      queueViewportCheck();
      return;
    }

    const target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));

    if (!target || !target.classList.contains("page-sheet")) {
      queueViewportCheck();
      return;
    }

    target.classList.add("is-visible");
    target.scrollIntoView({ block: "start", behavior: "auto" });
    queueViewportCheck();
  };

  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    {
      root: null,
      threshold: 0.18,
      rootMargin: "0px 0px -10% 0px",
    }
  );

  const currentObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        entry.target.classList.toggle("is-current", entry.isIntersecting);
      });
    },
    {
      root: null,
      threshold: 0.56,
    }
  );

  sheets.forEach((sheet) => {
    revealObserver.observe(sheet);
    currentObserver.observe(sheet);
  });

  requestAnimationFrame(() => {
    sheets[0].classList.add("is-visible");
    markViewportSheets();
    revealHashTarget();
  });

  window.addEventListener("scroll", queueViewportCheck, { passive: true });
  window.addEventListener("hashchange", revealHashTarget);
  window.addEventListener("resize", queueViewportCheck);
  window.setTimeout(revealHashTarget, 120);
  window.setTimeout(revealHashTarget, 420);
})();
