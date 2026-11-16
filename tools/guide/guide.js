(function () {
  const root = document.getElementById('complete-guide');
  if (!root) return;
  const sidebar = root.querySelector('.sidebar');
  const modeStepper = root.querySelector('#mode-stepper');
  const modeChapter = root.querySelector('#mode-chapter');
  const chapterFiles = Object.fromEntries(Array.from(root.querySelectorAll('.map-chapter')).map(item => [item.dataset.chapter, item.dataset.file]));

  // Highlight the current chapter and topic in the contents list and keep it visible in the sidebar.
  function markContents(article) {
    root.querySelectorAll('.map-chapter').forEach(item => {
      item.classList.toggle('is-current', item.dataset.chapter === article.dataset.chapter);
    });
    let active = null;
    root.querySelectorAll('.course-map li ol a[data-topic]').forEach(link => {
      const match = link.dataset.topic === article.id;
      if (match) { link.setAttribute('aria-current', 'step'); active = link; }
      else link.removeAttribute('aria-current');
    });
    if (active && sidebar && getComputedStyle(sidebar).position === 'sticky') {
      const box = sidebar.getBoundingClientRect();
      const item = active.getBoundingClientRect();
      if (item.top < box.top + 40 || item.bottom > box.bottom - 40) {
        sidebar.scrollTop += item.top - box.top - box.height / 3;
      }
    }
  }

  const chapterSelect = root.querySelector('#guide-chapter');
  if (chapterSelect) setUpStepper(); else if (root.classList.contains('mode-chapter')) setUpChapterPage();

  // ------------------------------------------------------------ step-by-step reader
  function setUpStepper() {
    const topicSelect = root.querySelector('#guide-topic');
    const articles = Array.from(root.querySelectorAll('article[data-chapter]'));
    const previous = root.querySelector('#guide-previous');
    const next = root.querySelector('#guide-next');
    const stepPrevious = root.querySelector('#guide-step-previous');
    const stepNext = root.querySelector('#guide-step-next');
    const previousTitle = root.querySelector('#guide-previous-title');
    const nextTitle = root.querySelector('#guide-next-title');
    const position = root.querySelector('#guide-position');
    const progress = root.querySelector('#guide-progress');
    const navigator = root.querySelector('.navigator');
    const reading = root.querySelector('#guide-reading');
    const shortChapter = Object.fromEntries(Array.from(chapterSelect.options).map(option => [option.value, option.textContent]));
    let current = null;

    // Bring the start of the new topic into view without hiding it under the sticky navigator.
    function scrollToReading(focusId) {
      const target = focusId && document.getElementById(focusId);
      if (target) { try { target.scrollIntoView({ block: 'start' }); } catch (_) {} return; }
      const sticky = getComputedStyle(navigator).position === 'sticky';
      const anchor = sticky ? reading : navigator;
      const offset = sticky ? navigator.offsetHeight + 8 : 8;
      const top = anchor.getBoundingClientRect().top + window.scrollY - offset;
      if (window.scrollY > top) window.scrollTo(0, Math.max(0, top));
    }

    function selectArticle(article, options) {
      const settings = Object.assign({ remember: true, scroll: true, focusId: null }, options);
      if (!article) return;
      current = article;
      chapterSelect.value = article.dataset.chapter;
      const peers = articles.filter(item => item.dataset.chapter === article.dataset.chapter);
      topicSelect.replaceChildren(...peers.map((item, index) => {
        const option = document.createElement('option');
        option.value = item.id;
        option.textContent = peers.length > 1 ? (index + 1) + '. ' + item.dataset.title : item.dataset.title;
        return option;
      }));
      topicSelect.value = article.id;
      topicSelect.disabled = peers.length < 2;
      articles.forEach(item => { item.hidden = item !== article; });
      root.classList.toggle('on-first-topic', article === articles[0]);
      const index = articles.indexOf(article);
      const before = articles[index - 1];
      const after = articles[index + 1];
      previous.disabled = stepPrevious.disabled = !before;
      next.disabled = stepNext.disabled = !after;
      previousTitle.textContent = before ? before.dataset.title : '';
      nextTitle.textContent = after ? after.dataset.title : '';
      position.textContent = (index + 1) + ' of ' + articles.length + ' · ' + shortChapter[article.dataset.chapter];
      progress.style.width = ((index + 1) / articles.length * 100) + '%';
      document.title = article.dataset.title + ' · Subliminal learning, explained slowly';
      markContents(article);
      if (modeChapter) modeChapter.href = chapterFiles[article.dataset.chapter] + '#' + (settings.focusId || article.id);
      if (modeStepper) modeStepper.href = 'index.html#' + article.id;
      if (settings.remember) {
        try { history.replaceState(null, '', '#' + (settings.focusId || article.id)); } catch (_) {}
      }
      if (settings.scroll) scrollToReading(settings.focusId);
    }

    function step(offset) {
      const index = articles.indexOf(current);
      selectArticle(articles[Math.min(articles.length - 1, Math.max(0, index + offset))]);
    }

    chapterSelect.addEventListener('change', () => selectArticle(articles.find(item => item.dataset.chapter === chapterSelect.value)));
    topicSelect.addEventListener('change', () => selectArticle(document.getElementById(topicSelect.value)));
    previous.addEventListener('click', () => step(-1));
    next.addEventListener('click', () => step(1));
    stepPrevious.addEventListener('click', () => step(-1));
    stepNext.addEventListener('click', () => step(1));
    document.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      const focus = document.activeElement;
      if (focus && (focus.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(focus.tagName) || focus.closest('[role="tablist"], #sl-explainer'))) return;
      event.preventDefault();
      step(event.key === 'ArrowLeft' ? -1 : 1);
    });
    root.addEventListener('click', event => {
      const link = event.target.closest('a[href]');
      if (!link || event.metaKey || event.ctrlKey || event.shiftKey || event.button) return;
      const href = link.getAttribute('href');
      const hash = href.startsWith('#') ? href.slice(1) : href.startsWith('index.html#') ? href.slice(11) : null;
      if (!hash) return;
      const target = document.getElementById(decodeURIComponent(hash));
      const article = target && target.closest('article[data-chapter]');
      if (!article) return;
      event.preventDefault();
      selectArticle(article, { focusId: target === article ? null : target.id });
    });
    window.addEventListener('hashchange', () => {
      const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      const article = target && target.closest('article[data-chapter]');
      if (article && article !== current) selectArticle(article, { remember: false, focusId: target === article ? null : target.id });
    });
    const initialTarget = location.hash ? document.getElementById(decodeURIComponent(location.hash.slice(1))) : null;
    const initial = initialTarget && initialTarget.closest('article[data-chapter]');
    selectArticle(initial || articles[0], { remember: false, scroll: Boolean(initial), focusId: initial && initialTarget !== initial ? initialTarget.id : null });
  }

  // ------------------------------------------------------------ one chapter on a page
  function setUpChapterPage() {
    const articles = Array.from(root.querySelectorAll('article[data-chapter]'));
    if (!articles.length) return;
    let current = null;
    let queued = false;
    function update() {
      queued = false;
      const line = Math.min(160, window.innerHeight * 0.3);
      let active = articles[0];
      for (const article of articles) {
        if (article.getBoundingClientRect().top <= line) active = article; else break;
      }
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) active = articles[articles.length - 1];
      if (active === current) return;
      current = active;
      markContents(active);
      if (modeStepper) modeStepper.href = 'index.html#' + active.id;
    }
    window.addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
    window.addEventListener('resize', update);
    update();
  }
})();
