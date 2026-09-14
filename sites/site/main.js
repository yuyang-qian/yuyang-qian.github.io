/* The standalone page can respond to a pointer; GitHub README images cannot. */
(() => {
  const root = document.documentElement;
  const systemTheme = matchMedia('(prefers-color-scheme: dark)');
  const themeButton = document.querySelector('#theme-toggle');
  const ANIMATION_SPEED = Number(document.querySelector('.profile-card').dataset.animationSpeed) || 9;
  const HOVER_ANIMATION_SPEED = Number(document.querySelector('.profile-card').dataset.hoverAnimationSpeed) || 3;
  const themes = ['system', 'light', 'dark'];
  let theme = 'system';
  try { const saved = localStorage.getItem('zinyy-theme'); if (themes.includes(saved)) theme = saved; } catch { /* Private or local-file browsing. */ }
  let dark = false;
  const canvas = document.querySelector('#wave-grid');
  const card = document.querySelector('.profile-card');
  const ctx = canvas.getContext('2d');
  let width = 0, height = 0, frame = 0, lastTime = 0, time = 0;
  const pointer = { x: -1000, y: -1000, targetX: -1000, targetY: -1000, strength: 0, active: false };

  function applyTheme() {
    if (theme === 'system') delete root.dataset.theme;
    else root.dataset.theme = theme;
    dark = theme === 'dark' || (theme === 'system' && systemTheme.matches);
    document.querySelectorAll('picture[data-theme-logo] source').forEach(source => {
      source.media = dark ? 'all' : 'not all';
    });
    themeButton.textContent = { system: '◐', light: '☼', dark: '☾' }[theme];
    const next = themes[(themes.indexOf(theme) + 1) % themes.length];
    themeButton.setAttribute('aria-label', `Theme: ${theme}. Switch to ${next} theme.`);
    themeButton.title = `Theme: ${theme}`;
    draw();
  }

  function draw() {
    if (!ctx || !width) return;
    ctx.clearRect(0, 0, width, height);
    const cell = width < 720 ? 34 : 42;
    for (let y = 0; y < height; y += cell) {
      for (let x = 0; x < width; x += cell) {
        const wave = Math.sin(x * .006 + y * .005 - time * .23 + Math.sin(y * .006 + time * .11) * 1.6);
        const crest = ((wave + 1) / 2) ** 5;
        const right = .12 + .88 * (x / width) ** 2;
        const top = .28 + .72 * Math.exp(-y / (height * .85));
        const distance = Math.hypot(x + cell / 2 - pointer.x, y + cell / 2 - pointer.y);
        const glow = Math.exp(-(distance ** 2) / 27000) * pointer.strength;
        const ripple = (.5 + .5 * Math.cos(distance * .037 - time * 1.6)) * glow;
        const alpha = (dark ? .018 : .035) + crest * right * top * (dark ? .28 : .30) + ripple * (dark ? .2 : .20);
        ctx.fillStyle = dark ? `rgba(46,119,221,${alpha})` : `rgba(23,85,174,${alpha})`;
        ctx.fillRect(x + 1, y + 1, cell - 3, cell - 3);
      }
    }
  }

  function resize() {
    width = card.clientWidth;
    height = card.clientHeight;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    if (ctx) ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }

  function tick(now) {
    frame = 0;
    if (document.hidden || !ctx) return;
    const delta = lastTime ? Math.min((now - lastTime) / 1000, .06) : 0;
    // 30 fps is sufficient for a slow ambient wave and limits background work.
    if (!lastTime || now - lastTime >= 1000 / 30) {
      const speed = pointer.active ? HOVER_ANIMATION_SPEED : ANIMATION_SPEED;
      time += delta * speed;
      lastTime = now;
      const easing = 1 - Math.exp(-delta * 6);
      pointer.x += (pointer.targetX - pointer.x) * easing;
      pointer.y += (pointer.targetY - pointer.y) * easing;
      pointer.strength += ((pointer.active ? 1 : 0) - pointer.strength) * easing;
      draw();
    }
    frame = requestAnimationFrame(tick);
  }

  function syncMotion() {
    cancelAnimationFrame(frame);
    frame = 0;
    lastTime = 0;
    if (!document.hidden && ctx) frame = requestAnimationFrame(tick);
    else draw();
  }

  themeButton.addEventListener('click', () => {
    theme = themes[(themes.indexOf(theme) + 1) % themes.length];
    try { localStorage.setItem('zinyy-theme', theme); } catch { /* Keep working without storage. */ }
    applyTheme();
  });
  systemTheme.addEventListener('change', applyTheme);
  document.addEventListener('visibilitychange', syncMotion);
  window.addEventListener('resize', resize, { passive: true });
  new ResizeObserver(resize).observe(card);
  function trackPointer(event) {
    if (event.pointerType === 'touch') return;
    const bounds = card.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    if (!pointer.active) { pointer.x = x; pointer.y = y; }
    pointer.active = true;
    pointer.targetX = x;
    pointer.targetY = y;
  }
  card.addEventListener('pointerenter', trackPointer, { passive: true });
  card.addEventListener('pointermove', trackPointer, { passive: true });
  card.addEventListener('pointerleave', () => { pointer.active = false; });
  window.addEventListener('blur', () => { pointer.active = false; });
  applyTheme();
  resize();
  syncMotion();
})();
