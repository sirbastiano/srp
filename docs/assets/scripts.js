(() => {
  const sidebarButton = document.querySelector('[data-sidebar-toggle]');
  if (sidebarButton) {
    sidebarButton.addEventListener('click', () => {
      const opened = document.body.classList.toggle('sidebar-open');
      sidebarButton.setAttribute('aria-expanded', String(opened));
    });
  }

  // Add anchor links to headings.
  const headingSelector = 'article h1, article h2, article h3, article h4';
  for (const heading of document.querySelectorAll(headingSelector)) {
    if (!heading.id) {
      const slug = heading.textContent
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
      if (slug) heading.id = slug;
    }
    if (!heading.id) continue;
    if (heading.querySelector('.anchor-link')) continue;
    const anchor = document.createElement('a');
    anchor.className = 'anchor-link';
    anchor.href = `#${heading.id}`;
    anchor.setAttribute('aria-label', `Link to ${heading.textContent.trim()}`);
    anchor.textContent = '#';
    heading.appendChild(anchor);
  }

  // Add copy buttons to code blocks.
  for (const pre of document.querySelectorAll('pre')) {
    const code = pre.querySelector('code');
    if (!code) continue;
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent || '');
        button.textContent = 'Copied';
        setTimeout(() => (button.textContent = 'Copy'), 1200);
      } catch (_err) {
        button.textContent = 'Failed';
        setTimeout(() => (button.textContent = 'Copy'), 1200);
      }
    });
    pre.appendChild(button);
  }

  // Simple in-page filter for search-ready sections.
  const searchInput = document.getElementById('site-search');
  if (searchInput) {
    const candidates = Array.from(document.querySelectorAll('[data-searchable]'));
    const runFilter = () => {
      const q = searchInput.value.trim().toLowerCase();
      for (const node of candidates) {
        if (!q) {
          node.classList.remove('hidden-by-filter');
          continue;
        }
        const haystack = [node.getAttribute('data-search') || '', node.textContent || '']
          .join(' ')
          .toLowerCase();
        if (haystack.includes(q)) {
          node.classList.remove('hidden-by-filter');
        } else {
          node.classList.add('hidden-by-filter');
        }
      }
    };
    searchInput.addEventListener('input', runFilter);
  }
})();
