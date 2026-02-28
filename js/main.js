// =============================================
//  BITS WILP EC3 Prep Hub — Main JS
// =============================================

// ── NAV TOGGLE (mobile) ──────────────────────
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');

if (navToggle) {
  navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
}

// Close nav when a link is clicked (mobile)
document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', () => navLinks?.classList.remove('open'));
});

// ── ACTIVE NAV LINK ───────────────────────────
const currentPage = window.location.pathname.split('/').pop() || 'index.html';
document.querySelectorAll('.nav-links a').forEach(a => {
  if (a.getAttribute('href') === currentPage) a.classList.add('active');
});

// ── ACCORDION ─────────────────────────────────
document.querySelectorAll('.accordion-header').forEach(header => {
  header.addEventListener('click', () => {
    const isOpen = header.classList.contains('open');
    // Close all
    document.querySelectorAll('.accordion-header').forEach(h => {
      h.classList.remove('open');
      h.nextElementSibling?.classList.remove('open');
    });
    // Open clicked (if was closed)
    if (!isOpen) {
      header.classList.add('open');
      header.nextElementSibling?.classList.add('open');
    }
  });
});

// ── TAB SYSTEM ────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(target)?.classList.add('active');
  });
});

// ── CHECKLIST ─────────────────────────────────
document.querySelectorAll('.checklist li').forEach(item => {
  item.addEventListener('click', () => {
    item.classList.toggle('checked');
    if (item.classList.contains('checked')) {
      item.querySelector('.check-box').textContent = '✓';
      showToast('✅ Great job! Keep going!');
    } else {
      item.querySelector('.check-box').textContent = '';
    }
    saveChecklist();
  });
});

function saveChecklist() {
  const states = [];
  document.querySelectorAll('.checklist li').forEach(li => {
    states.push(li.classList.contains('checked'));
  });
  try { localStorage.setItem('ec3_checklist', JSON.stringify(states)); } catch(e) {}
}

function loadChecklist() {
  try {
    const saved = JSON.parse(localStorage.getItem('ec3_checklist') || '[]');
    document.querySelectorAll('.checklist li').forEach((li, i) => {
      if (saved[i]) {
        li.classList.add('checked');
        li.querySelector('.check-box').textContent = '✓';
      }
    });
  } catch(e) {}
}
loadChecklist();

// ── TOAST ─────────────────────────────────────
function showToast(msg, duration = 2500) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = msg;
  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── SMOOTH REVEAL (Intersection Observer) ─────
const revealEls = document.querySelectorAll('.card, .tip-card, .accordion-item, .resource-list li');
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity = '1';
      e.target.style.transform = 'translateY(0)';
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });

revealEls.forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(18px)';
  el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  observer.observe(el);
});
