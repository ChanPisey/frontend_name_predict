const API_URL = window.location.origin;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const nameInput   = document.getElementById('nameInput');
const guessBtn    = document.getElementById('guessBtn');
const resultBox   = document.getElementById('resultBox');
const resultTitle = document.getElementById('resultTitle');
const resultDesc  = document.getElementById('resultDesc');
const resultBadge = document.getElementById('resultBadge');
const apiStatus   = document.getElementById('apiStatus');

// ── Khmer-only validation ─────────────────────────────────────────────────────
const KHMER_ONLY_REGEX   = /^[\u1780-\u17FF\s]+$/;
const KHMER_FILTER_REGEX = /[^\u1780-\u17FF\s]/g;

nameInput.addEventListener('input', () => {
  const before = nameInput.value;
  const after  = before.replace(KHMER_FILTER_REGEX, '');
  if (before !== after) nameInput.value = after;
});

// ── Mobile menu ───────────────────────────────────────────────────────────────
const menuBtn = document.getElementById('menuBtn');
const menu    = document.getElementById('menu');
if (menuBtn && menu) {
  menuBtn.addEventListener('click', () => {
    const open = menu.classList.toggle('open');
    menuBtn.setAttribute('aria-expanded', open);
  });
}

// ── API health check ──────────────────────────────────────────────────────────
async function checkAPIHealth() {
  if (!apiStatus) return;
  try {
    const res  = await fetch(`${API_URL}/health`);
    const data = await res.json();
    if (data.status === 'healthy') {
      apiStatus.className   = 'api-badge online';
      apiStatus.textContent = 'ដំណើរការ';
    } else {
      throw new Error('not healthy');
    }
  } catch {
    apiStatus.className   = 'api-badge offline';
    apiStatus.textContent = '✗ មិនដំណើរការ';
  }
}
checkAPIHealth();

// ── API predict ───────────────────────────────────────────────────────────────
async function predictFromAPI(name) {
  const res = await fetch(`${API_URL}/predict`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Fallback demo gender guess ────────────────────────────────────────────────
function guessGender(name) {
  const femaleEndings = ['ា','ី','ិ','ំ','ន','ល'];
  const last = [...name].pop() || '';
  if (femaleEndings.includes(last)) return { label: 'female', confidence: 0.62 };
  return { label: 'male', confidence: 0.55 };
}

// ── Render result ─────────────────────────────────────────────────────────────
function setResultTitle(genderLabel, color, token) {
  resultTitle.textContent = '';
  const span = document.createElement('span');
  span.style.cssText = `color:${color};font-weight:900`;
  span.textContent   = genderLabel;
  resultTitle.appendChild(span);
  resultTitle.append(` — "${token}"`);
}

async function render() {
  const clean = nameInput.value.trim();

  if (!clean) {
    resultBox.style.display = 'none';
    return;
  }

  if (!KHMER_ONLY_REGEX.test(clean)) {
    resultBox.style.display  = 'block';
    resultTitle.textContent  = 'បញ្ចូលមិនត្រឹមត្រូវ';
    resultDesc.textContent   = 'សូមបញ្ចូលអក្សរខ្មែរ ប៉ុណ្ណោះ (មិនអនុញ្ញាតអក្សរឡាតាំង លេខ ឬសញ្ញា)';
    resultBadge.textContent  = '';
    resultBadge.style.borderColor = '';
    resultBadge.style.color       = '';
    return;
  }

  guessBtn.disabled    = true;
  guessBtn.textContent = '…';
  resultBox.style.display = 'none';

  try {
    const data   = await predictFromAPI(clean);
    const isMale = data.gender === 'Male';
    const pct    = data.confidence;
    const color  = isMale ? '#2563eb' : '#db2777';

    setResultTitle(isMale ? '♂ ប្រុស' : '♀ ស្រី', color, clean);
    resultDesc.textContent   = 'ទំនាយភេទតាម AI Model';
    resultBadge.textContent  = `ភាពទុកចិត្ត: ${typeof pct === 'number' ? pct.toFixed(1) : pct}%`;
    resultBadge.style.borderColor = isMale ? '#bfdbfe' : '#fbcfe8';
    resultBadge.style.color       = isMale ? '#1d4ed8' : '#be185d';

  } catch {
    const g = guessGender(clean);
    resultTitle.textContent  = `លទ្ធផលភេទ (demo) "${clean}"`;
    resultDesc.textContent   =
      g.label === 'unknown'
        ? 'មិនអាចកំណត់បាន'
        : `ទាយថា: ${g.label === 'male' ? 'ប្រុស' : 'ស្រី'} — demo, API មិនអាចប្រើបាន`;
    resultBadge.textContent       = `ភាពទុកចិត្ត: ${(g.confidence * 100).toFixed(0)}%`;
    resultBadge.style.borderColor = '';
    resultBadge.style.color       = '';

  } finally {
    guessBtn.disabled    = false;
    guessBtn.textContent = 'ទស្សន៏ទាយ';
    resultBox.style.display = 'block';
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────
guessBtn.addEventListener('click', render);
nameInput.addEventListener('keydown', e => { if (e.key === 'Enter') render(); });
