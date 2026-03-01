const API_URL = 'http://localhost:8000';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const nameInput   = document.getElementById('nameInput');
const guessBtn    = document.getElementById('guessBtn');
const resultBox   = document.getElementById('resultBox');
const resultTitle = document.getElementById('resultTitle');
const resultDesc  = document.getElementById('resultDesc');
const resultBadge = document.getElementById('resultBadge');
const apiStatus   = document.getElementById('apiStatus');

// ── Khmer-only validation ─────────────────────────────────────────────────────
// Khmer block: U+1780–U+17FF (letters + signs). Allow spaces.
// If you want to allow Khmer numerals too, they are also in this block.
const KHMER_ONLY_REGEX = /^[\u1780-\u17FF\s]+$/;
const KHMER_FILTER_REGEX = /[^\u1780-\u17FF\s]/g;

// Optional: auto-remove non-Khmer characters while typing (best UX)
nameInput.addEventListener('input', () => {
  const before = nameInput.value;
  const after = before.replace(KHMER_FILTER_REGEX, '');
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

// ── Mode chips ────────────────────────────────────────────────────────────────
let currentMode = 'gender';
document.querySelectorAll('.chip[data-mode]').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.chip[data-mode]').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    currentMode = chip.dataset.mode;
    resultBox.style.display = 'none';
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function sanitizeName(s) {
  return (s || '').trim();
}

// ── API health check ──────────────────────────────────────────────────────────
async function checkAPIHealth() {
  if (!apiStatus) return;
  try {
    const res  = await fetch(`${API_URL}/health`);
    const data = await res.json();
    if (data.status === 'healthy') {
      apiStatus.className     = 'api-badge online';
      apiStatus.textContent   = 'ដំណើរការ';
    } else {
      throw new Error('not healthy');
    }
  } catch {
    apiStatus.className   = 'api-badge offline';
    apiStatus.textContent = '✗ មិនដំណើរការ';
  }
}
checkAPIHealth();

// ── Real API predict ──────────────────────────────────────────────────────────
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

// ── Fallback helpers ──────────────────────────────────────────────────────────
function fakeOrigin(name) {
  return { hint: 'ប្រហែលជាឈ្មោះខ្មែរ (demo)', confidence: 0.5 };
}
function splitName(name) {
  return { parts: name.split(/\s+/).filter(Boolean), confidence: 0.9 };
}

// ── Main render ───────────────────────────────────────────────────────────────
async function render() {
  const raw   = nameInput.value;
  const clean = sanitizeName(raw);

  if (!clean) {
    resultBox.style.display = 'none';
    return;
  }

  // ✅ Validate Khmer-only (block API calls if invalid)
  if (!KHMER_ONLY_REGEX.test(clean)) {
    resultBox.style.display = 'block';
    resultTitle.textContent = 'បញ្ចូលមិនត្រឹមត្រូវ';
    resultDesc.textContent  = 'សូមបញ្ចូលអក្សរខ្មែរ ប៉ុណ្ណោះ (មិនអនុញ្ញាតអក្សរឡាតាំង លេខ ឬសញ្ញា)';
    resultBadge.textContent = '';
    resultBadge.style.borderColor = '';
    resultBadge.style.color = '';
    return;
  }

  const token = clean;

  // Non-gender modes use local demo logic immediately
  if (currentMode !== 'gender') {
    renderLocalMode(clean, token);
    resultBox.style.display = 'block';
    return;
  }

  // ── Gender mode: try API first ────────────────────────────────────────────
  guessBtn.disabled    = true;
  guessBtn.textContent = '…';
  resultBox.style.display = 'none';

  try {
    const data = await predictFromAPI(clean);

    const isMale   = data.gender === 'Male';
    const pct      = data.confidence; // already 0-100
    const kccsText = Array.isArray(data.kccs) ? data.kccs.join(' + ') : '';

    resultTitle.innerHTML =
      `<span style="color:${isMale ? '#2563eb' : '#db2777'};font-weight:900">` +
      `${isMale ? '♂ ប្រុស' : '♀ ស្រី'}` +
      `</span> — "${token}"`;

    resultDesc.innerHTML =
      kccsText
        ? `<span style="font-family:'Khmer OS',serif;font-size:15px">${kccsText}</span>`
        : 'ទំនាយភេទតាម AI Model (KCC + FastText + BiLSTM)';

    resultBadge.textContent = `ភាពទុកចិត្ត: ${typeof pct === 'number' ? pct.toFixed(1) : pct}%`;
    resultBadge.style.borderColor = isMale ? '#bfdbfe' : '#fbcfe8';
    resultBadge.style.color       = isMale ? '#1d4ed8' : '#be185d';

  } catch {
    const g = guessGender(clean);
    resultTitle.textContent = `លទ្ធផលភេទ (demo) "${token}"`;
    resultDesc.textContent  =
      g.label === 'unknown'
        ? 'មិនអាចកំណត់បាន'
        : `ទាយថា: ${g.label === 'male' ? 'ប្រុស' : 'ស្រី'}  — demo, API មិនអាចប្រើបាន`;
    resultBadge.textContent = `ភាពទុកចិត្ត: ${(g.confidence * 100).toFixed(0)}%`;
    resultBadge.style.borderColor = '';
    resultBadge.style.color       = '';
  } finally {
    guessBtn.disabled    = false;
    guessBtn.textContent = 'ស្វែងរក';
    resultBox.style.display = 'block';
  }
}

function renderLocalMode(clean, token) {
  if (currentMode === 'origin') {
    const o = fakeOrigin(clean);
    resultTitle.textContent = `ប្រភពឈ្មោះ "${token}"`;
    resultDesc.textContent  = o.hint;
    resultBadge.textContent = `ភាពទុកចិត្ត: ${(o.confidence * 100).toFixed(0)}%`;
  
  } else if (currentMode === 'split') {
    const s = splitName(clean);
    resultTitle.textContent = 'បំបែកឈ្មោះ';
    resultDesc.textContent  = s.parts.length
      ? `ផ្នែក: ${s.parts.map(p => `"${p}"`).join(', ')}`
      : 'រកមិនឃើញផ្នែកណាទេ។';
    resultBadge.textContent = `ភាពទុកចិត្ត: ${(s.confidence * 100).toFixed(0)}%`;
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────
guessBtn.addEventListener('click', render);
nameInput.addEventListener('keydown', e => { if (e.key === 'Enter') render(); });