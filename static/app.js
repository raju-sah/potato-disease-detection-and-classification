// ── State Management ────────────────────────────────────────────────────────
const AppState = {
  currentFile: null,
  currentPreviewUrl: null,
  isAnalyzing: false,
  lastResult: null,
  scanHistory: [],
  webcamStream: null
};

// ── DOM Element Cache ────────────────────────────────────────────────────────
const DOM = {
  // Theme & Status
  themeToggleBtn: document.getElementById('themeToggleBtn'),
  systemStatusText: document.getElementById('systemStatusText'),
  toastContainer: document.getElementById('toastContainer'),

  // Inputs & Dropzone
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  browseFileBtn: document.getElementById('browseFileBtn'),
  previewContainer: document.getElementById('previewContainer'),
  previewImg: document.getElementById('previewImg'),
  removeImgBtn: document.getElementById('removeImgBtn'),
  openCameraBtn: document.getElementById('openCameraBtn'),
  closeCameraBtn: document.getElementById('closeCameraBtn'),
  cameraContainer: document.getElementById('cameraContainer'),
  webcamVideo: document.getElementById('webcamVideo'),
  snapPhotoBtn: document.getElementById('snapPhotoBtn'),
  pasteClipboardBtn: document.getElementById('pasteClipboardBtn'),
  samplesGrid: document.getElementById('samplesGrid'),
  refreshSamplesBtn: document.getElementById('refreshSamplesBtn'),

  // Settings
  ttaToggle: document.getElementById('ttaToggle'),
  ttaPassesRow: document.getElementById('ttaPassesRow'),
  ttaSlider: document.getElementById('ttaSlider'),
  ttaPassesVal: document.getElementById('ttaPassesVal'),
  confSlider: document.getElementById('confSlider'),
  confThreshVal: document.getElementById('confThreshVal'),
  runDiagnosisBtn: document.getElementById('runDiagnosisBtn'),
  btnIcon: document.getElementById('btnIcon'),
  btnText: document.getElementById('btnText'),

  // Results
  emptyStateCard: document.getElementById('emptyStateCard'),
  resultsCard: document.getElementById('resultsCard'),
  reportTimestamp: document.getElementById('reportTimestamp'),
  diagnosisBanner: document.getElementById('diagnosisBanner'),
  resEmoji: document.getElementById('resEmoji'),
  resTitle: document.getElementById('resTitle'),
  resPathogen: document.getElementById('resPathogen'),
  resSeverityBadge: document.getElementById('resSeverityBadge'),
  emergencyAlertBox: document.getElementById('emergencyAlertBox'),
  
  // Metrics
  metricConfidence: document.getElementById('metricConfidence'),
  metricSeverity: document.getElementById('metricSeverity'),
  metricLatency: document.getElementById('metricLatency'),
  metricMode: document.getElementById('metricMode'),

  // Probabilities
  probValHealthy: document.getElementById('probValHealthy'),
  probFillHealthy: document.getElementById('probFillHealthy'),
  probValEarlyBlight: document.getElementById('probValEarlyBlight'),
  probFillEarlyBlight: document.getElementById('probFillEarlyBlight'),
  probValLateBlight: document.getElementById('probValLateBlight'),
  probFillLateBlight: document.getElementById('probFillLateBlight'),

  // Tabs & Diagnostics
  symptomsList: document.getElementById('symptomsList'),
  treatmentList: document.getElementById('treatmentList'),
  preventionList: document.getElementById('preventionList'),
  causesList: document.getElementById('causesList'),

  // Action Tools
  printReportBtn: document.getElementById('printReportBtn'),
  copySummaryBtn: document.getElementById('copySummaryBtn'),
  resetScanBtn: document.getElementById('resetScanBtn'),

  // History
  historySection: document.getElementById('historySection'),
  historyGrid: document.getElementById('historyGrid')
};

// ── Toast Utility ────────────────────────────────────────────────────────────
function showToast(message, icon = '🍃', duration = 3500) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
  DOM.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Theme Switcher ───────────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('potatodx_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  DOM.themeToggleBtn.innerHTML = saved === 'dark' ? '🌙' : '☀️';
}

DOM.themeToggleBtn.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('potatodx_theme', next);
  DOM.themeToggleBtn.innerHTML = next === 'dark' ? '🌙' : '☀️';
  showToast(`Theme switched to ${next} mode`, next === 'dark' ? '🌙' : '☀️');
});

// ── Settings Handlers ────────────────────────────────────────────────────────
DOM.ttaToggle.addEventListener('change', (e) => {
  DOM.ttaPassesRow.style.display = e.target.checked ? 'flex' : 'none';
});

DOM.ttaSlider.addEventListener('input', (e) => {
  DOM.ttaPassesVal.textContent = `${e.target.value} Passes`;
});

DOM.confSlider.addEventListener('input', (e) => {
  DOM.confThreshVal.textContent = `${e.target.value}%`;
});

// ── File Selection & Drag-and-Drop ───────────────────────────────────────────
DOM.browseFileBtn.addEventListener('click', () => DOM.fileInput.click());
DOM.dropzone.addEventListener('click', (e) => {
  if (e.target !== DOM.browseFileBtn && !DOM.browseFileBtn.contains(e.target)) {
    DOM.fileInput.click();
  }
});

DOM.fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFile(e.target.files[0]);
  }
});

DOM.dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  DOM.dropzone.classList.add('drag-active');
});

DOM.dropzone.addEventListener('dragleave', () => {
  DOM.dropzone.classList.remove('drag-active');
});

DOM.dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  DOM.dropzone.classList.remove('drag-active');
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFile(e.dataTransfer.files[0]);
  }
});

// Clipboard Paste
DOM.pasteClipboardBtn.addEventListener('click', async () => {
  try {
    const items = await navigator.clipboard.read();
    for (const item of items) {
      for (const type of item.types) {
        if (type.startsWith('image/')) {
          const blob = await item.getType(type);
          const file = new File([blob], 'clipboard_leaf.png', { type });
          handleFile(file);
          showToast('Image pasted from clipboard', '📋');
          return;
        }
      }
    }
    showToast('No image found in clipboard', '⚠️');
  } catch (err) {
    showToast('Clipboard access denied or unsupported', '⚠️');
  }
});

// Global Paste
window.addEventListener('paste', (e) => {
  if (e.clipboardData && e.clipboardData.files && e.clipboardData.files[0]) {
    const file = e.clipboardData.files[0];
    if (file.type.startsWith('image/')) {
      handleFile(file);
      showToast('Image pasted from clipboard', '📋');
    }
  }
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please select a valid image file (JPG, PNG, WEBP)', '❌');
    return;
  }

  stopWebcam();
  AppState.currentFile = file;
  
  if (AppState.currentPreviewUrl) {
    URL.revokeObjectURL(AppState.currentPreviewUrl);
  }
  AppState.currentPreviewUrl = URL.createObjectURL(file);
  
  DOM.previewImg.src = AppState.currentPreviewUrl;
  DOM.previewContainer.classList.add('active');
  DOM.dropzone.style.display = 'none';
  DOM.runDiagnosisBtn.disabled = false;
  
  showToast(`Loaded ${file.name}`, '🖼️');
}

DOM.removeImgBtn.addEventListener('click', () => {
  resetFileInput();
});

function resetFileInput() {
  AppState.currentFile = null;
  if (AppState.currentPreviewUrl) {
    URL.revokeObjectURL(AppState.currentPreviewUrl);
    AppState.currentPreviewUrl = null;
  }
  DOM.fileInput.value = '';
  DOM.previewImg.src = '';
  DOM.previewContainer.classList.remove('active');
  DOM.dropzone.style.display = 'block';
  DOM.runDiagnosisBtn.disabled = true;
}

// ── Live Camera Stream ───────────────────────────────────────────────────────
DOM.openCameraBtn.addEventListener('click', async () => {
  resetFileInput();
  try {
    AppState.webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    DOM.webcamVideo.srcObject = AppState.webcamStream;
    DOM.cameraContainer.classList.add('active');
    DOM.dropzone.style.display = 'none';
    showToast('Camera active — center leaf in view', '📸');
  } catch (err) {
    console.error(err);
    showToast('Unable to access webcam: ' + err.message, '❌');
  }
});

DOM.closeCameraBtn.addEventListener('click', stopWebcam);

function stopWebcam() {
  if (AppState.webcamStream) {
    AppState.webcamStream.getTracks().forEach(t => t.stop());
    AppState.webcamStream = null;
  }
  DOM.cameraContainer.classList.remove('active');
  if (!AppState.currentFile) {
    DOM.dropzone.style.display = 'block';
  }
}

DOM.snapPhotoBtn.addEventListener('click', () => {
  if (!AppState.webcamStream) return;
  const canvas = document.createElement('canvas');
  canvas.width = DOM.webcamVideo.videoWidth || 640;
  canvas.height = DOM.webcamVideo.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(DOM.webcamVideo, 0, 0, canvas.width, canvas.height);
  
  canvas.toBlob((blob) => {
    const file = new File([blob], `camera_leaf_${Date.now()}.jpg`, { type: 'image/jpeg' });
    stopWebcam();
    handleFile(file);
    showToast('Photo captured!', '📸');
  }, 'image/jpeg', 0.95);
});

// ── Preset Test Samples (Real PLD Dataset) ──────────────────────────────────
async function loadPresetSamples() {
  try {
    const res = await fetch('/api/samples');
    if (!res.ok) return;
    const data = await res.json();
    DOM.samplesGrid.innerHTML = '';
    
    if (data.samples && data.samples.length > 0) {
      data.samples.forEach(sample => {
        const chip = document.createElement('div');
        chip.className = 'sample-chip';
        
        let badgeClass = 'badge-hl';
        if (sample.expected_class === 'Early_Blight') badgeClass = 'badge-eb';
        else if (sample.expected_class === 'Late_Blight') badgeClass = 'badge-lb';

        chip.innerHTML = `
          <img src="${sample.url}" alt="${sample.name}" class="sample-chip-thumb" loading="lazy">
          <span class="sample-chip-badge ${badgeClass}">${sample.badge || sample.expected_class.replace('_', ' ')}</span>
          <span class="sample-chip-name" title="${sample.name}">${sample.name}</span>
        `;

        chip.addEventListener('click', async () => {
          try {
            showToast(`Loading real sample: ${sample.name}...`, '🧪');
            const imgRes = await fetch(sample.url);
            const blob = await imgRes.blob();
            const file = new File([blob], sample.id, { type: blob.type || 'image/jpeg' });
            handleFile(file);
            // Auto trigger analysis on sample click
            runDiagnosis();
          } catch (err) {
            showToast('Failed to load sample image', '❌');
          }
        });

        DOM.samplesGrid.appendChild(chip);
      });
    } else {
      DOM.samplesGrid.innerHTML = '<div style="grid-column: 1/-1; font-size: 0.8rem; color: var(--text-muted); text-align: center;">No preset samples found</div>';
    }
  } catch (err) {
    console.error('Samples loading error:', err);
  }
}

DOM.refreshSamplesBtn.addEventListener('click', loadPresetSamples);

// ── AI Inference Request ─────────────────────────────────────────────────────
DOM.runDiagnosisBtn.addEventListener('click', runDiagnosis);

async function runDiagnosis() {
  if (!AppState.currentFile || AppState.isAnalyzing) return;

  AppState.isAnalyzing = true;
  DOM.runDiagnosisBtn.disabled = true;
  DOM.btnIcon.className = 'spinner';
  DOM.btnIcon.textContent = '';
  DOM.btnText.textContent = 'Analysing Foliar Pathology...';

  const formData = new FormData();
  formData.append('file', AppState.currentFile);
  formData.append('use_tta', DOM.ttaToggle.checked);
  formData.append('tta_passes', DOM.ttaSlider.value);
  formData.append('confidence_threshold', DOM.confSlider.value);

  try {
    const startTime = performance.now();
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errJson = await res.json();
      throw new Error(errJson.detail || 'Prediction failed');
    }

    const data = await res.json();
    AppState.lastResult = data;
    renderResults(data);
    addToHistory(data, AppState.currentPreviewUrl);
    showToast(`Diagnosis: ${data.prediction.display_name} (${data.prediction.confidence}%)`, data.prediction.emoji);
  } catch (err) {
    console.error('Diagnosis Error:', err);
    showToast(`Error: ${err.message}`, '❌', 5000);
  } finally {
    AppState.isAnalyzing = false;
    DOM.runDiagnosisBtn.disabled = false;
    DOM.btnIcon.className = '';
    DOM.btnIcon.textContent = '🧠';
    DOM.btnText.textContent = 'Analyze Leaf Pathology';
  }
}

// ── Render Results ───────────────────────────────────────────────────────────
function renderResults(data) {
  const p = data.prediction;
  const diag = data.diagnostics;
  const meta = data.meta;

  DOM.emptyStateCard.style.display = 'none';
  DOM.resultsCard.classList.add('active');

  // Timestamp
  DOM.reportTimestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  // Banner
  DOM.resEmoji.textContent = p.emoji;
  DOM.resTitle.textContent = p.display_name;
  DOM.resPathogen.textContent = p.pathogen;
  DOM.resSeverityBadge.textContent = `SEVERITY: ${p.severity}`;
  DOM.resSeverityBadge.className = `severity-pill ${p.badge_class}`;

  DOM.diagnosisBanner.className = 'diagnosis-banner';
  if (p.severity_level === 0) DOM.diagnosisBanner.classList.add('severity-none');
  else if (p.severity_level === 2) DOM.diagnosisBanner.classList.add('severity-moderate');
  else if (p.severity_level === 3) DOM.diagnosisBanner.classList.add('severity-severe');

  // Emergency Alert
  if (p.urgent_alert) {
    DOM.emergencyAlertBox.classList.add('active');
  } else {
    DOM.emergencyAlertBox.classList.remove('active');
  }

  // Key Metrics
  DOM.metricConfidence.textContent = `${p.confidence}%`;
  DOM.metricConfidence.style.color = p.color;

  DOM.metricSeverity.textContent = p.severity;
  DOM.metricSeverity.style.color = p.color;

  DOM.metricLatency.textContent = `${meta.inference_time_ms} ms`;
  DOM.metricMode.textContent = meta.tta_applied ? `TTA ×${meta.tta_passes}` : 'Single Pass';

  // Probabilities Bar Chart
  const probs = data.probabilities;
  if (probs['Healthy']) {
    DOM.probValHealthy.textContent = `${probs['Healthy'].probability}%`;
    DOM.probFillHealthy.style.width = `${probs['Healthy'].probability}%`;
  }
  if (probs['Early_Blight']) {
    DOM.probValEarlyBlight.textContent = `${probs['Early_Blight'].probability}%`;
    DOM.probFillEarlyBlight.style.width = `${probs['Early_Blight'].probability}%`;
  }
  if (probs['Late_Blight']) {
    DOM.probValLateBlight.textContent = `${probs['Late_Blight'].probability}%`;
    DOM.probFillLateBlight.style.width = `${probs['Late_Blight'].probability}%`;
  }

  // Tabbed Lists
  renderList(DOM.symptomsList, diag.symptoms);
  renderList(DOM.treatmentList, diag.treatment);
  renderList(DOM.preventionList, diag.prevention);
  renderList(DOM.causesList, diag.causes);

  // Scroll smoothly to results on mobile devices
  if (window.innerWidth <= 1024) {
    DOM.resultsCard.scrollIntoView({ behavior: 'smooth' });
  }
}

function renderList(targetElem, items) {
  targetElem.innerHTML = '';
  if (!items || items.length === 0) {
    targetElem.innerHTML = '<li>No specific items noted.</li>';
    return;
  }
  items.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    targetElem.appendChild(li);
  });
}

// ── Tab Switching ────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    btn.classList.add('active');
    const targetId = btn.getAttribute('data-tab');
    const pane = document.getElementById(targetId);
    if (pane) pane.classList.add('active');
  });
});

// ── Action Toolbar ───────────────────────────────────────────────────────────
DOM.printReportBtn.addEventListener('click', () => {
  window.print();
});

DOM.copySummaryBtn.addEventListener('click', async () => {
  if (!AppState.lastResult) return;
  const p = AppState.lastResult.prediction;
  const d = AppState.lastResult.diagnostics;
  
  const text = `🥔 POTATO LEAF PATHOLOGY DIAGNOSIS REPORT
--------------------------------------------------
Diagnosis: ${p.display_name} (${p.pathogen})
Confidence: ${p.confidence}%
Severity: ${p.severity}
Inference Latency: ${AppState.lastResult.meta.inference_time_ms} ms

KEY SYMPTOMS:
${d.symptoms.map(s => `• ${s}`).join('\n')}

TREATMENT PROTOCOL:
${d.treatment.map(t => `• ${t}`).join('\n')}

PREVENTION STRATEGIES:
${d.prevention.map(pr => `• ${pr}`).join('\n')}
--------------------------------------------------
Generated by Potato Leaf Disease AI`;

  try {
    await navigator.clipboard.writeText(text);
    showToast('Clinical summary copied to clipboard!', '📋');
  } catch (err) {
    showToast('Failed to copy to clipboard', '⚠️');
  }
});

DOM.resetScanBtn.addEventListener('click', () => {
  resetFileInput();
  DOM.resultsCard.classList.remove('active');
  DOM.emptyStateCard.style.display = 'block';
  showToast('Ready for next scan', '🔄');
});

// ── History Tracking ─────────────────────────────────────────────────────────
function addToHistory(data, thumbUrl) {
  const p = data.prediction;
  const item = {
    id: Date.now(),
    name: p.display_name,
    emoji: p.emoji,
    confidence: p.confidence,
    severity: p.severity,
    color: p.color,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    thumbUrl: thumbUrl
  };

  AppState.scanHistory.unshift(item);
  if (AppState.scanHistory.length > 6) {
    AppState.scanHistory.pop();
  }
  renderHistory();
}

function renderHistory() {
  if (AppState.scanHistory.length === 0) {
    DOM.historySection.style.display = 'none';
    return;
  }

  DOM.historySection.style.display = 'block';
  DOM.historyGrid.innerHTML = '';

  AppState.scanHistory.forEach(item => {
    const card = document.createElement('div');
    card.className = 'history-card';
    card.innerHTML = `
      <img src="${item.thumbUrl}" class="history-thumb" alt="Scan thumbnail">
      <div style="flex: 1; overflow: hidden;">
        <div class="history-meta-title" style="color: ${item.color}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
          ${item.emoji} ${item.name}
        </div>
        <div class="history-meta-sub">${item.confidence}% • ${item.timestamp}</div>
      </div>
    `;
    DOM.historyGrid.appendChild(card);
  });
}

// ── Initialize App ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  loadPresetSamples();
  showToast('Potato Leaf AI System Ready', '🥔');
});
