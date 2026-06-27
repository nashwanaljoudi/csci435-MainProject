const API = "";

// ── Element refs ──────────────────────────────────────────────────────────
const els = {
  statusBadge:      document.getElementById("statusBadge"),
  statusText:       document.getElementById("statusText"),
  startBtn:         document.getElementById("startBtn"),
  stopBtn:          document.getElementById("stopBtn"),
  webcam:           document.getElementById("webcam"),
  captureCanvas:    document.getElementById("captureCanvas"),
  annotatedFeed:    document.getElementById("annotatedFeed"),
  cameraPlaceholder:document.getElementById("cameraPlaceholder"),
  cameraHud:        document.getElementById("cameraHud"),
  commitFlash:      document.getElementById("commitFlash"),
  letterDisplay:    document.getElementById("letterDisplay"),
  letterGlowRing:   document.getElementById("letterGlowRing"),
  confidenceFill:   document.getElementById("confidenceFill"),
  confidenceValue:  document.getElementById("confidenceValue"),
  wordTiles:        document.getElementById("wordTiles"),
  tilesEmpty:       document.getElementById("tilesEmpty"),
  sentenceText:     document.getElementById("sentenceText"),
  fullText:         document.getElementById("fullText"),
  finalizeBtn:      document.getElementById("finalizeBtn"),
  backspaceBtn:     document.getElementById("backspaceBtn"),
  clearBtn:         document.getElementById("clearBtn"),
  speakLetterBtn:   document.getElementById("speakLetterBtn"),
  speakWordBtn:     document.getElementById("speakWordBtn"),
  speakSentenceBtn: document.getElementById("speakSentenceBtn"),
  speakBtn:         document.getElementById("speakBtn"),
  fpsLabel:         document.getElementById("fpsLabel"),
  modelLabel:       document.getElementById("modelLabel"),
};

// ── State ─────────────────────────────────────────────────────────────────
let stream          = null;
let processing      = false;
let loopHandle      = null;
let lastFrameTime   = performance.now();
let frameCount      = 0;
let renderedWord    = "";        // uppercase snapshot of last-rendered word
let currentLetter   = "";
let currentWord     = "";
let currentSentence = "";

const ALPHA = new Set("ABCDEFGHIJKLMNOPQRSTUVWXYZ".split(""));

// ── Status ────────────────────────────────────────────────────────────────
function setStatus(cls, text) {
  els.statusBadge.classList.remove("is-live", "is-error");
  if (cls) els.statusBadge.classList.add(cls);
  els.statusText.textContent = text;
}

// ── Letter formatting ─────────────────────────────────────────────────────
function fmtLetter(l) {
  if (!l)                            return "—";
  if (ALPHA.has(l.toUpperCase()))    return l.toUpperCase();
  if (l === "space")                 return "␣";
  if (l === "del")                   return "⌫";
  if (l === "nothing")               return "·";
  return l;
}

// ── Word tiles ────────────────────────────────────────────────────────────
function renderWordTiles(rawWord) {
  const word = (rawWord || "").toUpperCase().trim();
  if (word === renderedWord) return;

  const container = els.wordTiles;
  const empty     = els.tilesEmpty;

  if (!word) {
    container.innerHTML = "";
    container.appendChild(empty);
    empty.style.display = "";
    renderedWord = "";
    return;
  }

  empty.style.display = "none";

  // Incremental append when letters are only added to the end
  if (word.startsWith(renderedWord) && word.length > renderedWord.length) {
    const tiles = container.querySelectorAll(".word-tile");
    if (tiles.length > 0) {
      tiles[tiles.length - 1].classList.remove("word-tile--latest");
    }
    for (let i = renderedWord.length; i < word.length; i++) {
      container.appendChild(makeTile(word[i], i === word.length - 1));
    }
  } else {
    // Full re-render (backspace, reset, etc.)
    container.innerHTML = "";
    container.appendChild(empty);
    empty.style.display = "none";
    for (let i = 0; i < word.length; i++) {
      container.appendChild(makeTile(word[i], i === word.length - 1));
    }
  }

  renderedWord = word;
}

function makeTile(letter, isLatest) {
  const el = document.createElement("span");
  el.className = "word-tile" + (isLatest ? " word-tile--latest" : "");
  el.textContent = letter;
  return el;
}

// ── Text UI ───────────────────────────────────────────────────────────────
function updateTextUI({ word, sentence, full_text: fullRaw }) {
  currentWord     = (word     || "").trim();
  currentSentence = (sentence || "").trim();
  const fullText  = (fullRaw  || "").trim();

  renderWordTiles(currentWord);

  els.sentenceText.textContent = currentSentence || "—";
  els.sentenceText.classList.toggle("is-empty", !currentSentence);

  els.fullText.textContent = fullText || "Start signing to build text…";
  els.fullText.classList.toggle("is-empty", !fullText);

  els.speakWordBtn.disabled     = !currentWord;
  els.speakSentenceBtn.disabled = !currentSentence;
  els.speakBtn.disabled         = !fullText;
}

// ── Prediction UI ─────────────────────────────────────────────────────────
function updatePrediction({ letter, confidence, committed }) {
  // Hold the last detected letter when a frame returns none (e.g. window warm-up)
  // instead of flickering back to "—"; the smoothed pipeline drives this value.
  if (letter) currentLetter = letter;

  const shown     = letter || currentLetter;
  const display   = fmtLetter(shown);
  const isNothing = !shown || shown === "nothing";
  const isCommitted = Boolean(committed && ALPHA.has(String(committed).toUpperCase()));

  els.letterDisplay.textContent = display;
  els.letterDisplay.classList.toggle("is-nothing",   isNothing);
  els.letterDisplay.classList.toggle("is-committed", isCommitted);

  // Glow ring: brighten when a real letter is actively detected
  const showGlow = !isNothing && confidence != null && confidence > 0.3;
  els.letterGlowRing.style.opacity   = showGlow ? Math.min(confidence, 1) : 0;
  els.letterGlowRing.style.background = isCommitted
    ? "radial-gradient(circle, rgba(0,207,168,.45) 0%, transparent 70%)"
    : "radial-gradient(circle, rgba(77,127,255,.38) 0%, transparent 70%)";

  const pct = confidence != null ? Math.round(confidence * 100) : 0;
  els.confidenceFill.style.width   = `${pct}%`;
  els.confidenceValue.textContent  = `${pct}%`;

  els.speakLetterBtn.disabled = isNothing || shown === "space" || shown === "del";

  if (isCommitted) showCommitFlash(fmtLetter(committed));
}

function showCommitFlash(symbol) {
  els.commitFlash.textContent = symbol;
  els.commitFlash.hidden      = false;
  els.commitFlash.style.animation = "none";
  void els.commitFlash.offsetWidth; // force reflow
  els.commitFlash.style.animation = "";
  setTimeout(() => { els.commitFlash.hidden = true; }, 650);
}

// ── Speech ────────────────────────────────────────────────────────────────
const allSpeakBtns = () => [
  els.speakLetterBtn, els.speakWordBtn, els.speakSentenceBtn, els.speakBtn,
];

function setSpeakingUI(activeBtn, on) {
  allSpeakBtns().forEach(btn => {
    btn.classList.toggle("is-speaking", on && btn === activeBtn);
    const span = btn.querySelector("span");
    if (span) {
      if (btn === activeBtn) {
        span.textContent = on ? "Speaking…" : btn.dataset.label;
      }
    }
  });
}

function speak(text, btn) {
  if (!text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();

  const utt    = new SpeechSynthesisUtterance(text);
  utt.rate     = 0.95;
  utt.pitch    = 1;
  utt.volume   = 1;
  utt.onstart  = () => setSpeakingUI(btn, true);
  utt.onend    = () => setSpeakingUI(btn, false);
  utt.onerror  = () => setSpeakingUI(btn, false);
  window.speechSynthesis.speak(utt);
}

function speakLetter() {
  if (!currentLetter || currentLetter === "nothing") return;
  const name =
    currentLetter === "space" ? "space" :
    currentLetter === "del"   ? "delete" :
    currentLetter.toUpperCase();
  speak(name, els.speakLetterBtn);
}

function speakWord() {
  if (!currentWord) return;
  speak(currentWord.toLowerCase(), els.speakWordBtn);
}

function speakSentence() {
  if (!currentSentence) return;
  speak(currentSentence.toLowerCase(), els.speakSentenceBtn);
}

function speakAll() {
  const text = els.fullText.textContent.trim();
  if (!text || text === "Start signing to build text…") return;
  speak(text.toLowerCase(), els.speakBtn);
}

// ── API helper ────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
  return res.json();
}

async function checkHealth() {
  try {
    const data = await api("/api/health");
    els.modelLabel.textContent = data.model_loaded
      ? "Model: ready"
      : "Model: missing — run train_model.py";
    setStatus("", "Ready");
  } catch {
    setStatus("is-error", "Backend offline");
    els.modelLabel.textContent = "Model: unavailable";
  }
}

// ── Frame capture ─────────────────────────────────────────────────────────
function captureFrame() {
  const v = els.webcam;
  if (!v.videoWidth || !v.videoHeight) return null;
  const c = els.captureCanvas;
  c.width  = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  return c.toDataURL("image/jpeg", 0.82);
}

// ── Processing loop ───────────────────────────────────────────────────────
async function processLoop() {
  if (!stream || processing) return;
  processing = true;
  try {
    const image = captureFrame();
    if (!image) return;

    const data = await api("/api/process-frame", {
      method: "POST",
      body: JSON.stringify({ image, require_motion: true }),
    });

    if (data.annotated_image) {
      els.annotatedFeed.src = `data:image/jpeg;base64,${data.annotated_image}`;
      els.annotatedFeed.classList.add("is-visible");
    }

    updatePrediction(data);
    updateTextUI(data);

    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
      els.fpsLabel.textContent = `${(frameCount / ((now - lastFrameTime) / 1000)).toFixed(1)} FPS`;
      frameCount   = 0;
      lastFrameTime = now;
    }
  } catch (err) {
    console.error(err);
    setStatus("is-error", "Processing error");
  } finally {
    processing = false;
  }
}

function startLoop() {
  stopLoop();
  const tick = () => { processLoop(); loopHandle = setTimeout(tick, 180); };
  tick();
}
function stopLoop() {
  if (loopHandle) { clearTimeout(loopHandle); loopHandle = null; }
}

// ── Camera ────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    els.webcam.srcObject = stream;
    els.cameraPlaceholder.classList.add("is-hidden");
    els.cameraHud.hidden = false;
    els.startBtn.disabled = true;
    els.stopBtn.disabled  = false;
    setStatus("is-live", "Live");
    startLoop();
  } catch (err) {
    console.error(err);
    setStatus("is-error", "Camera denied");
    alert("Could not access the camera. Please allow camera permissions and try again.");
  }
}

function stopCamera() {
  stopLoop();
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  els.webcam.srcObject = null;
  els.annotatedFeed.classList.remove("is-visible");
  els.annotatedFeed.removeAttribute("src");
  els.cameraPlaceholder.classList.remove("is-hidden");
  els.cameraHud.hidden  = true;
  els.startBtn.disabled = false;
  els.stopBtn.disabled  = true;
  setStatus("", "Stopped");
  els.fpsLabel.textContent = "— FPS";
}

async function postAction(path) {
  const data = await api(path, { method: "POST" });
  updateTextUI(data);
  return data;
}

// ── Seed speak-button labels (for restoring after speaking) ───────────────
document.querySelectorAll(".speak-btn").forEach(btn => {
  const span = btn.querySelector("span");
  if (span) btn.dataset.label = span.textContent;
});

// ── Event listeners ───────────────────────────────────────────────────────
els.startBtn.addEventListener("click", startCamera);
els.stopBtn.addEventListener("click", stopCamera);
els.finalizeBtn.addEventListener("click", () => postAction("/api/finalize-word"));
els.backspaceBtn.addEventListener("click", () => postAction("/api/backspace"));
els.clearBtn.addEventListener("click", () => postAction("/api/reset"));
els.speakLetterBtn.addEventListener("click", speakLetter);
els.speakWordBtn.addEventListener("click", speakWord);
els.speakSentenceBtn.addEventListener("click", speakSentence);
els.speakBtn.addEventListener("click", speakAll);

// ── Init ──────────────────────────────────────────────────────────────────
checkHealth();
