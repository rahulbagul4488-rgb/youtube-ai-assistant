"use strict";

const API_BASE = "https://youtube-ai-assistant-ikot.onrender.com";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusDot     = document.getElementById("status-dot");
const statusText    = document.getElementById("status-text");
const btnIngest     = document.getElementById("btn-ingest");
const chatMessages  = document.getElementById("chat-messages");
const questionInput = document.getElementById("question-input");
const btnSend       = document.getElementById("btn-send");

// ── State ─────────────────────────────────────────────────────────────────────
let currentVideoId = null;
let videoIndexed   = false;

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(text, color) {
  statusText.textContent = text;
  statusDot.className    = color || "";
}

function addMessage(text, role) {
  const placeholder = chatMessages.querySelector(".msg.system");
  if (placeholder) placeholder.remove();

  const div = document.createElement("div");
  div.className   = "msg " + (role || "assistant");
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function showTyping() {
  const div = document.createElement("div");
  div.className = "msg assistant typing";
  div.id        = "typing-indicator";
  div.innerHTML = "<span></span><span></span><span></span>";
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function enableChat() {
  questionInput.disabled = false;
  btnSend.disabled       = false;
  questionInput.focus();
}

function setBusy(isBusy) {
  btnIngest.disabled     = isBusy || !currentVideoId;
  btnSend.disabled       = isBusy || !videoIndexed;
  questionInput.disabled = isBusy || !videoIndexed;
}

// ── Backend health check ──────────────────────────────────────────────────────
async function checkBackend() {
  try {
    const res = await fetch(API_BASE + "/health", {
      signal: AbortSignal.timeout(3000)
    });
    return res.ok;
  } catch {
    return false;
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  setStatus("Connecting to backend…", "yellow");

  const backendOk = await checkBackend();
  if (!backendOk) {
    setStatus("Backend offline — start uvicorn!", "red");
    return;
  }

  // Active tab la directly message pathav video ID sathi
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (!tabs[0]) {
      setStatus("No active tab found.", "red");
      return;
    }

    chrome.tabs.sendMessage(
      tabs[0].id,
      { action: "getVideoId" },
      function (response) {
        if (chrome.runtime.lastError) {
          setStatus("Open a YouTube video first.", "red");
          return;
        }
        if (!response || !response.videoId) {
          setStatus("No video found on this page.", "red");
          return;
        }

        currentVideoId = response.videoId;
        setStatus("Video detected: " + currentVideoId, "green");
        btnIngest.disabled    = false;
        btnIngest.textContent = "⚡ Index This Video";
      }
    );
  });
}

// ── Ingest ────────────────────────────────────────────────────────────────────
btnIngest.addEventListener("click", async function () {
  if (!currentVideoId) return;

  setBusy(true);
  setStatus("Indexing video…", "yellow");
  btnIngest.textContent = "⏳ Indexing…";

  try {
    const res  = await fetch(API_BASE + "/ingest", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ video_id: currentVideoId }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "HTTP " + res.status);

    videoIndexed = true;

    const label = data.status === "already_exists" ? "Already indexed" : "Indexed";
    setStatus(label + " — " + data.chunk_count + " chunks ready", "green");
    btnIngest.textContent = "✅ Video Indexed";

    addMessage(
      data.status === "already_exists"
        ? "This video was already indexed (" + data.chunk_count + " chunks). Ask me anything!"
        : "Video indexed! Created " + data.chunk_count + " chunks. Ask me anything!",
      "assistant"
    );
    enableChat();

  } catch (err) {
    setStatus("Indexing failed!", "red");
    btnIngest.textContent = "⚡ Retry Indexing";
    addMessage("❌ Error: " + err.message, "system");
    setBusy(false);
    btnIngest.disabled = false;
  }
});

// ── Send question ─────────────────────────────────────────────────────────────
async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question || !videoIndexed) return;

  addMessage(question, "user");
  questionInput.value = "";
  questionInput.style.height = "40px";
  setBusy(true);
  showTyping();

  try {
    const res  = await fetch(API_BASE + "/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ video_id: currentVideoId, question: question }),
    });
    const data = await res.json();

    removeTyping();
    if (!res.ok) throw new Error(data.detail || "HTTP " + res.status);

    addMessage(data.answer, "assistant");

  } catch (err) {
    removeTyping();
    addMessage("❌ Error: " + err.message, "system");
  } finally {
    setBusy(false);
    if (videoIndexed) enableChat();
  }
}

btnSend.addEventListener("click", sendQuestion);

questionInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

questionInput.addEventListener("input", function () {
  questionInput.style.height = "40px";
  questionInput.style.height = questionInput.scrollHeight + "px";
});

// ── Start ─────────────────────────────────────────────────────────────────────
init();