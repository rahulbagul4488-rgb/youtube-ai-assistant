"use strict";

const API_BASE = "https://youtube-ai-assistant-ikot.onrender.com";

const statusDot     = document.getElementById("status-dot");
const statusText    = document.getElementById("status-text");
const btnIngest     = document.getElementById("btn-ingest");
const chatMessages  = document.getElementById("chat-messages");
const questionInput = document.getElementById("question-input");
const btnSend       = document.getElementById("btn-send");

let currentVideoId = null;
let videoIndexed   = false;

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

async function checkBackend() {
  try {
    const res = await fetch(API_BASE + "/health", {
      signal: AbortSignal.timeout(5000)
    });
    return res.ok;
  } catch {
    return false;
  }
}

async function fetchTranscriptFromPage(videoId) {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]) return reject(new Error("No active tab"));
      chrome.scripting.executeScript(
        {
          target: { tabId: tabs[0].id },
          func: async (vid) => {
            try {
              const pageRes  = await fetch(`https://www.youtube.com/watch?v=${vid}`);
              const pageHtml = await pageRes.text();
              const match = pageHtml.match(/"captionTracks":\[.*?"baseUrl":"(.*?)"/);
              if (!match) return { error: "No captions found for this video." };
              const captionUrl = match[1].replace(/\\u0026/g, "&");
              const xmlRes     = await fetch(captionUrl);
              const xmlText    = await xmlRes.text();
              const parser     = new DOMParser();
              const xmlDoc     = parser.parseFromString(xmlText, "text/xml");
              const texts      = xmlDoc.getElementsByTagName("text");
              const transcript = Array.from(texts)
                .map(t => t.textContent
                  .replace(/&amp;/g, "&")
                  .replace(/&lt;/g, "<")
                  .replace(/&gt;/g, ">")
                  .replace(/&#39;/g, "'")
                  .replace(/&quot;/g, '"')
                  .trim()
                )
                .filter(Boolean)
                .join(" ");
              return { transcript };
            } catch (e) {
              return { error: e.message };
            }
          },
          args: [videoId],
        },
        (results) => {
          if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
          const result = results?.[0]?.result;
          if (!result)             return reject(new Error("Script execution failed."));
          if (result.error)        return reject(new Error(result.error));
          if (!result.transcript)  return reject(new Error("Empty transcript."));
          resolve(result.transcript);
        }
      );
    });
  });
}

async function init() {
  setStatus("Connecting to backend…", "yellow");
  const backendOk = await checkBackend();
  if (!backendOk) {
    setStatus("Backend offline!", "red");
    return;
  }
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]) { setStatus("No active tab.", "red"); return; }
    chrome.tabs.sendMessage(tabs[0].id, { action: "getVideoId" }, (response) => {
      if (chrome.runtime.lastError || !response || !response.videoId) {
        setStatus("Open a YouTube video first.", "red");
        return;
      }
      currentVideoId = response.videoId;
      setStatus("Video detected: " + currentVideoId, "green");
      btnIngest.disabled    = false;
      btnIngest.textContent = "⚡ Index This Video";
    });
  });
}

btnIngest.addEventListener("click", async () => {
  if (!currentVideoId) return;
  setBusy(true);
  setStatus("Fetching transcript…", "yellow");
  btnIngest.textContent = "⏳ Fetching transcript…";

  let transcript;
  try {
    transcript = await fetchTranscriptFromPage(currentVideoId);
  } catch (err) {
    setStatus("Transcript failed!", "red");
    btnIngest.textContent = "⚡ Retry";
    addMessage("❌ Could not get transcript: " + err.message, "system");
    setBusy(false);
    btnIngest.disabled = false;
    return;
  }

  setStatus("Indexing video…", "yellow");
  btnIngest.textContent = "⏳ Indexing…";

  try {
    const res  = await fetch(API_BASE + "/ingest", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ video_id: currentVideoId, transcript }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "HTTP " + res.status);
    videoIndexed = true;
    const label  = data.status === "already_exists" ? "Already indexed" : "Indexed";
    setStatus(label + " — " + data.chunk_count + " chunks", "green");
    btnIngest.textContent = "✅ Video Indexed";
    addMessage(
      data.status === "already_exists"
        ? "Already indexed (" + data.chunk_count + " chunks). Ask anything!"
        : "Indexed! " + data.chunk_count + " chunks created. Ask anything!",
      "assistant"
    );
    enableChat();
  } catch (err) {
    setStatus("Indexing failed!", "red");
    btnIngest.textContent = "⚡ Retry";
    addMessage("❌ " + err.message, "system");
    setBusy(false);
    btnIngest.disabled = false;
  }
});

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question || !videoIndexed) return;
  addMessage(question, "user");
  questionInput.value        = "";
  questionInput.style.height = "40px";
  setBusy(true);
  showTyping();
  try {
    const res  = await fetch(API_BASE + "/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ video_id: currentVideoId, question }),
    });
    const data = await res.json();
    removeTyping();
    if (!res.ok) throw new Error(data.detail || "HTTP " + res.status);
    addMessage(data.answer, "assistant");
  } catch (err) {
    removeTyping();
    addMessage("❌ " + err.message, "system");
  } finally {
    setBusy(false);
    if (videoIndexed) enableChat();
  }
}

btnSend.addEventListener("click", sendQuestion);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "40px";
  questionInput.style.height = questionInput.scrollHeight + "px";
});

init();
