async function apiCall(url, options = {}) {
  const headers = { ...(options.headers || {}) };

  if (API_BASE.includes('ngrok')) {
    headers["ngrok-skip-browser-warning"] = "true";
  }

  console.log('[API] Making request to:', url);
  console.log('[API] Headers:', headers);

  return fetch(url, {
    ...options,
    headers: headers
  });
}

const stateLanguageMap = {
  "Andhra Pradesh": "Telugu",
  "Arunachal Pradesh": "English",
  "Assam": "Assamese",
  "Bihar": "Hindi",
  "Chhattisgarh": "Hindi",
  "Goa": "Konkani",
  "Gujarat": "Gujarati",
  "Haryana": "Hindi",
  "Himachal Pradesh": "Hindi",
  "Jharkhand": "Hindi",
  "Karnataka": "Kannada",
  "Kerala": "Malayalam",
  "Madhya Pradesh": "Hindi",
  "Maharashtra": "Marathi",
  "Manipur": "Manipuri",
  "Meghalaya": "English",
  "Mizoram": "Mizo",
  "Nagaland": "English",
  "Odisha": "Odia",
  "Punjab": "Punjabi",
  "Rajasthan": "Hindi",
  "Sikkim": "Nepali",
  "Tamil Nadu": "Tamil",
  "Telangana": "Telugu",
  "Tripura": "Bengali",
  "Uttar Pradesh": "Hindi",
  "Uttarakhand": "Hindi",
  "West Bengal": "Bengali",
  "Delhi": "Hindi",
  "Jammu and Kashmir": "Urdu",
  "Ladakh": "Urdu"
};
const allIndianLanguages = [
  "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada",
  "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi",
  "Nepali", "Odia", "Punjabi", "Sanskrit", "Santali", "Sindhi", "Tamil",
  "Telugu", "Urdu", "English"
];


function populateStateDropdowns() {
  const stateSelects = [document.getElementById("manualStateSelect"), document.getElementById("stateSelect")];
  const langSelect = document.getElementById("manualLanguageSelect");

  stateSelects.forEach(select => {
    if (select) {
      select.innerHTML = '<option value="">--Select State--</option>';
      Object.keys(stateLanguageMap).forEach((state) => {
        const opt = document.createElement("option");
        opt.value = state;
        opt.textContent = state;
        select.appendChild(opt);
      });
    }
  });

  if (langSelect) {
    langSelect.innerHTML = '<option value="">--Select Language--</option>';
    allIndianLanguages.forEach((lang) => {
      const opt = document.createElement("option");
      opt.value = lang;
      opt.textContent = lang;
      langSelect.appendChild(opt);
    });
  }
}



let deviceId = localStorage.getItem("agrichat_device_id");
if (!deviceId) {
  deviceId = crypto.randomUUID();
  localStorage.setItem("agrichat_device_id", deviceId);
}
let currentSession = null;
let showingArchived = false;

async function loadSessions() {
  try {
    console.log('[SESSIONS] Loading sessions for device:', deviceId);
    const res = await apiCall(`${API_BASE}/sessions?device_id=${deviceId}`);

    console.log('[SESSIONS] API response status:', res.status);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }

    const data = await res.json();
    console.log('[SESSIONS] Parsed data:', data);

    const { sessions } = data;
    if (!sessions || !Array.isArray(sessions)) {
      throw new Error('Invalid sessions data structure');
    }

    const activeDiv = document.getElementById("activeSessions");
    const archivedDiv = document.getElementById("archivedSessions");
    activeDiv.innerHTML = "";
    archivedDiv.innerHTML = "";

    let activeCount = 0;
    let archivedCount = 0;

    sessions.forEach((s) => {

      const isActiveSession = currentSession?.session_id === s.session_id;
      const container = document.createElement("div");
      container.className = `session-entry  ${s.status === 'archived' ? 'archived' : ''} ${isActiveSession ? 'active' : ''}`;
      container.innerHTML = `
    <a href="#" class="session-link">
      <div class="session-date">
        <i class="fas fa-calendar"></i> ${new Date(s.timestamp).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
      </div>
      <div class="session-preview">${s.messages?.[0]?.question?.slice(0, 50) || ""}...</div>
      <div class="badges">
        <span class="badge">${s.crop}</span>
        <span class="badge">${s.state}</span>
        ${s.has_unread ? `<span class="badge" style="background:#ff6b35;">New</span>` : ""}
      </div>
    </a>
    <div class="session-actions">
      <button class="btn-delete" onclick="deleteSession('${s.session_id}')"><i class="fas fa-trash"></i> Delete
      </button>
      <button class="${s.status === "archived" ? "btn-unarchive" : "btn-archive"}" onclick="toggleSessionStatus('${s.session_id}', '${s.status}')">
        <i class="fas ${s.status === "archived" ? "fa-undo" : "fa-archive"}"></i>
        ${s.status === "archived" ? "Restore" : "Archive"}
      </button>
    </div>
  `;

      container.querySelector(".session-link").addEventListener("click", async () => {
        const resp = await apiCall(`${API_BASE}/session/${s.session_id}`);
        const { session } = await resp.json();
        currentSession = session;
        loadChat(currentSession);
        loadSessions();
      });

      if (s.status === "archived") {
        archivedDiv.appendChild(container);
        archivedCount++;
      } else {
        activeDiv.appendChild(container);
        activeCount++;
      }
    });
    document.getElementById("noSessions").style.display = (activeCount + archivedCount === 0) ? "block" : "none";

    console.log('[SESSIONS] Successfully loaded', sessions.length, 'sessions');
  } catch (error) {
    console.error('[SESSIONS] Failed to load sessions:', error);
    console.error('[SESSIONS] Error details:', error.message);

    // Show error message to user
    const activeDiv = document.getElementById("activeSessions");
    const archivedDiv = document.getElementById("archivedSessions");
    activeDiv.innerHTML = '<div class="error-message">Failed to load sessions. Please refresh the page.</div>';
    archivedDiv.innerHTML = '';

    // Still show the no sessions message
    document.getElementById("noSessions").style.display = "block";
  }
}

function toggleView() {
  showingArchived = !showingArchived;
  document.getElementById("viewToggleText").textContent = showingArchived ? "Archived" : "Active";
  document.getElementById("activeSessions").style.display = showingArchived ? "none" : "block";
  document.getElementById("archivedSessions").style.display = showingArchived ? "block" : "none";
}

async function toggleSessionStatus(session_id, currentStatus) {
  const action = currentStatus === "archived" ? "restore" : "archive";
  const confirmed = confirm(`Are you sure you want to ${action} this session?`);
  if (!confirmed) return;
  await apiCall(`${API_BASE}/toggle-status/${session_id}/${currentStatus}`, { method: "POST" });
  currentStatus = currentStatus === "archived" ? "active" : "archived";
  if (currentStatus === "archived") {
    currentSession = null;
    document.getElementById("chatWindow").style.display = "none";
    document.getElementById("chat-form").style.display = "none";
    document.getElementById("archivedNotice").style.display = "none";
    document.getElementById("exportBtn").style.display = "none";
    document.getElementById("startScreen").style.display = "block";
  } else {
    const headers = {};
    if (API_BASE.includes('ngrok')) {
      headers["ngrok-skip-browser-warning"] = "true";
    }

    const resp = await fetch(`${API_BASE}/session/${session_id}`, {
      headers: headers
    });
    const { session } = await resp.json();
    currentSession = session;
    document.getElementById("viewToggleText").textContent = "Active";
    document.getElementById("activeSessions").style.display = "block";
    document.getElementById("archivedSessions").style.display = "none";
    loadChat(currentSession);
  }
  loadSessions();
}

async function deleteSession(session_id) {
  const confirmed = confirm("Are you sure you want to delete this session?");
  if (!confirmed) return;

  const headers = {};
  if (API_BASE.includes('ngrok')) {
    headers["ngrok-skip-browser-warning"] = "true";
  }

  await fetch(`${API_BASE}/delete-session/${session_id}`, {
    method: "DELETE",
    headers: headers
  });
  if (currentSession?.session_id === session_id) {
    currentSession = null;
    document.getElementById("chatWindow").style.display = "none";
    document.getElementById("chat-form").style.display = "none";
    document.getElementById("archivedNotice").style.display = "none";
    document.getElementById("exportBtn").style.display = "none";
    document.getElementById("startScreen").style.display = "block";
  }
  loadSessions();
}

async function rateAnswer(index, rating, btn) {
  if (!currentSession) return;

  const formData = new FormData();
  formData.append("question_index", index);
  formData.append("rating", rating);

  const headers = {};
  if (API_BASE.includes('ngrok')) {
    headers["ngrok-skip-browser-warning"] = "true";
  }

  await fetch(`${API_BASE}/session/${currentSession.session_id}/rate`, {
    method: "POST",
    headers: headers,
    body: formData,
  });

  const messageDiv = btn.closest(".message");
  const upBtn = messageDiv.querySelector(".thumbs-up");
  const downBtn = messageDiv.querySelector(".thumbs-down");

  upBtn.classList.remove("selected");
  downBtn.classList.remove("selected");

  if (rating === "up") {
    upBtn.classList.add("selected");
  } else {
    downBtn.classList.add("selected");
  }
}

function copyToClipboard(button) {
  const answer = button.closest(".message").querySelector(".bot-answer").innerText;
  navigator.clipboard.writeText(answer).then(() => {
    button.classList.add('selected');

    button.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 512 512">
        <path fill="currentColor" d="M504.5 75.5c-10-10-26.2-10-36.2 0L184 359.8l-140.3-140.3
        c-10-10-26.2-10-36.2 0s-10 26.2 0 36.2l160 160c10 10 26.2 10 36.2 0l320-320c10-10
        10-26.2 0-36.2z"/>
      </svg>
    `;

    setTimeout(() => {
      button.classList.remove('selected');

      button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 448 512">
          <path fill="currentColor" d="M320 448v40c0 13.255-10.745 24-24 24H24c-13.255 0-24-10.745-24-24V120c0-13.255
          10.745-24 24-24h72v296c0 30.879 25.121 56 56 56h168zm0-344V0H152c-13.255 0-24
          10.745-24 24v368c0 13.255 10.745 24 24 24h272c13.255 0 24-10.745 24-24V128H344c-13.2
          0-24-10.8-24-24zm120.971-31.029L375.029 7.029A24 24 0 0 0 358.059 0H352v96h96v-6.059a24
          24 0 0 0-7.029-16.97z"/>
        </svg>
      `;
    }, 800);
  });
}



function appendMessage(sender, text, index = null, rating = null) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;

  if (sender === "user") {
    div.innerHTML = ` ${text}`;
  } else {
    div.innerHTML = `
    <div class="bot-answer">${text}</div>
    <div class="message-actions">
      <button class="copy-btn" onclick="copyToClipboard(this)" title="Copy">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 448 512">
          <path fill="currentColor" d="M320 448v40c0 13.255-10.745 24-24 24H24c-13.255 0-24-10.745-24-24V120c0-13.255 10.745-24 24-24h72v296c0 30.879 25.121 56 56 56h168zm0-344V0H152c-13.255 0-24 10.745-24 24v368c0 13.255 10.745 24 24 24h272c13.255 0 24-10.745 24-24V128H344c-13.2 0-24-10.8-24-24zm120.971-31.029L375.029 7.029A24 24 0 0 0 358.059 0H352v96h96v-6.059a24 24 0 0 0-7.029-16.97z"/>
        </svg>
      </button>
      <button class="rate-btn thumbs-up ${rating === 'up' ? 'selected' : ''}" onclick="rateAnswer(${index}, 'up', this)" title="Thumbs up">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">
          <path fill="currentColor" fill-rule="evenodd" d="M15 9.7h4a2 2 0 0 1 1.6.9a2 2 0 0 1 .3 1.8l-2.4 7.2c-.3.9-.5 1.4-1.9 1.4c-2 0-4.2-.7-6.1-1.3L9 19.3V9.5A32 32 0 0 0 13.2 4c.1-.4.5-.7.9-.9h1.2c.4.1.7.4 1 .7l.2 1.3zM4.2 10H7v8a2 2 0 1 1-4 0v-6.8c0-.7.5-1.2 1.2-1.2" clip-rule="evenodd"/>
        </svg>
      </button>
      <button class="rate-btn thumbs-down ${rating === 'down' ? 'selected' : ''}" onclick="rateAnswer(${index}, 'down', this)" title="Thumbs down">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">
          <path fill="currentColor" fill-rule="evenodd" d="M9 14.3H5a2 2 0 0 1-1.6-.9a2 2 0 0 1-.3-1.8l2.4-7.2C5.8 3.5 6 3 7.4 3c2 0 4.2.7 6.1 1.3l1.4.4v9.8a32 32 0 0 0-4.2 5.5c-.1.4-.5.7-.9.9a1.7 1.7 0 0 1-2.1-.7c-.2-.4-.3-.8-.3-1.3zm10.8-.3H17V6a2 2 0 1 1 4 0v6.8c0 .7-.5 1.2-1.2 1.2" clip-rule="evenodd"/>
        </svg>
      </button>
    </div>
  `;
  }


  document.getElementById("chatWindow").appendChild(div);
}

function displayRecommendations(recommendations) {
  const existingRecs = document.querySelectorAll('.inline-recommendations');
  existingRecs.forEach(rec => rec.remove());

  if (!recommendations || recommendations.length === 0) {
    return;
  }

  const recDiv = document.createElement("div");
  recDiv.className = "inline-recommendations";
  recDiv.innerHTML = `
    <div class="inline-rec-header">
      <i class="fas fa-lightbulb"></i>
      <span>You might also ask:</span>
    </div>
    <div class="inline-rec-items">
      ${recommendations.map(rec => `
        <div class="inline-rec-item" data-question="${rec.question}">
          <span class="inline-rec-question">${rec.question}</span>
          <span class="inline-rec-score">${(rec.similarity_score * 100).toFixed(0)}%</span>
        </div>
      `).join('')}
    </div>
  `;

  recDiv.addEventListener('click', (e) => {
    const item = e.target.closest('.inline-rec-item');
    if (item) {
      const question = item.getAttribute('data-question');
      document.getElementById("user-input").value = question;
      document.getElementById("user-input").focus();
    }
  });

  const chatWindow = document.getElementById("chatWindow");
  chatWindow.appendChild(recDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function loadChat(session) {
  document.getElementById("startScreen").style.display = "none";
  document.getElementById("chatWindow").style.display = "block";
  document.getElementById("chatWindow").innerHTML = "";
  document.getElementById("exportBtn").style.display = "inline-block";
  document.getElementById("locationEdit").style.display = "none";

  if (window.innerWidth > 768) {
    document.querySelector('.main-header').classList.add('chat-active-width');
  } else {

    document.querySelector('.main-header').classList.remove('chat-active-width');
  }

  if (!session || typeof session.status === "undefined" || !Array.isArray(session.messages)) {
    alert("Could not load chat session (data missing or malformed).");
    return;
  }

  if (session.status === "archived") {
    document.getElementById("archivedNotice").style.display = "block";
    document.getElementById("chat-form").style.display = "none";
  } else {
    document.getElementById("archivedNotice").style.display = "none";
    document.getElementById("chat-form").style.display = "flex";
  }

  session.messages.forEach((msg, idx) => {
    appendMessage("user", msg.question);
    appendMessage("bot", msg.answer, idx, msg.rating || null);
  });

  if (session.recommendations && session.recommendations.length > 0) {
    displayRecommendations(session.recommendations);
  }

  document.getElementById("recommendationsSection").style.display = "none";

  document.getElementById("chatWindow").scrollTop = document.getElementById("chatWindow").scrollHeight;
}

window.addEventListener("DOMContentLoaded", async () => {
  populateStateDropdowns();
  detectLocationAndLanguage();
  const savedState = localStorage.getItem("agrichat_user_state");
  if (savedState) {
    const stateSelect = document.getElementById("stateSelect");
    if (stateSelect) stateSelect.value = savedState;
  }

  try {
    await loadSessions();
  } catch (error) {
    console.error('[INIT] Failed to load sessions on page load:', error);
    // Continue with page initialization even if sessions fail to load
  }

  document.getElementById("editLocationBtn").addEventListener("click", () => {
    const locationEdit = document.getElementById("locationEdit");
    const isHidden = locationEdit.style.display === "none" || locationEdit.style.display === "";
    locationEdit.style.display = isHidden ? "flex" : "none";
  });
  document.getElementById("new-session-btn").addEventListener("click", () => {
    currentSession = null;
    document.getElementById("startScreen").style.display = "block";
    document.getElementById("chatWindow").style.display = "none";
    document.getElementById("chat-form").style.display = "none";
    document.getElementById("exportBtn").style.display = "none";
    document.getElementById("archivedNotice").style.display = "none";
  });

  document.getElementById("viewToggle").addEventListener("click", toggleView);

  document.getElementById("start-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    //     const textarea = document.getElementById("start-input");
    //     // const textarea = e.target.querySelector("textarea");
    //     const question = textarea.value.trim();
    //     // const state = document.getElementById("stateSelect").value;

    //     const stateSelect = document.getElementById("stateSelect");
    //     const state = stateSelect ? stateSelect.value : "none"; // or "" as default

    //     // if (!state) {
    //     //   alert("Please select your state.");
    //     //   return;
    //     // }
    //     localStorage.setItem("agrichat_user_state", state); 
    // (NEW) After the audioBlob is ready...
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });



    console.log("Start form found:", document.getElementById("start-form"));

    const textarea = e.target.querySelector("textarea");
    const question = textarea.value.trim();
    const state = document.getElementById("manualStateSelect").value;

    if (!state) {
      alert("Please select your state.");
      return;
    }
    const lang = stateLanguageMap[state] || "English";
    localStorage.setItem("agrichat_user_state", state);
    localStorage.setItem("agrichat_user_language", lang);
    showLoader();
    await updateLanguageInBackend(state, lang);

    if (!question) return;

    const formData = new FormData();
    formData.append("question", question);
    formData.append("device_id", deviceId);
    formData.append("state", state);
    formData.append("language", lang);
    formData.append("file", audioBlob, "recording.webm");

    showLoader();
    const res = await apiCall(`${API_BASE}/query-form`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    console.log('API response:', data);
    currentSession = data.session;
    loadChat(currentSession);
    hideLoader();
    loadSessions();
    textarea.value = "";
  });

  document.getElementById("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.getElementById("user-input");
    const question = input.value.trim();
    if (!question || !currentSession) return;

    appendMessage("user", question);
    input.value = "";

    const formData = new FormData();
    formData.append("question", question);
    formData.append("device_id", deviceId);
    formData.append("state", localStorage.getItem("agrichat_user_state") || "");

    showLoader();
    const res = await apiCall(`${API_BASE}/session/${currentSession.session_id}/query`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    const last = data.session.messages.at(-1);
    hideLoader();
    appendMessage("bot", last.answer);

    currentSession = data.session;

    if (currentSession.recommendations && currentSession.recommendations.length > 0) {
      displayRecommendations(currentSession.recommendations);
    }
  });

  document.getElementById("restoreBtn").addEventListener("click", async () => {
    if (currentSession) {
      toggleSessionStatus(currentSession.session_id, "archived");
    }
  });


  document.getElementById("exportBtn").addEventListener("click", () => {
    if (currentSession) {
      window.open(`${API_BASE}/export/csv/${currentSession.session_id}`, "_blank");
    }
  });
});

function showLoader() {
  const loadingOverlay = document.getElementById("loadingOverlay");
  loadingOverlay.style.display = "flex";

  let thinkingText = document.getElementById("thinkingText");
  if (!thinkingText) {
    thinkingText = document.createElement("div");
    thinkingText.id = "thinkingText";
    thinkingText.style.cssText = `
      color: #4CAF50;
      font-size: 16px;
      margin-top: 20px;
      text-align: center;
      font-weight: 500;
    `;
    loadingOverlay.appendChild(thinkingText);
  }

  const thinkingStates = [
    "Understanding your question...",
    "Searching agricultural database...",
    "Processing with AI...",
    "Generating response..."
  ];

  let currentState = 0;
  thinkingText.textContent = thinkingStates[0];


  loadingOverlay.thinkingInterval = setInterval(() => {
    currentState = (currentState + 1) % thinkingStates.length;
    thinkingText.textContent = thinkingStates[currentState];
  }, 1500);
}

function hideLoader() {
  const loadingOverlay = document.getElementById("loadingOverlay");
  loadingOverlay.style.display = "none";

  if (loadingOverlay.thinkingInterval) {
    clearInterval(loadingOverlay.thinkingInterval);
    loadingOverlay.thinkingInterval = null;
  }
}

async function detectLocationAndLanguage(updateBackend = false) {
  if (!navigator.geolocation) return;

  navigator.geolocation.getCurrentPosition(async (position) => {
    const lat = position.coords.latitude;
    const lon = position.coords.longitude;
    console.log("Coordinates:", lat, lon);
    const response = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`);
    const data = await response.json();
    console.log("Reverse geocoded data:", data);
    const state = data.address.state || data.address.county || "Unknown";
    const language = stateLanguageMap[state] || "English";
    console.log("Detected:", state, language);
    localStorage.setItem("agrichat_user_state", state);
    localStorage.setItem("agrichat_user_language", language);

    updateLocationUI(state, language);

    if (updateBackend) {
      await updateLanguageInBackend(state, language);
    }
  });
}

function updateLocationUI(state, language) {
  document.getElementById("locationState").textContent = state;
  document.getElementById("locationLanguage").textContent = language;

  const manualStateSelect = document.getElementById("manualStateSelect");
  const manualLangSelect = document.getElementById("manualLanguageSelect");

  if (manualStateSelect) manualStateSelect.value = state;
  if (manualLangSelect) manualLangSelect.value = language;
}


async function updateLanguageInBackend(state, language) {
  try {
    const response = await apiCall(`${API_BASE}/update-language`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        device_id: deviceId,
        state: state,
        language: language
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    console.log("Language updated successfully");
  } catch (error) {
    console.error("Error updating language:", error);
  }
}


document.getElementById("manualStateSelect").addEventListener("change", async (e) => {
  const selectedState = e.target.value;
  const defaultLang = stateLanguageMap[selectedState] || "Hindi";
  const selectedLang = document.getElementById("manualLanguageSelect").value || defaultLang;

  localStorage.setItem("agrichat_user_state", selectedState);
  localStorage.setItem("agrichat_user_language", selectedLang);

  updateLocationUI(selectedState, selectedLang);
  await updateLanguageInBackend(selectedState, selectedLang);
});

document.getElementById("manualLanguageSelect").addEventListener("change", async (e) => {
  const selectedLang = e.target.value;
  const selectedState = document.getElementById("manualStateSelect").value;

  localStorage.setItem("agrichat_user_language", selectedLang);
  if (selectedState) localStorage.setItem("agrichat_user_state", selectedState);

  updateLocationUI(selectedState, selectedLang);
  if (selectedState) await updateLanguageInBackend(selectedState, selectedLang);
});

document.getElementById("resetLocationBtn").addEventListener("click", async () => {
  await detectLocationAndLanguage(true);
  document.getElementById("locationEdit").style.display = "none";
});

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const startFormVoiceBtn = document.querySelector("#start-form .feature-button[aria-label='Start voice input']");
const chatFormVoiceBtn = document.querySelector("#chat-form .feature-button[aria-label='Start voice input']");

function initializeVoiceRecording() {
  if (startFormVoiceBtn) {
    startFormVoiceBtn.addEventListener("click", (e) => {
      e.preventDefault();
      handleVoiceInput(document.getElementById("start-input"));
    });
  }

  if (chatFormVoiceBtn) {
    chatFormVoiceBtn.addEventListener("click", (e) => {
      e.preventDefault();
      handleVoiceInput(document.getElementById("user-input"));
    });
  }
}

async function handleVoiceInput(targetTextarea) {
  if (!isRecording) {
    await startRecording(targetTextarea);
  } else {
    stopRecording();
  }
}

async function startRecording(targetTextarea) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });

    audioChunks = [];
    isRecording = true;

    updateVoiceButtonState(true);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      isRecording = false;
      updateVoiceButtonState(false, true);
      stream.getTracks().forEach(track => track.stop());

      try {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

        const transcript = await transcribeAudio(audioBlob);

        if (transcript && transcript.trim()) {
          targetTextarea.value = transcript;

          if (targetTextarea.id === 'start-input') {
            document.getElementById('start-form').dispatchEvent(new Event('submit'));
          }
        } else {
          showNotification('No speech detected. Please try again.', 'warning');
        }
      } catch (error) {
        console.error('Transcription error:', error);
        showNotification('Voice transcription failed. Please try again.', 'error');
      } finally {
        updateVoiceButtonState(false);
      }
    };

    mediaRecorder.onerror = (event) => {
      console.error('MediaRecorder error:', event.error);
      isRecording = false;
      updateVoiceButtonState(false);
      showNotification('Recording error. Please try again.', 'error');
    };

    mediaRecorder.start();
    showNotification('Recording started. Click the microphone again to stop.', 'info');

  } catch (error) {
    console.error('Error starting recording:', error);
    isRecording = false;
    updateVoiceButtonState(false);

    if (error.name === 'NotAllowedError') {
      showNotification('Microphone access denied. Please allow microphone access and try again.', 'error');
    } else {
      showNotification('Could not start recording. Please check your microphone.', 'error');
    }
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    showNotification('Recording stopped. Processing...', 'info');
  }
}

async function transcribeAudio(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.webm');

  const headers = {};
  if (API_BASE.includes('ngrok')) {
    headers["ngrok-skip-browser-warning"] = "true";
  }

  const response = await fetch(`${API_BASE}/transcribe-audio`, {
    method: 'POST',
    headers: headers,
    body: formData
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Transcription failed');
  }

  const result = await response.json();
  return result.transcript;
}

function updateVoiceButtonState(recording, processing = false) {
  const buttons = [startFormVoiceBtn, chatFormVoiceBtn].filter(Boolean);

  buttons.forEach(button => {
    if (recording) {
      button.classList.add('recording');
      button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <rect x="6" y="6" width="12" height="12" rx="2"/>
        </svg>
      `;
      button.setAttribute('aria-label', 'Stop recording');
    } else if (processing) {
      button.classList.remove('recording');
      button.classList.add('processing');
      button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
      `;
      button.setAttribute('aria-label', 'Processing...');
    } else {
      button.classList.remove('recording', 'processing');
      button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <g>
            <path d="M5 3a3 3 0 0 1 6 0v5a3 3 0 0 1-6 0V3z"/>
            <path d="M3.5 6.5A.5.5 0 0 1 4 7v1a4 4 0 0 0 8 0V7a.5.5 0 0 1 1 0v1a5 5 0 0 1-4.5 4.975V15h3a.5.5 0 0 1 0 1h-7a.5.5 0 0 1 0-1h3v-2.025A5 5 0 0 1 3 8V7a.5.5 0 0 1 .5-.5z"/>
          </g>
        </svg>
      `;
      button.setAttribute('aria-label', 'Start voice input');
    }
  });
}

function showNotification(message, type = 'info') {

  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.innerHTML = `
    <div class="notification-content">
      <span class="notification-message">${message}</span>
      <button class="notification-close">&times;</button>
    </div>
  `;

  document.body.appendChild(notification);

  setTimeout(() => notification.classList.add('show'), 100);

  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => notification.remove(), 300);
  }, 4000);

  notification.querySelector('.notification-close').addEventListener('click', () => {
    notification.classList.remove('show');
    setTimeout(() => notification.remove(), 300);
  });
}

document.addEventListener('DOMContentLoaded', initializeVoiceRecording);
