const API_BASE = "https://agrichat-annam.onrender.com/api";
// const API_BASE = "http://localhost:8000/api";

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
  const res = await fetch(`${API_BASE}/sessions?device_id=${deviceId}`);
  const { sessions } = await res.json();

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
    const resp = await fetch(`${API_BASE}/session/${s.session_id}`);
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
  await fetch(`${API_BASE}/toggle-status/${session_id}/${currentStatus}`, { method: "POST" });
  currentStatus = currentStatus === "archived" ? "active" : "archived";
    if (currentStatus === "archived") {
      currentSession = null;
      document.getElementById("chatWindow").style.display = "none";
      document.getElementById("chat-form").style.display = "none";
      document.getElementById("archivedNotice").style.display = "none";
      document.getElementById("exportBtn").style.display = "none";
      document.getElementById("startScreen").style.display = "block";
    } else {
      const resp = await fetch(`${API_BASE}/session/${session_id}`);
    const { session } = await resp.json();
    currentSession = session;
    document.getElementById("viewToggleText").textContent = "Active";
  document.getElementById("activeSessions").style.display = "block";
  document.getElementById("archivedSessions").style.display = "none";
    loadChat(currentSession);
    }
  loadSessions();
}

async function deleteSession(session_id){
  const confirmed = confirm("Are you sure you want to delete this session?");
  if (!confirmed) return;
  await fetch(`${API_BASE}/delete-session/${session_id}`,{
    method:"DELETE"
  });
  if(currentSession?.session_id === session_id){
    currentSession=null;
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

  await fetch(`${API_BASE}/session/${currentSession.session_id}/rate`, {
    method: "POST",
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


// function copyToClipboard(button) {
//   const answer = button.closest(".message").querySelector(".bot-answer").innerText;
//   navigator.clipboard.writeText(answer).then(() => {
//     button.innerHTML = '<i class="fas fa-check"></i>';
//     setTimeout(() => {
//       button.innerHTML = '<i class="fas fa-copy"></i>';
//     }, 1000);
//   });
// }


function copyToClipboard(button) {
  const answer = button.closest(".message").querySelector(".bot-answer").innerText;
  navigator.clipboard.writeText(answer).then(() => {
    // Add the selected class to flash grey
    button.classList.add('selected');

    // Swap icon for checkmark SVG (optional, but users like feedback)
    button.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 512 512">
        <path fill="currentColor" d="M504.5 75.5c-10-10-26.2-10-36.2 0L184 359.8l-140.3-140.3
        c-10-10-26.2-10-36.2 0s-10 26.2 0 36.2l160 160c10 10 26.2 10 36.2 0l320-320c10-10
        10-26.2 0-36.2z"/>
      </svg>
    `;

    setTimeout(() => {
      button.classList.remove('selected');
      
      // Restore the Copy svg icon
      button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 448 512">
          <path fill="currentColor" d="M320 448v40c0 13.255-10.745 24-24 24H24c-13.255 0-24-10.745-24-24V120c0-13.255
          10.745-24 24-24h72v296c0 30.879 25.121 56 56 56h168zm0-344V0H152c-13.255 0-24
          10.745-24 24v368c0 13.255 10.745 24 24 24h272c13.255 0 24-10.745 24-24V128H344c-13.2
          0-24-10.8-24-24zm120.971-31.029L375.029 7.029A24 24 0 0 0 358.059 0H352v96h96v-6.059a24
          24 0 0 0-7.029-16.97z"/>
        </svg>
      `;
    }, 800); // stays grey for 0.8 seconds
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


function loadChat(session) {
  document.getElementById("startScreen").style.display = "none";
  document.getElementById("chatWindow").style.display = "block";
  document.getElementById("chatWindow").innerHTML = "";
  document.getElementById("exportBtn").style.display = "inline-block";

  // new
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


  document.getElementById("chatWindow").scrollTop = document.getElementById("chatWindow").scrollHeight;
}

window.addEventListener("DOMContentLoaded", () => {
  populateStateDropdowns();
  detectLocationAndLanguage();
  const savedState = localStorage.getItem("agrichat_user_state");
  if (savedState) {
    const stateSelect = document.getElementById("stateSelect");
    if (stateSelect) stateSelect.value = savedState;
  }

  loadSessions();

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

    showLoader();
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    console.log('API response:', data); // new line to log data
    currentSession = data.session;
    loadChat(currentSession);
    hideLoader();
    loadSessions();
    textarea.value="";
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
    const res = await fetch(`${API_BASE}/session/${currentSession.session_id}/query`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    const last = data.session.messages.at(-1);
    hideLoader();
    appendMessage("bot", last.answer);
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
  document.getElementById("loadingOverlay").style.display = "flex";
}

function hideLoader() {
  document.getElementById("loadingOverlay").style.display = "none";
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
  await fetch(`${API_BASE}/update-language`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      device_id: deviceId,
      state: state,
      language: language
    })
  });
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

