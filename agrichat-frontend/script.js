// Authentication System
class AuthManager {
  constructor() {
    this.user = null;
    this.init();
  }

  init() {
    // Clear any invalid or test user data from localStorage
    this.clearInvalidUserData();

    // Check if user is already logged in
    const savedUser = localStorage.getItem('agrichat_user');
    if (savedUser) {
      this.user = JSON.parse(savedUser);
      this.hideLoginForm();
      // Show welcome message on page load if user is already logged in
      setTimeout(() => this.showAccountDropdown(this.user), 100);

      // setTimeout(() => this.showWelcomeMessage(), 100);
    } else {
      this.showLoginForm();
    }
  }

  clearInvalidUserData() {
    const savedUser = localStorage.getItem('agrichat_user');
    if (savedUser) {
      try {
        const user = JSON.parse(savedUser);
        // Remove any test user data or data without proper authentication structure
        if (!user.username || !user.role || user.full_name === 'Punjab' || user.username === 'test') {
          localStorage.removeItem('agrichat_user');
          console.log('[AUTH] Cleared invalid user data from localStorage');
        }
      } catch (error) {
        localStorage.removeItem('agrichat_user');
        console.log('[AUTH] Cleared corrupted user data from localStorage');
      }
    }
  }

  async login(username, password) {
    try {
      const response = await apiCall(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password })
      });

      const result = await response.json();

      if (result.authenticated) {
        this.user = {
          username: result.username,
          role: result.role,
          full_name: result.full_name
        };
        localStorage.setItem('agrichat_user', JSON.stringify(this.user));
        this.hideLoginForm();
        this.showAccountDropdown(this.user);
        // this.showWelcomeMessage();
        return { success: true };
      } else {
        return { success: false, message: result.message };
      }
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, message: 'Connection error. Please try again.' };
    }
  }

  logout() {
    this.user = null;
    localStorage.removeItem('agrichat_user');

    // Remove user info from header
    const userInfo = document.querySelector('.user-info');
    if (userInfo) {
      userInfo.remove();
    }

    this.showLoginForm();
  }

  showLoginForm() {
    const overlay = document.getElementById('loginOverlay');
    if (overlay) {
      overlay.style.display = 'flex';
    }
  }

  hideLoginForm() {
    const overlay = document.getElementById('loginOverlay');
    if (overlay) {
      overlay.style.display = 'none';
    }
  }

  showAccountDropdown(user) {
    const usernameElem = document.getElementById('usernameDisplay');
    if (user && usernameElem) {
      usernameElem.textContent = `Hello, ${user.full_name}!`;
    }

    // Attach Logout action if not already present
    const logoutBtn = document.getElementById('logoutDropdownBtn');
    if (logoutBtn && typeof authManager !== "undefined" && typeof authManager.logout === "function") {
      logoutBtn.onclick = authManager.logout.bind(authManager);
      // logoutBtn.onclick = authManager.logout;
    }
  }

  // showWelcomeMessage() {
  //   if (this.user) {
  //     // Add user info to header
  //     const header = document.querySelector('.header-left');
  //     if (header) {
  //       // Remove existing user info if it exists
  //       const existingUserInfo = header.querySelector('.user-info');
  //       if (existingUserInfo) {
  //         existingUserInfo.remove();
  //       }

  //       const userInfo = document.createElement('div');
  //       userInfo.className = 'user-info';
  //       userInfo.innerHTML = `
  //         <span style="color: #033220; font-weight: 600; margin-left: 1em;">
  //           Hello, ${this.user.full_name}!
  //         </span>
  //         <button onclick="authManager.logout()" style="margin-left: 1em; padding: 0.3em 0.6em; background: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.8em;">
  //           Logout
  //         </button>
  //       `;
  //       header.appendChild(userInfo);
  //     }
  //   }
  // }

  isAuthenticated() {
    return this.user !== null;
  }

  getUser() {
    return this.user;
  }

  // Utility method to require authentication for actions
  requireAuth(action) {
    if (!this.isAuthenticated()) {
      this.showLoginForm();
      return false;
    }
    return true;
  }
}

// Initialize auth manager
const authManager = new AuthManager();

// Handle login form submission
document.addEventListener('DOMContentLoaded', function () {
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', async function (e) {
      e.preventDefault();

      const username = document.getElementById('loginUsername').value;
      const password = document.getElementById('loginPassword').value;
      const errorDiv = document.getElementById('loginError');

      // Show loading state
      const submitBtn = e.target.querySelector('button[type="submit"]');
      const originalText = submitBtn.textContent;
      submitBtn.textContent = 'Signing in...';
      submitBtn.disabled = true;

      try {
        const result = await authManager.login(username, password);

        if (result.success) {
          errorDiv.style.display = 'none';
        } else {
          errorDiv.textContent = result.message;
          errorDiv.style.display = 'block';
        }
      } catch (error) {
        errorDiv.textContent = 'An error occurred. Please try again.';
        errorDiv.style.display = 'block';
      }

      // Restore button state
      submitBtn.textContent = originalText;
      submitBtn.disabled = false;
    });
  }
});

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

// Enhanced source display formatting
function formatSourceDisplay(source) {
  if (source === 'Golden Database') {
    return 'Golden Database';
  } else if (source === 'RAG Database') {
    return 'RAG Database';
  } else if (source === 'PoPs Database' || source === 'pops') {
    return 'PoPs Database (Package of Practices)';
  } else if (source === 'Fallback LLM') {
    return 'AI Reasoning Engine (gpt-oss pipeline)';
  } else {
    return source;
  }
}

// Markdown processing helper function
function processMarkdown(text) {
  if (!text) return '';

  if (typeof marked !== 'undefined') {
    marked.setOptions({
      breaks: true,
      gfm: true,
      sanitize: false,
      smartLists: true,
      highlight: function (code, lang) {
        if (typeof hljs !== 'undefined' && lang) {
          try {
            return hljs.highlight(code, { language: lang }).value;
          } catch (e) {
            return hljs.highlightAuto(code).value;
          }
        }
        return code;
      }
    });

    try {
      return marked.parse(text);
    } catch (e) {
      console.warn('Markdown parsing failed:', e);
      return text; // Return original text if parsing fails
    }
  }

  return text; // Return original text if marked is not available
}

// Database Toggle State Management
let databaseToggles = {
  rag: true,
  pops: true,
  llm: true
};

// Load database toggle preferences from localStorage
function loadDatabasePreferences() {
  const saved = localStorage.getItem("agrichat_database_preferences");
  if (saved) {
    try {
      databaseToggles = { ...databaseToggles, ...JSON.parse(saved) };
    } catch (e) {
      console.warn("Failed to parse saved database preferences");
    }
  }

  // Update UI toggles
  const ragToggle = document.getElementById("ragToggle");
  const popsToggle = document.getElementById("popsToggle");
  const llmToggle = document.getElementById("llmToggle");

  if (ragToggle) ragToggle.checked = databaseToggles.rag;
  if (popsToggle) popsToggle.checked = databaseToggles.pops;
  if (llmToggle) llmToggle.checked = databaseToggles.llm;

  console.log('[TOGGLES] Loaded database preferences:', databaseToggles);
}



// Save database toggle preferences to localStorage
function saveDatabasePreferences() {
  localStorage.setItem("agrichat_database_preferences", JSON.stringify(databaseToggles));
}

// Get current database selection array for API
function getDatabaseSelection() {
  const selection = [];
  if (databaseToggles.rag) selection.push("rag");
  if (databaseToggles.pops) selection.push("pops");
  if (databaseToggles.llm) selection.push("llm");

  // Ensure at least one database is selected
  if (selection.length === 0) {
    selection.push("llm"); // Fallback to LLM if none selected
    databaseToggles.llm = true;
    document.getElementById("llmToggle").checked = true;
    saveDatabasePreferences();
  }

  return selection;
}

// Initialize database toggle event listeners
function initializeDatabaseToggles() {
  const ragToggle = document.getElementById("ragToggle");
  const popsToggle = document.getElementById("popsToggle");
  const llmToggle = document.getElementById("llmToggle");

  if (ragToggle) {
    ragToggle.addEventListener("change", () => {
      databaseToggles.rag = ragToggle.checked;
      saveDatabasePreferences();
      console.log('[TOGGLES] Golden Database toggled:', ragToggle.checked);
    });
  }

  if (popsToggle) {
    popsToggle.addEventListener("change", () => {
      databaseToggles.pops = popsToggle.checked;
      saveDatabasePreferences();
      console.log('[TOGGLES] PoP Database toggled:', popsToggle.checked);
    });
  }

  if (llmToggle) {
    llmToggle.addEventListener("change", () => {
      databaseToggles.llm = llmToggle.checked;
      saveDatabasePreferences();
      console.log('[TOGGLES] AI Reasoning Engine toggled:', llmToggle.checked);
    });
  }
}


//     saveDatabasePreferences();
//   });

//   llmToggle.addEventListener("change", () => {
//     databaseToggles.llm = llmToggle.checked;
//     saveDatabasePreferences();
//   });
// }

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



function appendMessage(sender, text, index = null, rating = null, thinking = null, source = null, confidence = null) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;

  if (sender === "user") {
    div.innerHTML = ` ${text}`;
  } else {
    let botAnswerHtml = text || '';

    // Process markdown using the helper function
    botAnswerHtml = processMarkdown(botAnswerHtml);

    let youtubeUrl = null;
    try {
      if (researchData && Array.isArray(researchData)) {
        for (const d of researchData) {
          if (d && d.youtube_url) { youtubeUrl = d.youtube_url; break; }
        }
      }
    } catch (e) {
      youtubeUrl = null;
    }

    // Helper to get YouTube thumbnail URL from a YouTube link
    function getYouTubeId(url) {
      if (!url || typeof url !== 'string') return null;
      // patterns: v=ID, /embed/ID, youtu.be/ID, /watch?v=ID
      const idMatch = url.match(/(?:v=|\/embed\/|youtu\.be\/|\/v\/)([A-Za-z0-9_-]{6,})/);
      if (idMatch && idMatch[1]) return idMatch[1];
      // fallback: try to extract last path segment
      try {
        const u = new URL(url);
        if (u.hostname && u.hostname.toLowerCase().includes('youtube')) {
          const params = new URLSearchParams(u.search);
          if (params.has('v')) return params.get('v');
        }
      } catch (e) {
        // ignore
      }
      return null;
    }

    if (youtubeUrl) {
      const vid = getYouTubeId(youtubeUrl);
      const thumbUrl = vid ? `https://img.youtube.com/vi/${vid}/hqdefault.jpg` : null;
      const thumbHtml = thumbUrl ?
        `<div class="bot-source-video" style="margin-top:6px;"><div class="suggested-video-title" style="font-weight:600;margin-bottom:6px;">Suggested Video</div><a href="${youtubeUrl}" target="_blank" rel="noopener noreferrer"><img src="${thumbUrl}" alt="Watch video" style="max-width:360px; width:100%; height:auto; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,0.15);"></a></div>` :
        `<div class="bot-source-video"><div class="suggested-video-title" style="font-weight:600;margin-bottom:4px;">Suggested Video</div><small><a class="inline-video-link" href="${youtubeUrl}" target="_blank" rel="noopener noreferrer">▶ Watch video</a></small></div>`;

      // Prefer to insert under the explicit RAG/PoPs/Golden source label if present
      try {
        const ragRegex = /(RAG Database(?:\s*\([^)]*\))?|PoPs Database|Golden Database|Fallback LLM)/i;
        if (ragRegex.test(botAnswerHtml)) {
          botAnswerHtml = botAnswerHtml.replace(ragRegex, (match) => `${match}\n${thumbHtml}`);
        } else if (botAnswerHtml.indexOf('</small>') !== -1) {
          // insert immediately after the first closing small tag (where Source is often placed)
          botAnswerHtml = botAnswerHtml.replace('</small>', `</small>${thumbHtml}`);
        } else if (botAnswerHtml.toLowerCase().indexOf('source:') !== -1) {
          // As a fallback, append next to the word Source:
          botAnswerHtml = botAnswerHtml.replace(/(source:\s*[^<\n\r]*)/i, `$1${thumbHtml}`);
        } else {
          // final fallback: append at the end inside a small tag
          botAnswerHtml = botAnswerHtml + thumbHtml;
        }
      } catch (e) {
        // If anything goes wrong, append as a small block at the end
        botAnswerHtml = botAnswerHtml + thumbHtml;
      }
    }

    // Show thinking process if available
    const hasThinking = thinking && thinking.trim().length > 0;

    if (hasThinking) {
      const thinkingHtml = `
        <div class="thinking-section" style="margin-bottom: 15px; padding: 12px; background: #243b24; border-left: 4px solid #7e9d85; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
          <div class="thinking-header" style="font-weight: 600; color: #e8f5e8; margin-bottom: 8px; display: flex; align-items: center; font-size: 0.95em;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style="margin-right: 8px;">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" fill="#7e9d85"/>
            </svg>
            Thinking Process
          </div>
          <div class="thinking-content" style="color: #e8f5e8; font-size: 0.9em; line-height: 1.5; font-style: italic;">
            ${thinking.replace(/\n/g, '<br>')}
          </div>
        </div>
      `;

      div.innerHTML = `
        ${thinkingHtml}
        <div class="bot-answer">${botAnswerHtml}</div>
        ${source ? `<div style="margin-top: 8px; font-size: 0.85em; color: #666;">Source: ${formatSourceDisplay(source)}${confidence ? ` | Confidence: ${(confidence * 100).toFixed(0)}%` : ''}</div>` : ''}
        <div class="message-actions">
          <button class="copy-btn" onclick="copyToClipboard(this)" title="Copy">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 448 512">
              <path fill="currentColor" d="M320 448v40c0 13.255-10.745 24-24 24H24c-13.255 0-24-10.745-24-24V120c0-13.255 10.745-24 24-24h72v296c0 30.879 25.121 56 56 56h168zm0-344V0H152c-13.255 0-24 10.745-24 24v368c0 13.255 10.745 24 24 24h272c13.255 0 24-10.745 24-24V128H344c-13.2 0-24-10.8-24zm120.971-31.029L375.029 7.029A24 24 0 0 0 358.059 0H352v96h96v-6.059a24 24 0 0 0-7.029-16.97z"/>
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
    } else {
      // Default rendering when no structured data
      div.innerHTML = `
      <div class="bot-answer">${botAnswerHtml}</div>
      ${source ? `<div style="margin-top: 8px; font-size: 0.85em; color: #666;">Source: ${formatSourceDisplay(source)}${confidence ? ` | Confidence: ${(confidence * 100).toFixed(0)}%` : ''}</div>` : ''}
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
    appendMessage("bot", msg.answer, idx, msg.rating || null, msg.thinking || null, msg.source || null, msg.confidence || null);
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

  // Ensure user info is displayed if logged in
  if (authManager && authManager.isAuthenticated()) {
    authManager.showAccountDropdown(authManager.user);
    // authManager.showWelcomeMessage();
  }

  // Initialize database toggles
  loadDatabasePreferences();
  initializeDatabaseToggles();

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

    // Check authentication
    if (!authManager.isAuthenticated()) {
      authManager.showLoginForm();
      return;
    }

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

    // Handle audio transcription if audio is present
    let finalQuestion = question;
    if (audioBlob) {
      try {
        console.log('Transcribing audio...');
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.webm");
        formData.append("language", lang);

        const transcribeRes = await apiCall(`${API_BASE}/transcribe-audio`, {
          method: "POST",
          body: formData,
        });

        const transcribeData = await transcribeRes.json();
        if (transcribeData.transcript) {
          finalQuestion = transcribeData.transcript;
          console.log('Audio transcribed:', finalQuestion);
        }
      } catch (error) {
        console.error('Audio transcription failed:', error);
        // Continue with original question if transcription fails
      }
    }

    // Send the query to the correct API endpoint
    const requestData = {
      question: finalQuestion,
      device_id: deviceId,
      state: state,
      language: lang,
      database_config: {
        rag_enabled: databaseToggles.rag,
        pops_enabled: databaseToggles.pops,
        llm_enabled: databaseToggles.llm
      }
    };

    const res = await apiCall(`${API_BASE}/query`, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData),
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

    // Check authentication
    if (!authManager.isAuthenticated()) {
      authManager.showLoginForm();
      return;
    }

    const input = document.getElementById("user-input");
    const question = input.value.trim();
    if (!question) return;

    appendMessage("user", question);
    input.value = "";

    // Always use thinking stream for better UX
    await handleThinkingStreamQuery(question);
  });

  async function handleSessionQuery(question) {
    const requestData = {
      question: question,
      device_id: deviceId,
      state: localStorage.getItem("agrichat_user_state") || "",
      database_config: {
        rag_enabled: databaseToggles.rag,
        pops_enabled: databaseToggles.pops,
        llm_enabled: databaseToggles.llm
      }
    };

    showLoader();
    const res = await apiCall(`${API_BASE}/session/${currentSession.session_id}/query`, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData),
    });
    const data = await res.json();
    const last = data.session.messages.at(-1);
    hideLoader();
    appendMessage("bot", last.answer, null, null, last.thinking || null, last.source || null, last.confidence || null);

    currentSession = data.session;

    if (currentSession.recommendations && currentSession.recommendations.length > 0) {
      displayRecommendations(currentSession.recommendations);
    }
  }

  async function handleThinkingStreamQuery(question) {
    const requestData = {
      question: question,
      device_id: deviceId,
      state: localStorage.getItem("agrichat_user_state") || "",
      language: localStorage.getItem("agrichat_user_language") || "English",
      database_config: {
        rag_enabled: databaseToggles.rag,
        pops_enabled: databaseToggles.pops,
        llm_enabled: databaseToggles.llm
      }
    };

    // Create thinking display containers
    let thinkingContainer = null;
    let answerContainer = null;
    let currentThinkingText = '';
    let sessionComplete = false;

    try {
      console.log('[THINKING] Starting thinking stream query:', requestData);

      const response = await apiCall(`${API_BASE}/query/thinking-stream`, {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      console.log('[THINKING] Got response, starting to read stream');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue; // Skip empty data lines

            try {
              // Try to parse the JSON, but be more robust about it
              let data;
              try {
                data = JSON.parse(jsonStr);
              } catch (parseError) {
                // If JSON parsing fails, try to sanitize and parse again
                console.warn('[THINKING] JSON parse failed, attempting to sanitize:', parseError.message);

                // Skip this malformed chunk and continue
                continue;
              }

              console.log('[THINKING] Stream data:', data);

              switch (data.type) {
                case 'thinking_start':
                  console.log('[THINKING] Starting thinking section');
                  // Create thinking section
                  thinkingContainer = createThinkingContainer();
                  if (!thinkingContainer) {
                    console.error('[THINKING] Failed to create thinking container');
                  }
                  break;

                case 'thinking_token':
                  // Update thinking display in real-time with new token
                  if (thinkingContainer) {
                    currentThinkingText = data.text; // This is the cumulative text
                    updateThinkingDisplay(thinkingContainer, currentThinkingText);
                  } else {
                    console.warn('[THINKING] No thinking container for token update');
                  }
                  break;

                case 'thinking_complete':
                  console.log('[THINKING] Thinking complete');
                  // Mark thinking as complete
                  if (thinkingContainer) {
                    finalizeThinkingDisplay(thinkingContainer);
                  }
                  break;

                case 'answer_start':
                  console.log('[THINKING] Starting answer section');
                  // Create answer section
                  answerContainer = createAnswerContainer();
                  if (!answerContainer) {
                    console.error('[THINKING] Failed to create answer container');
                  }
                  break;

                case 'answer':
                  console.log('[THINKING] Displaying final answer');
                  // Display final answer
                  if (answerContainer) {
                    displayFinalAnswer(answerContainer, data.answer, data.source, data.confidence);
                  } else {
                    console.warn('[THINKING] No answer container, falling back to appendMessage');
                    appendMessage("bot", data.answer, null, null, currentThinkingText, data.source, data.confidence);
                  }
                  break;

                case 'metadata_tags':
                  console.log('[THINKING] Displaying metadata tags');
                  if (answerContainer) {
                    displayMetadataTags(answerContainer, data.content);
                  } else {
                    console.warn('[THINKING] No answer container for metadata tags');
                  }
                  break;

                case 'session_complete':
                  // Update session info
                  sessionComplete = true;
                  currentSession = data.session;
                  if (currentSession.recommendations && currentSession.recommendations.length > 0) {
                    displayRecommendations(currentSession.recommendations);
                  }
                  break;

                case 'stream_end':
                  console.log('[THINKING] Stream ended');
                  return; // Exit the stream processing loop

                case 'error':
                  // Ignore errors that come after successful session completion
                  if (sessionComplete) {
                    console.warn('[THINKING] Ignoring error after session completion:', data.message);
                    break;
                  }
                  console.error('Stream error:', data.message);
                  hideLoader();
                  appendMessage("bot", "Sorry, I encountered an error processing your question.", null, null, null, null, null);
                  break;
              }
            } catch (e) {
              console.error('[THINKING] Error parsing stream data:', e, 'Raw data:', jsonStr);
              // Continue processing other lines instead of stopping
            }
          }
        }
      }
    } catch (error) {
      console.error('[THINKING] Stream error:', error);
      hideLoader();
      appendMessage("bot", "Error processing question. Please try again.", null, null, null, null, null);
    } finally {
      hideLoader();
    }
  }

  function createThinkingContainer() {
    const messagesDiv = document.getElementById("chatWindow");
    if (!messagesDiv) {
      console.error("[THINKING] chatWindow element not found!");
      return null;
    }

    const thinkingId = 'thinking-' + Date.now();
    const thinkingDiv = document.createElement("div");
    thinkingDiv.className = "message bot thinking-message";
    thinkingDiv.innerHTML = `
      <div class="thinking-section" id="${thinkingId}" style="margin-bottom: 15px; background: #4f7c5b; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        
        <!-- Clickable Header for Expand/Collapse -->
        <div class="thinking-header" onclick="toggleThinking('${thinkingId}')" style="
          padding: 12px 16px; 
          background: linear-gradient(135deg, #3a5a41, #4f7c5b); 
          color: var(--white); 
          font-weight: 600; 
          display: flex; 
          align-items: center; 
          justify-content: space-between;
          cursor: pointer;
          user-select: none;
          transition: all 0.3s ease;
          border-bottom: 1px solid rgba(255,255,255,0.1);
        ">
          <div style="display: flex; align-items: center;">
            <div class="thinking-spinner" style="
              margin-right: 10px; 
              width: 18px; 
              height: 18px; 
              border: 2px solid rgba(255,255,255,0.3); 
              border-top: 2px solid var(--white); 
              border-radius: 50%; 
              animation: spin 1s linear infinite;
            "></div>
            <span class="thinking-status">Thinking...</span>
          </div>
          <div class="thinking-toggle" style="
            font-size: 1.2em; 
            transition: transform 0.3s ease;
            color: rgba(255,255,255,0.8);
          ">▼</div>
        </div>
        
        <!-- Expandable Content -->
        <div class="thinking-content-wrapper" style="
          max-height: 300px;
          overflow-y: auto;
          overflow-x: hidden;
          transition: max-height 0.4s ease, padding 0.4s ease;
          background: rgba(255,255,255,0.05);
          scrollbar-width: thin;
          scrollbar-color: rgba(255,255,255,0.3) transparent;
        ">
          <div class="thinking-content" style="
            padding: 16px; 
            color: var(--white); 
            font-size: 0.9em; 
            line-height: 1.6; 
            font-family: 'Courier New', monospace;
            min-height: 30px;
            word-wrap: break-word;
            background: rgba(0,0,0,0.1);
            border-radius: 4px;
            margin: 12px;
          ">
            <span class="thinking-cursor" style="
              animation: blink 1s infinite;
              color: #7dd87d;
              font-weight: bold;
            ">▋</span>
          </div>
        </div>
      </div>
    `;

    messagesDiv.appendChild(thinkingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    // Store reference to the thinking container for easy access
    thinkingDiv.dataset.thinkingId = thinkingId;

    return {
      container: thinkingDiv.querySelector('.thinking-content'),
      wrapper: thinkingDiv.querySelector('.thinking-content-wrapper'),
      header: thinkingDiv.querySelector('.thinking-header'),
      section: thinkingDiv.querySelector('.thinking-section')
    };
  }

  function updateThinkingDisplay(containerObj, text) {
    if (containerObj && containerObj.container) {
      // Format the text with proper styling
      const formattedText = text.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

      // Update with typing cursor
      containerObj.container.innerHTML = `
        <div style="word-wrap: break-word; white-space: pre-wrap;">
          ${formattedText}
          <span class="thinking-cursor" style="
            animation: blink 1s infinite;
            color: #7dd87d;
            font-weight: bold;
            margin-left: 2px;
          ">▋</span>
        </div>
      `;

      // Auto-scroll to bottom
      const chatWindow = document.getElementById("chatWindow");
      if (chatWindow) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
      }

      // Auto-expand if collapsed during streaming
      if (containerObj.wrapper && containerObj.wrapper.style.maxHeight === '0px') {
        containerObj.wrapper.style.maxHeight = '300px';
        const toggle = containerObj.section.querySelector('.thinking-toggle');
        if (toggle) {
          toggle.style.transform = 'rotate(0deg)';
          toggle.textContent = '▼';
        }
      }
    }
  }

  function finalizeThinkingDisplay(containerObj) {
    if (containerObj && containerObj.container) {
      // Remove the blinking cursor
      const cursor = containerObj.container.querySelector('.thinking-cursor');
      if (cursor) cursor.remove();

      // Update header to show completion
      const header = containerObj.section.querySelector('.thinking-header');
      const statusSpan = header.querySelector('.thinking-status');
      const spinner = header.querySelector('.thinking-spinner');

      if (statusSpan) {
        statusSpan.innerHTML = 'Thinking Complete';
      }

      if (spinner) {
        spinner.style.animation = 'none';
        spinner.innerHTML = '✓';
        spinner.style.border = 'none';
        spinner.style.background = '#28a745';
        spinner.style.color = 'white';
        spinner.style.display = 'flex';
        spinner.style.alignItems = 'center';
        spinner.style.justifyContent = 'center';
        spinner.style.fontSize = '12px';
        spinner.style.fontWeight = 'bold';
      }

      // Update header background to indicate completion
      header.style.background = 'linear-gradient(135deg, #28a745, #34ce57)';

      // Add subtle completion animation
      containerObj.section.style.animation = 'completePulse 0.6s ease-out';

      // Auto-collapse thinking section after completion to save space
      setTimeout(() => {
        const contentWrapper = containerObj.section.querySelector('.thinking-content-wrapper');
        const toggleIcon = header.querySelector('.thinking-toggle');
        if (contentWrapper && toggleIcon && containerObj.section) {
          // Smooth transition for collapse
          contentWrapper.style.transition = 'max-height 0.3s ease-out, padding 0.3s ease-out';
          contentWrapper.style.maxHeight = '0px';
          contentWrapper.style.paddingTop = '0px';
          contentWrapper.style.paddingBottom = '0px';
          toggleIcon.innerHTML = '▶';
          containerObj.section.dataset.expanded = 'false';

          console.log('[THINKING] Auto-collapsed thinking section');
        }
      }, 2000); // Wait 2 seconds before auto-collapse for better UX
    }
  }

  function createAnswerContainer() {
    const messagesDiv = document.getElementById("chatWindow");
    if (!messagesDiv) {
      console.error("[THINKING] chatWindow element not found for answer container!");
      return null;
    }
    const answerDiv = document.createElement("div");
    answerDiv.className = "message bot-message";
    const answerId = 'answer-' + Date.now();
    answerDiv.innerHTML = `
      <div class="answer-section" id="${answerId}" style="margin-bottom: 15px; background: #2d5a35; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        
        <!-- Clickable Header for Expand/Collapse -->
        <div class="answer-header" onclick="toggleAnswer('${answerId}')" style="
          padding: 12px 16px; 
          background: linear-gradient(135deg, #2d5a35, #3a7043); 
          color: var(--white); 
          font-weight: 600; 
          display: flex; 
          align-items: center; 
          justify-content: space-between;
          cursor: pointer;
          user-select: none;
          transition: all 0.3s ease;
          border-bottom: 1px solid rgba(255,255,255,0.1);
        ">
          <div style="display: flex; align-items: center;">
            <div class="answer-spinner" style="
              margin-right: 10px; 
              width: 18px; 
              height: 18px; 
              border: 2px solid rgba(255,255,255,0.3); 
              border-top: 2px solid var(--white); 
              border-radius: 50%; 
              animation: spin 1s linear infinite;
            "></div>
            <span class="answer-status">Generating Answer...</span>
          </div>
          <div class="answer-toggle" style="
            font-size: 1.2em; 
            transition: transform 0.3s ease;
            color: rgba(255,255,255,0.8);
          ">▼</div>
        </div>
        
        <!-- Expandable Content -->
        <div class="answer-content-wrapper" style="
          max-height: none;
          overflow: visible;
          transition: max-height 0.4s ease, padding 0.4s ease, overflow 0.4s ease;
          background: rgba(255,255,255,0.05);
        ">
          <div class="answer-content" style="
            padding: 16px; 
            color: var(--white); 
            font-size: 0.95em; 
            line-height: 1.6; 
            min-height: 30px;
            background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.08));
          ">
            <!-- Answer will be populated here -->
          </div>
        </div>
      </div>
    `;
    messagesDiv.appendChild(answerDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return answerDiv.querySelector('.answer-content');
  }

  function displayFinalAnswer(container, answer, source, confidence) {
    if (container) {
      const answerSection = container.closest('.answer-section');
      const header = answerSection.querySelector('.answer-header');

      // Update header to show completion (similar to thinking completion)
      const statusSpan = header.querySelector('.answer-status');
      const spinner = header.querySelector('.answer-spinner');

      if (statusSpan) {
        statusSpan.innerHTML = 'Answer';
      }

      if (spinner) {
        spinner.style.animation = 'none';
        spinner.innerHTML = '✓';
        spinner.style.border = 'none';
        spinner.style.background = '#28a745';
        spinner.style.color = 'white';
        spinner.style.display = 'flex';
        spinner.style.alignItems = 'center';
        spinner.style.justifyContent = 'center';
        spinner.style.fontSize = '12px';
        spinner.style.fontWeight = 'bold';
      }

      // Update header background to indicate completion
      header.style.background = 'linear-gradient(135deg, #28a745, #34ce57)';

      container.innerHTML = `
        <div class="bot-answer" style="
          color: var(--white); 
          font-size: 0.95em; 
          line-height: 1.6;
          margin-bottom: 12px;
        ">${processMarkdown(answer)}</div>
        ${source ? `<div class="message-source" style="
          margin-top: 12px; 
          padding: 8px 12px; 
          background: rgba(255,255,255,0.1); 
          border-radius: 4px; 
          font-size: 0.8em; 
          color: rgba(255,255,255,0.8);
          border-left: 3px solid rgba(255,255,255,0.3);
        ">Source: ${formatSourceDisplay(source)}</div>` : ''}
        <div class="message-actions" style="
          margin-top: 12px; 
          padding-top: 8px; 
          border-top: 1px solid rgba(255,255,255,0.1); 
          display: flex; 
          gap: 8px;
        ">
          <button class="action-btn copy-btn" onclick="copyToClipboard(this)" title="Copy answer" style="
            background: rgba(255,255,255,0.1); 
            border: 1px solid rgba(255,255,255,0.2); 
            color: var(--white); 
            padding: 6px 8px; 
            border-radius: 4px; 
            cursor: pointer; 
            display: flex; 
            align-items: center;
            transition: all 0.2s ease;
          " onmouseover="this.style.background='rgba(255,255,255,0.2)'" onmouseout="this.style.background='rgba(255,255,255,0.1)'">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
            </svg>
          </button>
          <button class="action-btn rate-btn" onclick="rateMessage(this, 1)" title="Good answer" style="
            background: rgba(255,255,255,0.1); 
            border: 1px solid rgba(255,255,255,0.2); 
            color: var(--white); 
            padding: 6px 8px; 
            border-radius: 4px; 
            cursor: pointer;
            transition: all 0.2s ease;
          " onmouseover="this.style.background='rgba(40,167,69,0.3)'" onmouseout="this.style.background='rgba(255,255,255,0.1)'">👍</button>
          <button class="action-btn rate-btn" onclick="rateMessage(this, -1)" title="Bad answer" style="
            background: rgba(255,255,255,0.1); 
            border: 1px solid rgba(255,255,255,0.2); 
            color: var(--white); 
            padding: 6px 8px; 
            border-radius: 4px; 
            cursor: pointer;
            transition: all 0.2s ease;
          " onmouseover="this.style.background='rgba(220,53,69,0.3)'" onmouseout="this.style.background='rgba(255,255,255,0.1)'">👎</button>
        </div>
      `;

      container.closest('#messages').scrollTop = container.closest('#messages').scrollHeight;
    }
  }

  function displayMetadataTags(container, metadataContent) {
    if (container && metadataContent) {
      // Check if metadata tags already exist to avoid duplicates
      const existingMetadata = container.querySelector('.metadata-tags');
      if (existingMetadata) {
        existingMetadata.remove();
      }

      // Create metadata tags section
      const metadataDiv = document.createElement('div');
      metadataDiv.className = 'metadata-tags';
      metadataDiv.style.cssText = `
        margin-top: 15px;
        padding: 12px;
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
        border-left: 3px solid rgba(255,255,255,0.3);
        font-size: 0.85em;
        color: rgba(255,255,255,0.9);
        line-height: 1.5;
      `;

      // Add a header for the metadata section
      const metadataHeader = document.createElement('div');
      metadataHeader.style.cssText = `
        font-weight: 600;
        margin-bottom: 8px;
        color: rgba(255,255,255,0.95);
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 6px;
      `;
      metadataHeader.textContent = '📊 Document Metadata';

      const metadataBody = document.createElement('div');
      metadataBody.innerHTML = processMarkdown(metadataContent);

      metadataDiv.appendChild(metadataHeader);
      metadataDiv.appendChild(metadataBody);

      // Append to the container
      container.appendChild(metadataDiv);

      // Scroll to show the new content
      container.closest('#messages').scrollTop = container.closest('#messages').scrollHeight;
    }
  }

  function formatSourceDisplay(source) {
    if (!source) return 'Unknown';

    // Enhanced source display mapping
    const sourceMap = {
      'Fallback LLM': 'AI Reasoning Engine (gpt-oss pipeline)',
      'RAG': 'Golden Database',
      'PoPs': 'Package of Practices (PoPs)',
      'rag': 'Golden Database',
      'pops': 'Package of Practices (PoPs)',
      'llm': 'AI Reasoning Engine (gpt-oss pipeline)',
      'fallback': 'AI Reasoning Engine (gpt-oss pipeline)',
      'RAG Database (Golden)': 'Golden Database',
      'RAG Database': 'RAG Database',
      'Golden Database': 'Golden Database'
    };

    const lowerSource = source.toLowerCase();
    for (const [key, value] of Object.entries(sourceMap)) {
      if (lowerSource.includes(key.toLowerCase())) {
        return value;
      }
    }

    return source;
  }

  // Global function for thinking toggle (accessible from onclick)
  window.toggleThinking = function (thinkingId) {
    const section = document.getElementById(thinkingId);
    if (!section) return;

    const wrapper = section.querySelector('.thinking-content-wrapper');
    const toggle = section.querySelector('.thinking-toggle');
    const header = section.querySelector('.thinking-header');

    if (!wrapper || !toggle) return;

    const isCollapsed = wrapper.style.maxHeight === '0px';

    if (isCollapsed) {
      // Expand
      wrapper.style.maxHeight = '300px';
      wrapper.style.padding = '';
      toggle.style.transform = 'rotate(0deg)';
      toggle.textContent = '▼';
      header.style.background = header.style.background.replace('rgba(0,0,0,0.1)', 'rgba(0,0,0,0.05)');
    } else {
      // Collapse
      wrapper.style.maxHeight = '0px';
      wrapper.style.padding = '0 12px';
      toggle.style.transform = 'rotate(-90deg)';
      toggle.textContent = '▶';
      header.style.background = header.style.background.replace('rgba(0,0,0,0.05)', 'rgba(0,0,0,0.1)');
    }

    // Add visual feedback
    header.style.transform = 'scale(0.98)';
    setTimeout(() => {
      header.style.transform = 'scale(1)';
    }, 150);
  }

  // Global function for answer toggle (accessible from onclick)
  window.toggleAnswer = function (answerId) {
    const section = document.getElementById(answerId);
    if (!section) return;

    const wrapper = section.querySelector('.answer-content-wrapper');
    const toggle = section.querySelector('.answer-toggle');
    const header = section.querySelector('.answer-header');

    if (!wrapper || !toggle) return;

    const isCollapsed = wrapper.style.maxHeight === '100px';

    if (isCollapsed) {
      // Expand
      wrapper.style.maxHeight = 'none';
      wrapper.style.overflow = 'visible';
      toggle.style.transform = 'rotate(0deg)';
      toggle.textContent = '▼';
    } else {
      // Collapse
      wrapper.style.maxHeight = '100px';
      wrapper.style.overflow = 'hidden';
      toggle.style.transform = 'rotate(-90deg)';
      toggle.textContent = '▶';
    }

    // Add visual feedback
    header.style.transform = 'scale(0.98)';
    setTimeout(() => {
      header.style.transform = 'scale(1)';
    }, 150);
  }

  // ...existing code... (streaming functionality removed by user request)

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
      color: var(--white);
      background-color: #3f6445;
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

// function updateLocationUI(state, language) {
//   document.getElementById("locationState").textContent = state;
//   document.getElementById("locationLanguage").textContent = language;

//   const manualStateSelect = document.getElementById("manualStateSelect");
//   // const manualLangSelect = document.getElementById("manualLanguageSelect");

//   if (manualStateSelect) manualStateSelect.value = state;
//   // if (manualLangSelect) manualLangSelect.value = language;
// }

function updateLocationUI(state, language) {
  const stateElem = document.getElementById("locationState");
  const langElem = document.getElementById("locationLanguage");
  const manualStateSelect = document.getElementById("manualStateSelect");

  if (stateElem) stateElem.textContent = state;
  if (langElem) langElem.textContent = language;
  if (manualStateSelect) manualStateSelect.value = state;
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
  const selectedLang = localStorage.getItem("agrichat_user_language") || defaultLang;

  // const selectedLang = document.getElementById("manualLanguageSelect").value || defaultLang;

  localStorage.setItem("agrichat_user_state", selectedState);
  localStorage.setItem("agrichat_user_language", selectedLang);

  updateLocationUI(selectedState, selectedLang);
  await updateLanguageInBackend(selectedState, selectedLang);
});

// document.getElementById("manualLanguageSelect").addEventListener("change", async (e) => {
//   const selectedLang = e.target.value;
//   const selectedState = document.getElementById("manualStateSelect").value;

//   localStorage.setItem("agrichat_user_language", selectedLang);
//   if (selectedState) localStorage.setItem("agrichat_user_state", selectedState);

//   updateLocationUI(selectedState, selectedLang);
//   if (selectedState) await updateLanguageInBackend(selectedState, selectedLang);
// });

const languageSelect = document.getElementById("manualLanguageSelect");
if (languageSelect) {
  languageSelect.addEventListener("change", async (e) => {
    const selectedLang = e.target.value;
    const selectedState = document.getElementById("manualStateSelect").value;

    localStorage.setItem("agrichat_user_language", selectedLang);
    if (selectedState) localStorage.setItem("agrichat_user_state", selectedState);

    updateLocationUI(selectedState, selectedLang);
    if (selectedState) await updateLanguageInBackend(selectedState, selectedLang);
  });
}

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
