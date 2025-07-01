// const API_BASE = "https://agrichat-annam.onrender.com/api";
const API_BASE = "http://localhost:8000/api";

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

function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  div.innerHTML = `<strong><i class="fas fa-${sender === "user" ? "user" : "robot"}"></i> ${sender === "user" ? "You" : "AgriChat"}:</strong> ${text}`;
  document.getElementById("chatWindow").appendChild(div);
}

function loadChat(session) {
  document.getElementById("startScreen").style.display = "none";
  document.getElementById("chatWindow").style.display = "block";
  document.getElementById("chatWindow").innerHTML = "";
  document.getElementById("exportBtn").style.display = "inline-block";

  if (session.status === "archived") {
    document.getElementById("archivedNotice").style.display = "block";
    document.getElementById("chat-form").style.display = "none";
  } else {
    document.getElementById("archivedNotice").style.display = "none";
    document.getElementById("chat-form").style.display = "flex";
  }

  session.messages.forEach((msg) => {
    appendMessage("user", msg.question);
    appendMessage("bot", msg.answer);
  });

  document.getElementById("chatWindow").scrollTop = document.getElementById("chatWindow").scrollHeight;
}

window.addEventListener("DOMContentLoaded", () => {
  loadSessions();

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
    const textarea = e.target.querySelector("textarea");
    const question = textarea.value.trim();
    if (!question) return;

    const formData = new FormData();
    formData.append("question", question);
    formData.append("device_id", deviceId);
    showLoader();
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
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