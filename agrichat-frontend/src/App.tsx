import { useCallback, useEffect, useMemo, useState } from "react";
import { Send, Settings, Plus, MessageSquare, Sun, Moon, Mic } from "lucide-react";
import { Button } from "./components/ui/Button";
import { Textarea } from "./components/ui/Textarea";
import { ChatMessage } from "./components/ChatMessage";
import { StreamingMessage } from "./components/StreamingMessage";
import { getCurrentLocation } from "./utils/geolocation";
import { useLocalStorage } from "./hooks/useLocalStorage";
import { ensureDeviceId } from "./utils/device";
import { continueSession, fetchSession, fetchSessions, streamThinking, transcribeAudio } from "./services/api";
import { cn } from "./utils/cn";
import type {
  DatabaseToggleConfig,
  SessionDocument,
  SessionMessage,
  ThinkingStreamEvent,
} from "./types/chat";
import { DEFAULT_LANGUAGE, DEFAULT_STATE } from "./config";

const DEFAULT_TOGGLES: DatabaseToggleConfig = {
  golden_enabled: true,
  pops_enabled: true,
  llm_enabled: true,
  similarity_threshold: 0.7,
  pops_similarity_threshold: 0.35,
  enable_adaptive_thresholds: true,
  strict_validation: false,
  show_database_path: true,
  show_confidence_scores: true,
};

interface StreamingDraft {
  question: string;
  thinking?: string;
  answer?: string;
}

function mergeSessionLists(newSession: SessionDocument, sessions: SessionDocument[]): SessionDocument[] {
  const filtered = sessions.filter((item) => item.session_id !== newSession.session_id);
  return [newSession, ...filtered].slice(0, 50);
}

function App() {
  // Core state
  const [deviceId] = useState(() => ensureDeviceId());
  const [sessions, setSessions] = useState<SessionDocument[]>([]);
  const [currentSession, setCurrentSession] = useState<SessionDocument | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [streamDraft, setStreamDraft] = useState<StreamingDraft | null>(null);

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(true); 
  const [showSettings, setShowSettings] = useState(false);
  const [isDarkMode, setIsDarkMode] = useLocalStorage<boolean>("agrichat_dark_mode", false);

  // Settings
  const [storedState, setStoredState] = useLocalStorage<string>("agrichat_user_state", DEFAULT_STATE);
  const [storedLanguage] = useLocalStorage<string>("agrichat_user_language", DEFAULT_LANGUAGE);

  // Auto-detect location on component mount
  useEffect(() => {
    const autoDetectLocation = async () => {
      try {
        const location = await getCurrentLocation();
        if (location.state && location.state !== "Unknown") {
          setStoredState(location.state);
        }
      } catch (error) {
        console.log("Location detection failed:", error);
        // Silently fail - user can still select manually
      }
    };

    // Only auto-detect if no state is already stored
    if (!storedState || storedState === DEFAULT_STATE) {
      autoDetectLocation();
    }
  }, [storedState, setStoredState]);
  const [storedToggles, setStoredToggles] = useLocalStorage<DatabaseToggleConfig>(
    "agrichat_database_config",
    DEFAULT_TOGGLES,
  );

  const toggles = useMemo<DatabaseToggleConfig>(
    () => ({
      ...DEFAULT_TOGGLES,
      ...storedToggles,
    }),
    [storedToggles],
  );

  // Dark mode effect
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const handleToggleChange = useCallback(
    (partial: Partial<DatabaseToggleConfig>) => {
      setStoredToggles({
        ...toggles,
        ...partial,
      });
    },
    [setStoredToggles, toggles],
  );

  const loadSessions = useCallback(async () => {
    try {
      const data = await fetchSessions(deviceId);
      setSessions(data.sessions);
    } catch (err) {
      console.error(err);
      setError((err as Error).message ?? "Failed to fetch sessions");
    }
  }, [deviceId]);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  const handleSelectSession = useCallback(
    async (sessionId: string) => {
      setError(null);
      setStreamDraft(null);
      setIsSending(false);
      setSelectedSessionId(sessionId);
      setSidebarOpen(false); // Close sidebar on mobile after selection
      try {
        const response = await fetchSession(sessionId, deviceId);
        setCurrentSession(response.session);
      } catch (err) {
        console.error(err);
        setError((err as Error).message ?? "Failed to load session");
      }
    },
    [deviceId],
  );

  const buildPayload = useCallback(
    (userQuestion: string) => ({
      question: userQuestion.trim(),
      device_id: deviceId,
      state: storedState ?? DEFAULT_STATE,
      language: storedLanguage ?? DEFAULT_LANGUAGE,
      database_config: toggles,
    }),
    [deviceId, storedLanguage, storedState, toggles],
  );

  const handleStreamEvent = useCallback(
    (event: ThinkingStreamEvent) => {
      switch (event.type) {
        case "thinking_start":
          setStreamDraft((prev) => (prev ? { ...prev, thinking: "" } : prev));
          break;
        case "thinking_token":
          setStreamDraft((prev) => (prev ? { ...prev, thinking: event.text } : prev));
          break;
        case "thinking_complete":
          setStreamDraft((prev) => (prev ? { ...prev, thinking: event.thinking ?? prev.thinking } : prev));
          break;
        case "answer":
          setStreamDraft((prev) => (prev ? { ...prev, answer: event.answer } : prev));
          break;
        case "session_complete":
          setCurrentSession(event.session);
          setSelectedSessionId(event.session.session_id);
          setSessions((prev) => mergeSessionLists(event.session, prev));
          setStreamDraft(null);
          setQuestion("");
          setIsSending(false);
          break;
        case "error":
          setError(event.message);
          setIsSending(false);
          break;
        case "stream_end":
        default:
          break;
      }
    },
    [],
  );

  const handleSend = useCallback(async () => {
    if (!question.trim() || isSending) {
      return;
    }

    setError(null);
    const payload = buildPayload(question);

    if (!selectedSessionId || !currentSession) {
      // Start new session with streaming
      setIsSending(true);
      setStreamDraft({ question: payload.question });
      try {
        await streamThinking(payload, {
          onEvent: handleStreamEvent,
          onError: (err) => {
            console.error(err);
            setError(err.message ?? "Streaming failed");
            setIsSending(false);
          },
        });
      } catch (err) {
        console.error(err);
        setError((err as Error).message ?? "Streaming failed");
        setIsSending(false);
      }
      return;
    }

    // Continue existing session
    setIsSending(true);
    try {
      const response = await continueSession(selectedSessionId, payload);
      setCurrentSession(response.session);
      setSessions((prev) => mergeSessionLists(response.session, prev));
      setQuestion("");
    } catch (err) {
      console.error(err);
      setError((err as Error).message ?? "Failed to send message");
    } finally {
      setIsSending(false);
    }
  }, [buildPayload, currentSession, handleStreamEvent, isSending, question, selectedSessionId]);

  useEffect(() => {
    if (!selectedSessionId && sessions.length > 0) {
      setSelectedSessionId(sessions[0].session_id);
      setCurrentSession(sessions[0]);
    }
  }, [selectedSessionId, sessions]);

  const messages: SessionMessage[] = currentSession?.messages ?? [];

  const startNewChat = () => {
    setCurrentSession(null);
    setSelectedSessionId(null);
    setQuestion("");
    setError(null);
    setStreamDraft(null);
    setSidebarOpen(false);
  };

  const handleFileUpload = async (file: File) => {
    try {
      setIsSending(true);
      const result = await transcribeAudio(file, storedLanguage);
      setQuestion(result.transcript);
    } catch (err) {
      setError((err as Error).message ?? "Failed to transcribe audio");
    } finally {
      setIsSending(false);
    }
  };



  return (
    <div className="flex h-screen bg-white text-gray-900 dark:bg-gray-800 dark:text-gray-100">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/20 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <div className="w-80 bg-gray-900 text-white flex-shrink-0 flex flex-col"
        style={{ minHeight: '100vh' }}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-white">AgriChat</h2>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsDarkMode(!isDarkMode)}
                className="text-gray-400 hover:text-white hover:bg-gray-700"
              >
                {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSidebarOpen(false)}
                className="lg:hidden"
              >
                ×
              </Button>
            </div>
          </div>

          {/* New Chat Button */}
          <div className="p-4">
            <Button 
              onClick={startNewChat}
              className="w-full justify-start gap-3 bg-transparent border-gray-600 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
              variant="outline"
            >
              <Plus className="h-4 w-4" />
              New Chat
            </Button>
          </div>

          {/* Sessions List */}
          <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
            {sessions.map((session) => (
              <button
                key={session.session_id}
                onClick={() => handleSelectSession(session.session_id)}
                className={cn(
                  "w-full text-left p-3 rounded-lg transition-colors group",
                  "hover:bg-gray-700",
                  selectedSessionId === session.session_id 
                    ? "bg-gray-700 text-white" 
                    : "text-gray-300 hover:text-white"
                )}
              >
                <div className="flex items-start gap-2">
                  <MessageSquare className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">
                      {session.messages[0]?.question || "New Chat"}
                    </p>
                    <p className="text-xs text-gray-500 group-hover:text-gray-400">
                      {new Date(session.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header */}
        <header className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden"
            >
              <MessageSquare className="h-5 w-5" />
            </Button>
            <div>
              <h1 className="text-xl font-semibold text-gray-800 dark:text-white">AgriChat Assistant</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Agricultural guidance powered by AI
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowSettings(!showSettings)}
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </header>

        {/* Toggle Controls */}
        <div className="p-4 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-center gap-8 max-w-4xl mx-auto">
            {/* Golden Enabled Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Golden</span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={toggles.golden_enabled}
                  onChange={(e) => handleToggleChange({ golden_enabled: e.target.checked })}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>

            {/* POPs Enabled Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">POPs</span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={toggles.pops_enabled}
                  onChange={(e) => handleToggleChange({ pops_enabled: e.target.checked })}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>

            {/* LLM Enabled Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">LLM</span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={toggles.llm_enabled}
                  onChange={(e) => handleToggleChange({ llm_enabled: e.target.checked })}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="mx-4 mt-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-sm">
            {error}
          </div>
        )}

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {!currentSession && !streamDraft && (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-6 px-4">
              <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center">
                <MessageSquare className="h-10 w-10 text-green-600" />
              </div>
              <div>
                <h2 className="text-3xl font-semibold mb-3 text-gray-800 dark:text-white">Welcome to AgriChat</h2>
                <p className="text-gray-600 dark:text-gray-300 max-w-lg leading-relaxed">
                  Your AI-powered agricultural assistant. Ask me anything about farming, crops, 
                  soil management, pest control, fertilizers, and agricultural best practices.
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg">
                {[
                  "What fertilizer is best for rice?",
                  "How to control cotton bollworms?", 
                  "When to plant wheat in Punjab?",
                  "Soil testing recommendations"
                ].map((suggestion) => (
                  <Button
                    key={suggestion}
                    variant="outline"
                    className="text-sm h-auto p-3 whitespace-normal"
                    onClick={() => setQuestion(suggestion)}
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <ChatMessage key={`${message.question}-${index}`} message={message} index={index} />
          ))}
          
          {streamDraft && (
            <StreamingMessage 
              question={streamDraft.question} 
              thinking={streamDraft.thinking} 
              answer={streamDraft.answer} 
            />
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="space-y-4">
            {/* Settings Panel */}
            {showSettings && (
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 space-y-4">

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">State</label>
                    <select 
                      value={storedState} 
                      onChange={(e) => setStoredState(e.target.value)}
                      className="w-full p-2 rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900"
                    >
                      <option value="Tamil Nadu">Tamil Nadu</option>
                      <option value="Punjab">Punjab</option>
                      <option value="Maharashtra">Maharashtra</option>
                      <option value="Andhra Pradesh">Andhra Pradesh</option>
                      <option value="Haryana">Haryana</option>
                    </select>
                  </div>
                </div>
                <div className="flex flex-wrap gap-4 text-sm">
                  {Object.entries(toggles).map(([key, value]) => (
                    typeof value === 'boolean' && (
                      <label key={key} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={value}
                          onChange={(e) => handleToggleChange({ [key]: e.target.checked })}
                          className="rounded"
                        />
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </label>
                    )
                  ))}
                </div>
              </div>
            )}

            {/* Input Row */}
            <div className="flex items-end gap-2">
              <div className="flex-1 relative">
                <Textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask about crops, soil, pests, fertilizers..."
                  className="min-h-[60px] max-h-32 pr-12 resize-none"
                  disabled={isSending}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                />
                {isSending && (
                  <div className="absolute right-3 bottom-3 text-gray-500 dark:text-gray-400">
                    <div className="typing-indicator">
                      <div className="typing-dot" style={{ animationDelay: '0ms' }}></div>
                      <div className="typing-dot" style={{ animationDelay: '150ms' }}></div>
                      <div className="typing-dot" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                )}
              </div>
              
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                id="audio-upload"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileUpload(file);
                }}
              />
              
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => document.getElementById('audio-upload')?.click()}
                disabled={isSending}
                className="flex-shrink-0"
              >
                <Mic className="h-4 w-4" />
              </Button>
              
              <Button 
                type="submit" 
                disabled={isSending || !question.trim()}
                className="flex-shrink-0"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
