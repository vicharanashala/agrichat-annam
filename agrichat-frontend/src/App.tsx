import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
import { Sidebar } from "./components/Sidebar";
import { ChatMessage } from "./components/ChatMessage";
import { StreamingMessage } from "./components/StreamingMessage";
import { Composer } from "./components/Composer";
import { useLocalStorage } from "./hooks/useLocalStorage";
import { ensureDeviceId } from "./utils/device";
import { continueSession, fetchSession, fetchSessions, streamThinking } from "./services/api";
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
  const [deviceId] = useState(() => ensureDeviceId());
  const [sessions, setSessions] = useState<SessionDocument[]>([]);
  const [currentSession, setCurrentSession] = useState<SessionDocument | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [streamDraft, setStreamDraft] = useState<StreamingDraft | null>(null);

  const [storedState, setStoredState] = useLocalStorage<string>("agrichat_user_state", DEFAULT_STATE);
  const [storedLanguage, setStoredLanguage] = useLocalStorage<string>("agrichat_user_language", DEFAULT_LANGUAGE);
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

  const handleNewChat = useCallback(() => {
    setError(null);
    setCurrentSession(null);
    setSelectedSessionId(null);
    setStreamDraft(null);
    setQuestion("");
  }, []);

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

  const handleSubmit = useCallback(async () => {
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

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        currentSessionId={selectedSessionId ?? undefined}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />
      <main className="main">
        <header className="main__header">
          <div>
            <h1>AgriChat Assistant</h1>
            <p className="muted">Conversational agronomy support for Indian farmers</p>
          </div>
          <div className="status">
            <span className="tag">Device: {deviceId.slice(0, 8)}</span>
            {isSending && <span className="tag processing">Thinking...</span>}
          </div>
        </header>
        {error && <div className="error-banner">{error}</div>}
        <section className="chat-window">
          {messages.map((message, index) => (
            <ChatMessage key={`${message.question}-${index}`} message={message} index={index} />
          ))}
          {streamDraft && <StreamingMessage question={streamDraft.question} thinking={streamDraft.thinking} answer={streamDraft.answer} />}
        </section>
        {currentSession?.recommendations && currentSession.recommendations.length > 0 && (
          <section className="recommendations">
            <h3>Recommendations</h3>
            <ul>
              {currentSession.recommendations.map((item, index) => (
                <li key={`${item.title}-${index}`}>
                  <strong>{item.title}</strong>
                  {item.description && <p>{item.description}</p>}
                  {item.link && (
                    <a href={item.link} target="_blank" rel="noreferrer">
                      Learn more
                    </a>
                  )}
                </li>
              ))}
            </ul>
          </section>
        )}
        <Composer
          question={question}
          onQuestionChange={setQuestion}
          onSubmit={handleSubmit}
          disabled={isSending}
          stateValue={storedState}
          onStateChange={setStoredState}
          language={storedLanguage}
          onLanguageChange={setStoredLanguage}
          toggles={toggles}
          onToggleChange={handleToggleChange}
        />
      </main>
    </div>
  );
}

export default App;
