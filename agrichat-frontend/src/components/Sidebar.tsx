import clsx from "clsx";
import type { SessionDocument } from "../types/chat";

interface SidebarProps {
  sessions: SessionDocument[];
  currentSessionId?: string;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
}

export function Sidebar({ sessions, currentSessionId, onSelectSession, onNewChat }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar__header">
        <h2>Sessions</h2>
        <button className="primary" onClick={onNewChat} type="button">
          + New Chat
        </button>
      </div>
      <div className="sidebar__list">
        {sessions.length === 0 && <p className="muted">No sessions yet. Start a new conversation!</p>}
        {sessions.map((session) => {
          const lastMessage = session.messages.at(-1);
          const preview = lastMessage?.question ?? "New conversation";
          return (
            <button
              key={session.session_id}
              type="button"
              onClick={() => onSelectSession(session.session_id)}
              className={clsx("sidebar__item", {
                active: currentSessionId === session.session_id,
                archived: session.status === "archived",
              })}
            >
              <div className="sidebar__item-title">{preview}</div>
              <div className="sidebar__item-meta">
                <span>{new Date(session.timestamp).toLocaleString()}</span>
                {session.status === "archived" && <span className="tag">Archived</span>}
              </div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
