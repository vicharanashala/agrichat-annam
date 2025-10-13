import DOMPurify from "dompurify";
import { useMemo } from "react";
import type { SessionMessage } from "../types/chat";

interface ChatMessageProps {
  message: SessionMessage;
  index: number;
}

export function ChatMessage({ message, index }: ChatMessageProps) {
  const safeAnswer = useMemo(() => {
    const html = message.answer || message.final_answer || "";
    return { __html: DOMPurify.sanitize(html) };
  }, [message.answer, message.final_answer]);

  return (
    <div className="chat-message">
      <div className="chat-message__question">
        <span className="label">You</span>
        <p>{message.question}</p>
      </div>
      <div className="chat-message__answer">
        <span className="label">AgriChat</span>
        {message.thinking && (
          <details className="thinking" open>
            <summary>Thinking</summary>
            <pre>{message.thinking}</pre>
          </details>
        )}
        <div className="answer" dangerouslySetInnerHTML={safeAnswer} />
        {(message.sources?.length || message.metadata) && (
          <div className="metadata">
            {message.sources?.length ? (
              <div className="metadata__section">
                <h4>Sources</h4>
                <ul>
                  {message.sources.map((source, sourceIndex) => (
                    <li key={`${index}-source-${sourceIndex}`}>
                      <strong>{source.source}</strong>
                      {source.preview && <p>{source.preview}</p>}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
            {message.metadata && (
              <div className="metadata__section">
                <h4>Metadata</h4>
                <pre>{JSON.stringify(message.metadata, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
