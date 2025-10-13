interface StreamingMessageProps {
  question: string;
  thinking?: string;
  answer?: string;
}

export function StreamingMessage({ question, thinking, answer }: StreamingMessageProps) {
  return (
    <div className="chat-message streaming">
      <div className="chat-message__question">
        <span className="label">You</span>
        <p>{question}</p>
      </div>
      <div className="chat-message__answer">
        <span className="label">AgriChat</span>
        <details className="thinking" open>
          <summary>Thinking...</summary>
          <pre>{thinking ?? "Generating reasoning..."}</pre>
        </details>
        {answer && (
          <div className="answer">
            <p>{answer}</p>
          </div>
        )}
      </div>
    </div>
  );
}
