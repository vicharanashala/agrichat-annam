interface StreamingMessageProps {
  question: string;
  thinking?: string;
  answer?: string;
}

export function StreamingMessage({ question, thinking, answer }: StreamingMessageProps) {
  return (
    <div className="space-y-4 animate-fade-in">
      {/* User Message */}
      <div className="flex justify-end">
        <div className="bg-blue-600 text-white rounded-2xl px-4 py-3 max-w-[80%]">
          <p className="whitespace-pre-wrap">{question}</p>
        </div>
      </div>

      {/* Assistant Response - Streaming */}
      <div className="flex justify-start">
        <div className="flex-1 max-w-[85%]">
          <div className="bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-2xl px-4 py-3">
            {thinking && (
              <details className="mb-3" open>
                <summary className="text-sm font-medium cursor-pointer hover:text-gray-600 dark:hover:text-gray-300">
                  🤔 Thinking...
                </summary>
                <div className="mt-2 text-sm text-gray-600 dark:text-gray-400 whitespace-pre-wrap font-mono text-xs bg-gray-50 dark:bg-gray-800 p-2 rounded">
                  {thinking}
                </div>
              </details>
            )}
            
            {answer ? (
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <p>{answer}</p>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
                <div className="typing-indicator">
                  <div className="typing-dot" style={{ animationDelay: '0ms' }}></div>
                  <div className="typing-dot" style={{ animationDelay: '150ms' }}></div>
                  <div className="typing-dot" style={{ animationDelay: '300ms' }}></div>
                </div>
                <span className="text-sm">AgriChat is thinking...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
