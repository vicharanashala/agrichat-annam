import { useState, useMemo } from "react";
import { Copy, ThumbsUp, ThumbsDown, MoreVertical } from "lucide-react";
import DOMPurify from "dompurify";
import type { SessionMessage } from "../types/chat";
import { Button } from "./ui/Button";
import { cn } from "../utils/cn";

interface ChatMessageProps {
  message: SessionMessage;
  index: number;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [showActions, setShowActions] = useState(false);
  const [copied, setCopied] = useState(false);

  const safeAnswer = useMemo(() => {
    const html = message.answer || message.final_answer || "";
    return { __html: DOMPurify.sanitize(html) };
  }, [message.answer, message.final_answer]);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="space-y-4 animate-fade-in">
      {/* User Message */}
      <div className="flex justify-end">
        <div className="bg-blue-600 text-white rounded-2xl px-4 py-3 max-w-[80%]">
          <p className="whitespace-pre-wrap">{message.question}</p>
        </div>
      </div>

      {/* Assistant Response */}
      <div className="flex justify-start">
        <div className="flex-1 max-w-[85%]">
          <div 
            className="bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-2xl px-4 py-3"
            onMouseEnter={() => setShowActions(true)}
            onMouseLeave={() => setShowActions(false)}
          >
            <div 
              className="prose prose-sm max-w-none dark:prose-invert"
              dangerouslySetInnerHTML={safeAnswer}
            />
            
            {/* Action Buttons */}
            <div className={cn(
              "flex items-center gap-1 mt-3 pt-2 border-t border-border/30 transition-opacity",
              showActions ? "opacity-100" : "opacity-0"
            )}>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard(message.answer || "")}
                className="h-8 px-2 text-xs"
              >
                <Copy className="h-3 w-3 mr-1" />
                {copied ? "Copied!" : "Copy"}
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2 text-xs"
              >
                <ThumbsUp className="h-3 w-3" />
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2 text-xs"
              >
                <ThumbsDown className="h-3 w-3" />
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2 text-xs ml-auto"
              >
                <MoreVertical className="h-3 w-3" />
              </Button>
            </div>
          </div>

          {/* Research Data Sources */}
          {Array.isArray(message.research_data) && message.research_data.length > 0 && (
            <div className="mt-2 text-xs text-muted-foreground">
              <details className="cursor-pointer">
                <summary className="hover:text-foreground">
                  Sources ({message.research_data.length})
                </summary>
                <div className="mt-1 space-y-1 pl-3 border-l border-border">
                  {Array.isArray(message.research_data) && message.research_data.map((source: any, idx: number) => (
                    <div key={idx} className="text-xs">
                      <span className="font-medium">{source.source}:</span> {source.content_preview}
                      {source.metadata?.cosine && (
                        <span className="text-muted-foreground ml-1">
                          ({Math.round(source.metadata.cosine * 100)}% match)
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
