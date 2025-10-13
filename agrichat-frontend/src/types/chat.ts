export interface DatabaseToggleConfig {
  golden_enabled: boolean;
  pops_enabled: boolean;
  llm_enabled: boolean;
  similarity_threshold: number;
  pops_similarity_threshold: number;
  enable_adaptive_thresholds: boolean;
  strict_validation: boolean;
  show_database_path: boolean;
  show_confidence_scores: boolean;
}

export interface QueryRequestPayload {
  question: string;
  device_id: string;
  state: string;
  language: string;
  database_config: DatabaseToggleConfig;
}

export interface SessionMessage {
  question: string;
  answer: string;
  final_answer?: string;
  thinking?: string;
  metadata?: Record<string, unknown>;
  pipeline_metadata?: Record<string, unknown>;
  research_data?: unknown;
  sources?: Array<{
    source: string;
    preview?: string;
    metadata?: Record<string, unknown>;
  }>;
  confidence?: number;
  similarity?: number;
  distance?: number;
  rating?: string | null;
  ragas_score?: number;
  reasoning_trace?: string[];
  clarifying_questions?: string[];
  context_note?: string;
}

export interface SessionDocument {
  session_id: string;
  timestamp: string;
  status: "active" | "archived" | string;
  messages: SessionMessage[];
  crop?: string;
  state?: string;
  language?: string;
  has_unread?: boolean;
  device_id?: string;
  recommendations?: Recommendation[];
}

export interface Recommendation {
  title: string;
  description?: string;
  link?: string;
}

export interface SessionListResponse {
  sessions: SessionDocument[];
}

export interface SessionResponse {
  session: SessionDocument;
}

export interface StreamingEventBase {
  type: string;
}

export interface ThinkingStartEvent extends StreamingEventBase {
  type: "thinking_start";
}

export interface ThinkingTokenEvent extends StreamingEventBase {
  type: "thinking_token";
  text: string;
}

export interface ThinkingCompleteEvent extends StreamingEventBase {
  type: "thinking_complete";
  thinking?: string;
}

export interface AnswerEvent extends StreamingEventBase {
  type: "answer";
  answer: string;
  source?: string;
  confidence?: number;
  metadata?: Record<string, unknown>;
  database_path?: string[];
  confidence_scores?: Record<string, unknown>;
  reasoning_steps?: string[];
  thinking?: string;
}

export interface SessionCompleteEvent extends StreamingEventBase {
  type: "session_complete";
  session: SessionDocument;
}

export interface StreamEndEvent extends StreamingEventBase {
  type: "stream_end";
}

export interface ErrorEvent extends StreamingEventBase {
  type: "error";
  message: string;
}

export type ThinkingStreamEvent =
  | ThinkingStartEvent
  | ThinkingTokenEvent
  | ThinkingCompleteEvent
  | AnswerEvent
  | SessionCompleteEvent
  | StreamEndEvent
  | ErrorEvent;
