export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api";

export const STREAM_ENDPOINT = `${API_BASE}/query/thinking-stream`;
export const QUERY_ENDPOINT = `${API_BASE}/query`;
export const SESSION_QUERY_ENDPOINT = (sessionId: string) => `${API_BASE}/session/${sessionId}/query`;
export const SESSIONS_ENDPOINT = `${API_BASE}/sessions`;
export const SESSION_DETAIL_ENDPOINT = (sessionId: string) => `${API_BASE}/session/${sessionId}`;
export const DELETE_SESSION_ENDPOINT = (sessionId: string) => `${API_BASE}/delete-session/${sessionId}`;
export const TOGGLE_STATUS_ENDPOINT = (sessionId: string, status: string) => `${API_BASE}/toggle-status/${sessionId}/${status}`;
export const EXPORT_SESSION_ENDPOINT = (sessionId: string) => `${API_BASE}/export/csv/${sessionId}`;
export const RATE_SESSION_ENDPOINT = (sessionId: string) => `${API_BASE}/session/${sessionId}/rate`;

export const DEFAULT_LANGUAGE = "English";
export const DEFAULT_STATE = "";

const credentialsValue = import.meta.env.VITE_REQUEST_CREDENTIALS ?? "omit";
export const REQUEST_CREDENTIALS = (credentialsValue === "include" || credentialsValue === "same-origin")
	? (credentialsValue as RequestCredentials)
	: "omit";
