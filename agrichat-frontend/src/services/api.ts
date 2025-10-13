import {
  SESSIONS_ENDPOINT,
  SESSION_DETAIL_ENDPOINT,
  QUERY_ENDPOINT,
  SESSION_QUERY_ENDPOINT,
  STREAM_ENDPOINT,
  DELETE_SESSION_ENDPOINT,
  TOGGLE_STATUS_ENDPOINT,
  RATE_SESSION_ENDPOINT,
  REQUEST_CREDENTIALS,
} from "../config";
import type {
  QueryRequestPayload,
  SessionListResponse,
  SessionResponse,
  ThinkingStreamEvent,
} from "../types/chat";

async function handleResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  const isJson = contentType.includes("application/json");

  if (!response.ok) {
    if (isJson) {
      try {
        const errorJson = await response.json();
        throw new Error(typeof errorJson === "string" ? errorJson : JSON.stringify(errorJson));
      } catch (jsonError) {
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
      }
    }
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  if (!isJson) {
    const text = await response.text();
    throw new Error(text || "Unexpected non-JSON response from server");
  }

  try {
    return (await response.json()) as T;
  } catch (error) {
    const text = await response.text();
    throw new Error(text || "Failed to parse JSON response");
  }
}

export async function fetchSessions(deviceId: string): Promise<SessionListResponse> {
  const response = await fetch(SESSIONS_ENDPOINT, {
    headers: {
      "X-Device-Id": deviceId,
    },
    credentials: REQUEST_CREDENTIALS,
  });
  return handleResponse(response);
}

export async function fetchSession(sessionId: string, deviceId: string): Promise<SessionResponse> {
  const response = await fetch(SESSION_DETAIL_ENDPOINT(sessionId), {
    headers: {
      "X-Device-Id": deviceId,
    },
    credentials: REQUEST_CREDENTIALS,
  });
  return handleResponse(response);
}

export async function startNewSession(payload: QueryRequestPayload): Promise<SessionResponse> {
  const response = await fetch(QUERY_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    credentials: REQUEST_CREDENTIALS,
  });
  return handleResponse(response);
}

export async function continueSession(
  sessionId: string,
  payload: QueryRequestPayload,
): Promise<SessionResponse> {
  const response = await fetch(SESSION_QUERY_ENDPOINT(sessionId), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    credentials: REQUEST_CREDENTIALS,
  });
  return handleResponse(response);
}

export async function deleteSession(sessionId: string, deviceId: string): Promise<void> {
  const response = await fetch(DELETE_SESSION_ENDPOINT(sessionId), {
    method: "DELETE",
    headers: {
      "X-Device-Id": deviceId,
    },
    credentials: REQUEST_CREDENTIALS,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to delete session ${sessionId}`);
  }
}

export async function toggleSessionStatus(
  sessionId: string,
  status: string,
  deviceId: string,
): Promise<string> {
  const response = await fetch(TOGGLE_STATUS_ENDPOINT(sessionId, status), {
    method: "POST",
    headers: {
      "X-Device-Id": deviceId,
    },
    credentials: REQUEST_CREDENTIALS,
  });
  return handleResponse<{ status: string }>(response).then((data) => data.status);
}

export async function rateSession(
  sessionId: string,
  questionIndex: number,
  rating: "up" | "down",
  deviceId: string,
): Promise<void> {
  const formData = new FormData();
  formData.append("question_index", String(questionIndex));
  formData.append("rating", rating);
  formData.append("device_id", deviceId);

  const response = await fetch(RATE_SESSION_ENDPOINT(sessionId), {
    method: "POST",
    body: formData,
    credentials: REQUEST_CREDENTIALS,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Failed to rate answer");
  }
}

export interface ThinkingStreamOptions {
  onEvent: (event: ThinkingStreamEvent) => void;
  onError?: (error: Error) => void;
  signal?: AbortSignal;
}

export async function streamThinking(
  payload: QueryRequestPayload,
  options: ThinkingStreamOptions,
): Promise<void> {
  const controller = new AbortController();
  const { signal: abortSignal } = controller;

  if (options.signal) {
    options.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    const response = await fetch(STREAM_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: abortSignal,
      credentials: REQUEST_CREDENTIALS,
    });

    if (!response.ok || !response.body) {
      throw new Error(`Streaming request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n\n");
      buffer = lines.pop() ?? "";

      for (const chunk of lines) {
        if (!chunk.startsWith("data:")) {
          continue;
        }
        const jsonText = chunk.slice(5).trim();
        if (!jsonText) {
          continue;
        }
        try {
          const event = JSON.parse(jsonText) as ThinkingStreamEvent;
          options.onEvent(event);
        } catch (error) {
          console.warn("Failed to parse stream event", error, jsonText);
        }
      }
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return;
    }
    if (options.onError) {
      options.onError(error as Error);
    } else {
      console.error("Stream error", error);
    }
  }
}

export async function transcribeAudio(audioFile: File, language?: string): Promise<{ transcript: string }> {
  const formData = new FormData();
  formData.append("file", audioFile);
  if (language) {
    formData.append("language", language);
  }

  const response = await fetch(`${import.meta.env.VITE_API_BASE || "http://localhost:8000"}/api/transcribe-audio`, {
    method: "POST",
    body: formData,
    credentials: REQUEST_CREDENTIALS,
  });

  return handleResponse<{ transcript: string }>(response);
}
