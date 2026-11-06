import type {
  Health,
  SessionDetail,
  SessionSummary,
  SourceItem,
  StreamHandlers,
  TraceStep,
  UsageInfo,
  UserInfo,
} from "./types";

interface ParsedEvent {
  event: string;
  data: Record<string, any>;
}

function parseBlock(block: string): ParsedEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const raw of block.split("\n")) {
    const line = raw.replace(/\r$/, "");
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

function dispatch(block: string, handlers: StreamHandlers): void {
  const parsed = parseBlock(block);
  if (!parsed) return;

  switch (parsed.event) {
    case "meta":
      handlers.onMeta?.(parsed.data as any);
      break;
    case "trace":
      handlers.onTrace?.(parsed.data as TraceStep);
      break;
    case "sources":
      handlers.onSources?.((parsed.data.items ?? []) as SourceItem[]);
      break;
    case "token":
      handlers.onToken?.(String(parsed.data.delta ?? ""));
      break;
    case "usage":
      handlers.onUsage?.(parsed.data as UsageInfo);
      break;
    case "done":
      handlers.onDone?.(parsed.data as any);
      break;
    case "error":
      handlers.onError?.(String(parsed.data.message ?? "请求出错"));
      break;
    default:
      break;
  }
}

export async function fetchHealth(): Promise<Health | null> {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) return null;
    return (await response.json()) as Health;
  } catch {
    return null;
  }
}

export async function fetchUsers(): Promise<UserInfo[]> {
  try {
    const response = await fetch("/api/users");
    if (!response.ok) return [];
    const payload = await response.json();
    return (payload.items ?? []) as UserInfo[];
  } catch {
    return [];
  }
}

export async function fetchSessions(userId: string): Promise<SessionSummary[]> {
  const response = await fetch(`/api/sessions?user_id=${encodeURIComponent(userId)}`);
  if (!response.ok) throw new Error(`加载会话失败（HTTP ${response.status}）`);
  const payload = await response.json();
  return (payload.items ?? []) as SessionSummary[];
}

export async function fetchSession(sessionId: string, userId: string): Promise<SessionDetail> {
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
  );
  if (!response.ok) throw new Error(`加载会话内容失败（HTTP ${response.status}）`);
  return (await response.json()) as SessionDetail;
}

export async function deleteSession(sessionId: string, userId: string): Promise<void> {
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" },
  );
  if (!response.ok) throw new Error(`删除会话失败（HTTP ${response.status}）`);
}

/** 用 fetch + ReadableStream 消费 POST SSE，EventSource 不支持 POST。 */
export async function streamChat(
  message: string,
  sessionId: string | null,
  userId: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, user_id: userId, stream: true }),
    signal,
  });

  if (!response.ok || !response.body) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload.detail ?? "";
    } catch {
      /* 忽略响应体解析失败 */
    }
    throw new Error(detail || `请求失败（HTTP ${response.status}）`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatch(block, handlers);
      boundary = buffer.indexOf("\n\n");
    }
  }

  if (buffer.trim()) dispatch(buffer, handlers);
}
