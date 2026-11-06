import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  deleteSession,
  fetchHealth,
  fetchSession,
  fetchSessions,
  fetchUsers,
  streamChat,
} from "./api";
import { Composer } from "./components/Composer";
import { MessageItem } from "./components/MessageItem";
import { TracePanel } from "./components/TracePanel";
import type {
  ChatMessage,
  Health,
  SessionMeta,
  SessionSummary,
  TraceStep,
  UserInfo,
} from "./types";

const SUGGESTIONS = [
  { tag: "差旅报销制度", question: "一线城市出差住宿标准每晚多少钱？" },
  { tag: "产品手册", question: "冰感控油洁面乳的水杨酸浓度是多少？" },
  { tag: "订单工具", question: "帮我查一下订单 ORD-2026-1004 的退款进度" },
  { tag: "客服 SOP", question: "七天无理由退货的运费谁承担？" },
];

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

function EmptyState({ onPick, user }: { onPick: (question: string) => void; user?: UserInfo }) {
  return (
    <div className="empty">
      <p className="empty-kicker">内部知识库 · 业务系统</p>
      <h2>先问一个具体的问题</h2>
      <p className="empty-desc">
        答案会标注引用来源；订单类问题会直接调用业务工具查询，而不是让模型去猜。
      </p>
      {user && (
        <p className="empty-user">
          当前身份：{user.name} · {user.department_label}
          <span className="empty-user-hint">切换身份可验证文档级权限</span>
        </p>
      )}
      <div className="cards">
        {SUGGESTIONS.map((item) => (
          <button type="button" className="index-card" key={item.question} onClick={() => onPick(item.question)}>
            <span className="card-tag">{item.tag}</span>
            <span className="card-q">{item.question}</span>
          </button>
        ))}
      </div>
      <p className="empty-foot">
        已索引：产品手册 / 差旅报销制度 / 客服 SOP / 数据安全规范 / 经销商政策 / 薪酬与期权 / 渠道返点
      </p>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [trace, setTrace] = useState<TraceStep[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [meta, setMeta] = useState<SessionMeta | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [userId, setUserId] = useState("carol");
  const [streaming, setStreaming] = useState(false);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState("");
  const [historyNotice, setHistoryNotice] = useState(false);
  // 窄屏默认收起链路抽屉，否则一进来就盖住聊天区
  const [traceOpen, setTraceOpen] = useState<boolean>(
    () => typeof window === "undefined" || window.innerWidth > 980,
  );

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void fetchHealth().then(setHealth);
    void fetchUsers().then((list) => {
      setUsers(list);
      if (list.length > 0 && !list.some((item) => item.id === "carol")) {
        setUserId(list[0].id);
      }
    });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  const patchMessage = useCallback(
    (id: string, patch: Partial<ChatMessage> | ((message: ChatMessage) => Partial<ChatMessage>)) => {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === id
            ? { ...message, ...(typeof patch === "function" ? patch(message) : patch) }
            : message,
        ),
      );
    },
    [],
  );

  const send = useCallback(
    async (text: string) => {
      if (streaming) return;

      const assistantId = uid();
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "user", content: text },
        { id: assistantId, role: "assistant", content: "", pending: true },
      ]);
      setTrace([]);
      setHistoryNotice(false);
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamChat(
          text,
          sessionId,
          userId,
          {
            onMeta: (nextMeta) => {
              setMeta(nextMeta);
              setSessionId(nextMeta.session_id);
            },
            onTrace: (step) => setTrace((prev) => [...prev, step]),
            onSources: (sources) => patchMessage(assistantId, { sources }),
            onToken: (delta) => patchMessage(assistantId, (message) => ({ content: message.content + delta })),
            onUsage: (usage) => patchMessage(assistantId, { usage }),
            onDone: (payload) => patchMessage(assistantId, { pending: false, route: payload.route }),
            onError: (message) => patchMessage(assistantId, { pending: false, error: message }),
          },
          controller.signal,
        );
      } catch (error) {
        const err = error as Error;
        if (err.name === "AbortError") {
          patchMessage(assistantId, { pending: false });
        } else {
          patchMessage(assistantId, { pending: false, error: err.message || "网络异常，请重试" });
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [patchMessage, sessionId, streaming, userId],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const switchUser = useCallback(
    (nextUserId: string) => {
      if (nextUserId === userId) return;
      abortRef.current?.abort();
      setUserId(nextUserId);
      setMessages([]);
      setTrace([]);
      setSessionId(null);
      setMeta(null);
      setHistoryNotice(false);
      setSessionsOpen(false);
    },
    [userId],
  );

  const startNewChat = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setTrace([]);
    setSessionId(null);
    setMeta(null);
    setHistoryNotice(false);
  }, []);

  const openSessions = useCallback(async () => {
    setSessionsOpen(true);
    setSessionsLoading(true);
    setSessionsError("");
    try {
      const items = await fetchSessions(userId);
      setSessions(items);
      // 旧版后端不返回 user_id，也不会按身份过滤；明确提示而不是静默展示
      if (items.length > 0 && items[0].user_id === undefined) {
        setSessionsError(
          "后端版本过旧：会话列表没有按身份过滤。请重建后端镜像后重试（docker compose build backend）。",
        );
      }
    } catch (error) {
      setSessionsError((error as Error).message);
    } finally {
      setSessionsLoading(false);
    }
  }, [userId]);

  const resumeSession = useCallback(
    async (item: SessionSummary) => {
      try {
        const detail = await fetchSession(item.session_id, userId);
        if (detail.turns.length === 0) {
          // 旧版后端会用 guest 身份打开别人的会话，返回一个空的新会话
          setSessionsError(
            item.messages > 0
              ? "后端返回了空会话：backend 镜像可能还是旧版本（会话未按身份加载）。请重建后端镜像后重试。"
              : "这个会话没有可恢复的消息。",
          );
          return;
        }
        abortRef.current?.abort();
        setMessages(
          detail.turns.map((turn, index) => ({
            id: `history-${item.session_id}-${index}`,
            role: turn.role,
            content: turn.content,
          })),
        );
        setSessionId(detail.session_id);
        setMeta(null);
        setTrace([]);
        setHistoryNotice(true);
        setSessionsError("");
        setSessionsOpen(false);
      } catch (error) {
        setSessionsError((error as Error).message);
      }
    },
    [userId],
  );

  const removeSession = useCallback(
    async (id: string) => {
      try {
        await deleteSession(id, userId);
        setSessions((prev) => prev.filter((item) => item.session_id !== id));
        if (id === sessionId) startNewChat();
      } catch (error) {
        setSessionsError((error as Error).message);
      }
    },
    [sessionId, startNewChat, userId],
  );

  const currentUser = useMemo(() => users.find((item) => item.id === userId), [users, userId]);

  const lastUsage = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].usage) return messages[i].usage;
    }
    return undefined;
  }, [messages]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="seal" aria-hidden="true">
            然
          </span>
          <div className="brand-text">
            <h1>理然 · 企业知识中枢</h1>
            <p>制度问答 / 产品资料 / 订单工具</p>
          </div>
        </div>

        <div className="topbar-right">
          {users.length > 0 && (
            <label className="user-picker">
              <span className="picker-label">身份</span>
              <select
                value={userId}
                onChange={(event) => switchUser(event.target.value)}
                aria-label="切换演示身份"
              >
                {users.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} · {item.department_label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <span className={`pill ${health?.mock_mode ? "pill-warn" : "pill-ok"}`}>
            {health === null ? "连接中" : health.mock_mode ? "MOCK 演示" : "模型在线"}
          </span>
          <span className="pill pill-plain">
            {health ? (health.langgraph ? "LangGraph" : "mini-graph") : "编排"}
          </span>
          <span className="pill pill-plain hide-sm">
            {health ? `${health.index.parents} 父块 · ${health.index.children} 子块` : "索引加载中"}
          </span>
          <button
            type="button"
            className="ghost-btn"
            onClick={startNewChat}
            disabled={messages.length === 0 && !streaming}
            title="清空当前对话，开始新会话"
          >
            新对话
          </button>
          <button type="button" className="ghost-btn" onClick={openSessions} title="历史会话">
            记录
          </button>
          <button
            type="button"
            className="ghost-btn trace-toggle"
            aria-pressed={traceOpen}
            onClick={() => setTraceOpen((value) => !value)}
          >
            {traceOpen ? "隐藏链路" : "查看链路"}
          </button>
        </div>
      </header>

      <main className={`workspace ${traceOpen ? "with-trace" : ""}`}>
        <section className="chat-col">
          <div className="messages" role="log" aria-live="polite">
            {historyNotice && (
              <div className="history-note">
                已恢复历史会话：历史消息不包含引用与用量，继续提问会沿用原有上下文。
              </div>
            )}
            {messages.length === 0 ? (
              <EmptyState onPick={send} user={currentUser} />
            ) : (
              messages.map((message) => <MessageItem key={message.id} message={message} />)
            )}
            <div ref={bottomRef} />
          </div>
          <Composer disabled={streaming} streaming={streaming} onSend={send} onStop={stop} />
        </section>

        <TracePanel
          steps={trace}
          usage={lastUsage}
          health={health}
          meta={meta}
          open={traceOpen}
          onClose={() => setTraceOpen(false)}
        />
      </main>

      {sessionsOpen && (
        <>
          <div className="drawer-backdrop" onClick={() => setSessionsOpen(false)} />
          <aside className="sessions-drawer" aria-label="历史会话">
            <header className="sessions-head">
              <div>
                <h2>历史会话</h2>
                <p className="sessions-sub">仅显示当前身份的会话</p>
              </div>
              <button
                type="button"
                className="icon-btn"
                onClick={() => setSessionsOpen(false)}
                aria-label="关闭历史会话"
              >
                ×
              </button>
            </header>

            <div className="sessions-list">
              {sessionsLoading && <p className="sessions-empty">加载中…</p>}
              {!sessionsLoading && sessionsError && <p className="sessions-empty">{sessionsError}</p>}
              {!sessionsLoading && !sessionsError && sessions.length === 0 && (
                <p className="sessions-empty">
                  还没有历史会话。问几个问题之后，这里会按时间倒序出现。
                </p>
              )}
              {sessions.map((item) => (
                <div className="session-item" key={item.session_id}>
                  <button
                    type="button"
                    className={`session-open ${item.session_id === sessionId ? "is-active" : ""}`}
                    onClick={() => resumeSession(item)}
                  >
                    <span className="session-preview">{item.preview || "（空会话）"}</span>
                    <span className="session-meta">
                      {item.messages} 条 · {item.updated_at || ""}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="session-del"
                    onClick={() => removeSession(item.session_id)}
                    aria-label="删除会话"
                    title="删除该会话"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>

            <footer className="sessions-foot">删除不可恢复 · 会话按身份隔离</footer>
          </aside>
        </>
      )}

      {!traceOpen && (
        <button type="button" className="trace-fab" onClick={() => setTraceOpen(true)}>
          链路
        </button>
      )}
    </div>
  );
}
