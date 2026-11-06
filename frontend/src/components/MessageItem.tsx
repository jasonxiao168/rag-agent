import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";

const ROUTE_LABEL: Record<string, string> = {
  kb: "知识检索",
  tool: "业务工具",
  smalltalk: "闲聊",
  clarify: "补充信息",
};

/**
 * Markdown 会把 `138****5678` 里的连续星号当成加粗标记，渲染后星号消失，
 * 掩码看起来就变成了 1385678。这里把 3 个以上的连续星号替换成外观接近的
 * U+2217（∗），既保留掩码形态，又不会被 Markdown 解析。
 */
function protectMasks(text: string): string {
  return text.replace(/\*{3,}/g, (match) => "∗".repeat(match.length));
}

function MarkdownView({ content }: { content: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="table-scroll">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {protectMasks(content)}
      </ReactMarkdown>
    </div>
  );
}

export function MessageItem({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const routeLabel = message.route ? ROUTE_LABEL[message.route] ?? message.route : null;
  const showCaret = Boolean(message.pending && message.content);

  return (
    <article className="msg msg-assistant">
      <header className="msg-head">
        <span className="who">小然</span>
        {routeLabel && <span className="route-tag">{routeLabel}</span>}
        {message.pending && <span className="live-dot" title="生成中" />}
      </header>

      <div className="answer">
        {message.content ? <MarkdownView content={message.content} /> : null}
        {!message.content && message.pending ? <span className="caret" /> : null}
        {showCaret ? <span className="caret" /> : null}
        {message.error ? <p className="error-text">{message.error}</p> : null}
      </div>

      {message.sources && message.sources.length > 0 && (
        <footer className="sources">
          <span className="sources-label">依据</span>
          {message.sources.map((source) => (
            <span className="source-chip" key={`${message.id}-${source.index}`} title={source.preview}>
              <b>{source.index}</b>
              <span>
                {source.title} · {source.section || "全文"}
              </span>
              {source.acl && source.acl !== "all" && <em className="acl-badge">受限</em>}
            </span>
          ))}
        </footer>
      )}

      {message.usage && (
        <footer className="usage-line">
          <span>{message.usage.total_tokens} tokens</span>
          <span>¥{message.usage.cost_cny.toFixed(4)}</span>
          <span>{Math.round(message.usage.latency_ms)} ms</span>
          <span>{message.usage.calls} 次模型调用</span>
        </footer>
      )}
    </article>
  );
}
