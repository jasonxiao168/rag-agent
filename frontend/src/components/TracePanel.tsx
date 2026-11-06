import type { CandidateTrace, Health, SessionMeta, TraceStep, UsageInfo } from "../types";

const INTENT_LABEL: Record<string, string> = {
  kb: "查知识库",
  tool: "调业务工具",
  smalltalk: "闲聊",
  clarify: "需要追问",
};

function Candidates({ candidates }: { candidates?: CandidateTrace[] }) {
  if (!candidates || candidates.length === 0) return null;
  const maxScore = Math.max(...candidates.map((item) => item.rerank_score ?? item.rrf_score ?? 0), 0.0001);

  return (
    <div className="candidates">
      <div className="cand-head">
        <span>候选子块</span>
        <span>粗排 / 精排</span>
      </div>
      {candidates.slice(0, 5).map((item) => {
        const score = item.rerank_score ?? item.rrf_score;
        const width = Math.max(4, Math.min(100, ((score ?? 0) / maxScore) * 100));
        return (
          <div className={`cand-row ${item.selected ? "is-selected" : ""}`} key={item.child_id}>
            <div className="cand-title" title={item.preview}>
              <span className="cand-rank">#{item.rank}</span>
              <span className="cand-sec">{item.section || item.title}</span>
            </div>
            <div className="cand-bar">
              <span style={{ width: `${width}%` }} />
            </div>
            <div className="cand-meta">
              <span>d{item.dense_rank ?? "—"}</span>
              <span>b{item.bm25_rank ?? "—"}</span>
              <span className="cand-score">
                {item.rerank_score === null ? "—" : item.rerank_score.toFixed(3)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ToolDetail({ step }: { step: TraceStep }) {
  const args = step.data?.arguments ?? {};
  const result = String(step.data?.result ?? "");
  return (
    <div className="tool-detail">
      <div className="tool-args">
        {Object.entries(args).map(([key, value]) => (
          <span key={key}>
            {key}: <b>{String(value)}</b>
          </span>
        ))}
      </div>
      {result && <pre className="tool-result">{result}</pre>}
    </div>
  );
}

const ROLE_LABEL: Record<string, string> = {
  system: "system · 系统提示",
  user: "user · 用户输入",
  assistant: "assistant",
};

function PromptPreview({ messages }: { messages?: { role: string; content: string }[] }) {
  if (!messages || messages.length === 0) return null;
  const chars = messages.reduce((total, message) => total + message.content.length, 0);

  return (
    <details className="prompt-preview">
      <summary>
        查看完整 Prompt（{messages.length} 条 · {chars} 字符）
      </summary>
      <div className="prompt-body">
        {messages.map((message, index) => (
          <div className="prompt-msg" key={index}>
            <div className="prompt-role">{ROLE_LABEL[message.role] ?? message.role}</div>
            <pre>{message.content}</pre>
          </div>
        ))}
      </div>
    </details>
  );
}

function StepItem({ step, order }: { step: TraceStep; order: number }) {
  const status = step.status ?? "ok";
  const intent = step.data?.intent;
  const willRetry = step.data?.will_retry === true;
  const routeModel = step.data?.model;
  const acl = step.data?.acl;

  const aclNotes: string[] = [];
  if (acl?.restricted) {
    aclNotes.push(`可见 ${acl.documents_visible}/${acl.documents_total} 篇`);
  }
  if ((acl?.pii_masked ?? 0) > 0) {
    aclNotes.push(`字段脱敏 ${acl?.pii_masked} 处`);
  }

  return (
    <li className={`step step-${status}`}>
      <span className="step-no">{String(order).padStart(2, "0")}</span>
      <div className="step-main">
        <div className="step-top">
          <h3>{step.title}</h3>
          {typeof step.latency_ms === "number" && (
            <span className="ms">{Math.round(step.latency_ms)}ms</span>
          )}
        </div>
        {step.detail && <p className="step-detail">{step.detail}</p>}

        {step.step === "route" && intent && (
          <div className="tag-row">
            <span className="mini-tag">{INTENT_LABEL[intent] ?? intent}</span>
            {routeModel ? <span className="mini-tag ghost">{String(routeModel)}</span> : null}
            {willRetry ? <span className="mini-tag warn">将重试</span> : null}
          </div>
        )}

        {step.step === "retrieval" && <Candidates candidates={step.data?.candidates} />}
        {step.step === "retrieval" && aclNotes.length > 0 && (
          <p className="acl-line">{aclNotes.join(" · ")}</p>
        )}
        {step.step === "tool_call" && <ToolDetail step={step} />}
        {step.step === "compose" && <PromptPreview messages={step.data?.messages} />}

        {step.usage && (
          <p className="step-usage">
            {step.usage.model} · {step.usage.total_tokens} tok
          </p>
        )}
      </div>
    </li>
  );
}

export function TracePanel({
  steps,
  usage,
  health,
  meta,
  open,
  onClose,
}: {
  steps: TraceStep[];
  usage?: UsageInfo;
  health: Health | null;
  meta: SessionMeta | null;
  open: boolean;
  onClose: () => void;
}) {
  return (
    <aside className={`trace-col ${open ? "open" : ""}`} aria-label="执行链路">
      <header className="trace-head">
        <div>
          <h2>执行链路</h2>
          <p className="trace-sub">
            {steps.length > 0 ? `${steps.length} 步` : "等待提问"}
            {health ? ` · ${health.index.parents} 父块 / ${health.index.children} 子块` : ""}
          </p>
        </div>
        <button type="button" className="icon-btn close-trace" onClick={onClose} aria-label="关闭链路面板">
          ×
        </button>
      </header>

      <div className="trace-body">
        {steps.length === 0 ? (
          <div className="trace-empty">
            <p>提问后，这里会依次出现：</p>
            <p className="trace-empty-list">路由判定 → 混合检索 → Rerank → 质量判定 → 工具调用 → 组装 Prompt</p>
            <p>每一步都会带上耗时与 token 消耗。</p>
          </div>
        ) : (
          <ol className="timeline">
            {steps.map((step, index) => (
              <StepItem key={`${step.step}-${index}`} step={step} order={index + 1} />
            ))}
          </ol>
        )}
      </div>

      <footer className="trace-foot">
        {meta && (
          <p className="trace-meta-line">
            {meta.provider === "mock" ? "MOCK 模式" : meta.provider} · 会话 {meta.session_id.slice(0, 10)} ·
            历史 {meta.history_turns} 条
          </p>
        )}
        {usage ? (
          <div className="usage-grid">
            <div>
              <span>输入</span>
              <b>{usage.prompt_tokens}</b>
            </div>
            <div>
              <span>输出</span>
              <b>{usage.completion_tokens}</b>
            </div>
            <div>
              <span>成本</span>
              <b>¥{usage.cost_cny.toFixed(4)}</b>
            </div>
            <div>
              <span>耗时</span>
              <b>{Math.round(usage.latency_ms)}ms</b>
            </div>
          </div>
        ) : (
          <p className="trace-meta-line">Token 与成本将在回答完成后汇总。</p>
        )}
      </footer>
    </aside>
  );
}
