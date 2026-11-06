export type Role = "user" | "assistant";

export interface SourceItem {
  index: number;
  title: string;
  section: string;
  score: number | null;
  preview?: string;
  acl?: string;
}

export interface UserInfo {
  id: string;
  name: string;
  department_label: string;
  departments: string[];
}

export interface UsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_cny: number;
  calls: number;
  latency_ms: number;
  by_stage?: Record<string, { calls: number; prompt_tokens: number; completion_tokens: number; cost_cny: number }>;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  sources?: SourceItem[];
  usage?: UsageInfo;
  route?: string;
  pending?: boolean;
  error?: string;
}

export interface TraceStep {
  step: string;
  title: string;
  detail?: string;
  status?: "ok" | "warn" | "error";
  latency_ms?: number;
  usage?: { model: string; stage: string; prompt_tokens: number; completion_tokens: number; total_tokens: number };
  data?: {
    intent?: string;
    reason?: string;
    model?: string;
    query?: string;
    timings?: Record<string, number>;
    best_score?: number | null;
    candidates?: CandidateTrace[];
    contexts?: { index: number; title: string; section: string; chars: number }[];
    sufficient?: boolean;
    will_retry?: boolean;
    new_query?: string;
    name?: string;
    arguments?: Record<string, unknown>;
    ok?: boolean;
    result?: string;
    style?: string;
    prompt_chars?: number;
    messages?: { role: string; content: string }[];
    acl?: {
      departments: string[];
      documents_visible: number;
      documents_total: number;
      restricted: boolean;
      pii_allowed?: boolean;
      pii_masked?: number;
    };
  } & Record<string, unknown>;
}

export interface CandidateTrace {
  rank: number;
  child_id: string;
  title: string;
  section: string;
  rrf_score: number;
  dense_rank: number | null;
  bm25_rank: number | null;
  rerank_score: number | null;
  selected: boolean;
  preview: string;
}

export interface SessionMeta {
  session_id: string;
  trace_id: string;
  provider: string;
  mock: boolean;
  model: string;
  history_turns: number;
  summary_chars: number;
  user?: UserInfo;
}

export interface SessionSummary {
  session_id: string;
  user_id: string;
  updated_at: string | null;
  messages: number;
  preview: string;
  summary?: string;
}

export interface SessionDetail {
  session_id: string;
  summary: string;
  turns: { role: Role; content: string; ts: string }[];
  updated_at: string;
  user_id: string;
}

export interface Health {
  status: string;
  provider: string;
  mock_mode: boolean;
  langgraph: boolean;
  models: Record<string, string>;
  index: {
    ready: boolean;
    docs: number;
    parents: number;
    children: number;
    vector_backend: string | null;
    manifest?: Record<string, unknown>;
  };
}

export interface StreamHandlers {
  onMeta?: (meta: SessionMeta) => void;
  onTrace?: (step: TraceStep) => void;
  onSources?: (sources: SourceItem[]) => void;
  onToken?: (delta: string) => void;
  onUsage?: (usage: UsageInfo) => void;
  onDone?: (payload: { session_id: string; route?: string; latency_ms?: number }) => void;
  onError?: (message: string) => void;
}
