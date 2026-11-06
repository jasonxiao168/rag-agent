"""全局配置。

设计原则：
1. 只依赖标准库，保证任何环境（含无外网 CI）都能 import；
2. 所有配置都可以用环境变量覆盖，方便 Docker / 本地 / 评测脚本三套环境共用；
3. 没有 DASHSCOPE_API_KEY 时自动进入 mock 模式，整条链路仍然可跑通（见 llm/client.py）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析器，避免为一个 20 行的功能引入 python-dotenv。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 环境变量优先级高于 .env
        os.environ.setdefault(key, value)


_load_dotenv(REPO_DIR / ".env")
_load_dotenv(BACKEND_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in _env_str(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # ---------- 服务 ----------
    app_name: str = "Enterprise RAG & Agent Bot"
    host: str = _env_str("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    )

    # ---------- 模型（对话 / 向量 / 精排可分别选择） ----------
    # LLM_PROVIDER: auto | deepseek | dashscope | mock
    llm_provider: str = _env_str("LLM_PROVIDER", "auto")

    dashscope_api_key: str = _env_str("DASHSCOPE_API_KEY", "")
    base_url: str = _env_str("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    native_base_url: str = _env_str("DASHSCOPE_NATIVE_BASE_URL", "https://dashscope.aliyuncs.com")

    deepseek_api_key: str = _env_str("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = _env_str("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_fast_model: str = _env_str("DEEPSEEK_FAST_MODEL", "deepseek-chat")

    # 生成用大模型，路由/摘要/判定用小模型（成本优化的关键）
    model_generation: str = _env_str("QWEN_GENERATION_MODEL", "qwen-plus")
    model_fast: str = _env_str("QWEN_FAST_MODEL", "qwen-turbo")
    model_embedding: str = _env_str("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
    model_rerank: str = _env_str("QWEN_RERANK_MODEL", "gte-rerank-v2")

    # 向量 / 精排后端：auto | fastembed | dashscope | hash/lexical/llm
    embedding_backend: str = _env_str("EMBEDDING_BACKEND", "auto")
    rerank_backend: str = _env_str("RERANK_BACKEND", "auto")
    fastembed_embed_model: str = _env_str("FASTEMBED_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    fastembed_rerank_model: str = _env_str("FASTEMBED_RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8")
    fastembed_cache_dir: str = _env_str("FASTEMBED_CACHE_DIR", "")
    # HuggingFace 镜像；受限网络下设为 https://hf-mirror.com
    hf_endpoint: str = _env_str("HF_ENDPOINT", "")

    embed_dim: int = _env_int("EMBED_DIM", 1024)
    embed_batch_size: int = _env_int("EMBED_BATCH_SIZE", 10)  # DashScope text-embedding-v3 单次上限 10 条
    request_timeout: float = _env_float("REQUEST_TIMEOUT", 60.0)

    # ---------- 检索 ----------
    parent_size: int = _env_int("PARENT_SIZE", 900)      # 父块：喂给大模型的上下文粒度
    child_size: int = _env_int("CHILD_SIZE", 220)        # 子块：用于精准召回
    parent_overlap: int = _env_int("PARENT_OVERLAP", 120)
    child_overlap: int = _env_int("CHILD_OVERLAP", 40)
    retrieve_top_k: int = _env_int("RETRIEVE_TOP_K", 10)  # 混合检索粗排数量
    rerank_top_n: int = _env_int("RERANK_TOP_N", 3)       # 精排后进入 Prompt 的数量
    use_hybrid: bool = _env_bool("USE_HYBRID", True)
    use_rerank: bool = _env_bool("USE_RERANK", True)
    context_max_chars: int = _env_int("CONTEXT_MAX_CHARS", 3200)
    rrf_k: int = _env_int("RRF_K", 60)
    min_rerank_score: float = _env_float("MIN_RERANK_SCORE", 0.05)
    max_rewrite_retries: int = _env_int("MAX_REWRITE_RETRIES", 1)
    # 字段级脱敏：文档可见，但手机号/身份证等 PII 需额外权限（permissions 含 pii）才展示明文
    mask_pii: bool = _env_bool("MASK_PII", True)

    # 检索质量分级：高于 high 直接采信，低于 low 直接判不足，中间灰区才调用小模型判定
    # （省掉一次 LLM 调用就是省钱，见 core/tokens.py 的成本统计）
    grade_high_score: float = _env_float("GRADE_HIGH_SCORE", 0.50)
    grade_low_score: float = _env_float("GRADE_LOW_SCORE", 0.12)

    # ---------- 上下文管理 ----------
    max_history_turns: int = _env_int("MAX_HISTORY_TURNS", 4)   # 滑动窗口保留的最近轮数
    summary_trigger_turns: int = _env_int("SUMMARY_TRIGGER_TURNS", 6)  # 超过该轮数触发摘要压缩

    # ---------- 路径 ----------
    data_dir: Path = field(default_factory=lambda: Path(_env_str("DATA_DIR", str(REPO_DIR / "data"))))
    raw_dir_name: str = "raw"
    index_dir_name: str = "index"
    session_dir_name: str = "sessions"

    # ---------- 成本（人民币 / 千 token，按各家官网价目表，可自行更新） ----------
    price_cny_per_1k: dict = field(
        default_factory=lambda: {
            "qwen-plus": (0.0008, 0.0020),
            "qwen-turbo": (0.0003, 0.0006),
            "qwen-max": (0.0024, 0.0096),
            "text-embedding-v3": (0.0005, 0.0),
            "gte-rerank-v2": (0.0008, 0.0),
            # DeepSeek 官方价目：输入 ¥2/百万、输出 ¥8/百万（缓存命中另计）
            "deepseek-chat": (0.0020, 0.0080),
            "deepseek-flash": (0.0020, 0.0080),
            "fastembed": (0.0, 0.0),
            "mock": (0.0, 0.0),
        }
    )

    @property
    def generation_model(self) -> str:
        return self.deepseek_model if self.provider == "deepseek" else self.model_generation

    @property
    def fast_model(self) -> str:
        return self.deepseek_fast_model if self.provider == "deepseek" else self.model_fast

    @property
    def provider(self) -> str:
        """对话模型提供方：显式配置优先，其次按 Key 推断。"""
        explicit = self.llm_provider.strip().lower()
        if explicit in {"deepseek", "dashscope", "mock"}:
            return explicit
        if self.deepseek_api_key:
            return "deepseek"
        if self.dashscope_api_key:
            return "dashscope"
        return "mock"

    @property
    def mock_mode(self) -> bool:
        """显式 MOCK_MODE 优先；否则没有可用 Key 就进入 mock，保证 demo 可跑。"""
        explicit = os.environ.get("MOCK_MODE", "auto").strip().lower()
        if explicit in {"1", "true", "yes", "on"}:
            return True
        if explicit in {"0", "false", "no", "off"}:
            return False
        return self.provider == "mock"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / self.raw_dir_name

    @property
    def index_dir(self) -> Path:
        return self.data_dir / self.index_dir_name

    @property
    def session_dir(self) -> Path:
        return self.data_dir / self.session_dir_name

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.index_dir, self.session_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
