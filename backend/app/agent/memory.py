"""对话记忆：滑动窗口 + 摘要压缩。

规则：
- 最近 N 轮原样保留（保证指代与语气连贯）；
- 超出窗口的旧消息，用小模型（qwen-turbo）压成一段 200 字摘要；
- 摘要 + 最近 N 轮 一起注入 Prompt，token 从「线性增长」变成「近似常数」。

面试可讲：直接全量拼接历史，10 轮对话后 prompt 会膨胀到 5000+ token，
而且大部分信息与当前问题无关；摘要压缩把成本压到 O(1)。
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.core.logging import get_logger
from app.core.tokens import Usage
from app.llm.client import LLMClient, get_llm
from app.llm.prompts import summary_messages

logger = get_logger("agent.memory")

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class Turn:
    role: str  # user / assistant
    content: str
    ts: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "ts": self.ts}


class SessionMemory:
    def __init__(
        self,
        session_id: str,
        llm: LLMClient | None = None,
        path: Path | None = None,
        user_id: str = "guest",
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.llm = llm or get_llm()
        self.path = path
        self.summary: str = ""
        self.turns: list[Turn] = []
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at: str = self.created_at

    # -- 读写 -------------------------------------------------------------- #
    def append(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "summary": self.summary,
            "turns": [t.as_dict() for t in self.turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, session_id: str, path: Path, llm: LLMClient | None = None, user_id: str = "guest"
    ) -> "SessionMemory":
        memory = cls(session_id, llm=llm, path=path, user_id=user_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                memory.summary = data.get("summary", "")
                memory.turns = [Turn(**item) for item in data.get("turns", [])]
                memory.created_at = data.get("created_at", memory.created_at)
                memory.updated_at = data.get("updated_at", memory.updated_at)
                # 老会话文件可能没有 user_id，按 guest 处理，避免被任意身份认领
                memory.user_id = data.get("user_id", "guest")
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("会话文件损坏，已重置 %s：%s", session_id, exc)
        return memory

    # -- 上下文 ------------------------------------------------------------ #
    def recent_turns(self, limit: int | None = None) -> list[dict]:
        keep = (limit or settings.max_history_turns) * 2
        return [t.as_dict() for t in self.turns[-keep:]]

    def context(self) -> tuple[str, list[dict]]:
        """返回 (历史摘要, 最近若干条消息)，供 Prompt 注入。"""
        return self.summary, self.recent_turns()

    # -- 摘要压缩 ---------------------------------------------------------- #
    def finalize_turn(self, question: str, answer: str) -> list[Usage]:
        """一轮对话结束后统一落库：写入两条消息，必要时触发摘要。"""
        self.append("user", question)
        self.append("assistant", answer)
        return self.maybe_summarize()

    def needs_summary(self) -> bool:
        keep = settings.max_history_turns * 2
        # 触发阈值不能小于窗口本身，否则刚压缩完又立刻触发
        trigger = max(settings.summary_trigger_turns, keep + 2)
        return len(self.turns) > trigger

    def maybe_summarize(self) -> list[Usage]:
        if not self.needs_summary():
            return []

        keep = settings.max_history_turns * 2
        old_turns = self.turns[:-keep]
        transcript_parts = []
        if self.summary:
            transcript_parts.append(f"【此前的摘要】{self.summary}")
        transcript_parts.extend(
            f"{'用户' if t.role == 'user' else '助手'}：{t.content}" for t in old_turns
        )
        transcript = "\n".join(transcript_parts)

        result = self.llm.chat(
            summary_messages(transcript),
            model=settings.fast_model,
            stage="summarize",
            max_tokens=300,
            temperature=0.2,
        )
        self.summary = result.text.strip()
        self.turns = self.turns[-keep:]
        logger.info("会话 %s 已压缩历史，当前保留 %d 条消息", self.session_id, len(self.turns))
        return [result.usage]


class SessionStore:
    def __init__(self, session_dir: Path | None = None) -> None:
        self.dir = Path(session_dir or settings.session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, SessionMemory] = {}

    @staticmethod
    def new_id() -> str:
        return f"s_{uuid.uuid4().hex[:12]}"

    def path_for(self, session_id: str) -> Path:
        if not _SESSION_ID.match(session_id):
            raise ValueError("非法的 session_id")
        return self.dir / f"{session_id}.json"

    def get(self, session_id: str | None, user_id: str = "guest") -> SessionMemory:
        """按 (session_id, user_id) 取会话；身份不匹配时换一个新会话。

        否则切换身份后复用同一个 session，摘要里可能带出上一位用户才能看的内容。
        """
        sid = session_id or self.new_id()
        if sid in self._cache:
            cached = self._cache[sid]
            if cached.user_id != user_id:
                sid = self.new_id()
            else:
                return cached

        memory = SessionMemory.load(sid, self.path_for(sid), user_id=user_id)
        if memory.user_id != user_id:
            memory = SessionMemory(sid, path=self.path_for(sid), user_id=user_id)
        self._cache[sid] = memory
        return memory

    def list_sessions(self, user_id: str | None = None, limit: int = 20) -> list[dict]:
        """按身份列出会话。

        只返回该用户的会话：会话预览里可能包含订单号、客户姓名等敏感信息，
        不做过滤等于把别人的对话标题暴露给所有人。
        """
        items = []
        for path in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            owner = data.get("user_id", "guest")
            if user_id is not None and owner != user_id:
                continue
            turns = data.get("turns", [])
            items.append(
                {
                    "session_id": data.get("session_id", path.stem),
                    "user_id": owner,
                    "updated_at": data.get("updated_at"),
                    "messages": len(turns),
                    "preview": (turns[-1]["content"][:40] if turns else ""),
                    "summary": data.get("summary", "")[:60],
                }
            )
            if len(items) >= limit:
                break
        return items

    def load_for_user(self, session_id: str, user_id: str) -> SessionMemory | None:
        """按 id + 身份加载会话；不存在或不属于该身份都返回 None（不区分，避免探测）。"""
        path = self.path_for(session_id)  # 非法 id 会抛 ValueError
        if not path.exists():
            return None
        memory = SessionMemory.load(session_id, path, user_id=user_id)
        if memory.user_id != user_id:
            return None
        self._cache[session_id] = memory
        return memory

    def delete(self, session_id: str, user_id: str) -> bool:
        path = self.path_for(session_id)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if data.get("user_id", "guest") != user_id:
            return False
        path.unlink()
        self._cache.pop(session_id, None)
        return True
