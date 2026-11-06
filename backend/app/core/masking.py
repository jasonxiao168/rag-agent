"""PII 脱敏。

原则：脱敏发生在「上下文进入 Prompt 之前」，而不是在回答生成之后。
这样模型根本看不到明文，也就不可能把它复述出来；引用与 trace 里同样只有掩码。

支持手机号、身份证、银行卡、邮箱四类。返回替换次数，便于在链路里展示「本次脱敏 N 处」。
"""
from __future__ import annotations

import re

# 边界都用 (?<!\d)/(?!\d)，避免在身份证号里误命中 11 位手机号
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
_ID_CARD = re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3}[\dXx])(?!\d)")
# 银行卡：13-19 位，允许空格分组；首位 3-6 避免误伤订单号/时间戳
_BANK_CARD = re.compile(r"(?<!\d)([3-6]\d{3})(\s?\d{4}){2,3}(\s?\d{1,4})?(?!\d)")
_EMAIL = re.compile(r"([A-Za-z0-9._%+-]{1,3})[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def mask_phone(match: re.Match) -> str:
    return f"{match.group(1)}****{match.group(3)}"


def mask_id_card(match: re.Match) -> str:
    # 保留前 6 位（地区码）与后 4 位
    return f"{match.group(1)}********{match.group(3)[-4:]}"


def mask_bank_card(match: re.Match) -> str:
    digits = re.sub(r"\s", "", match.group(0))
    return f"**** **** **** {digits[-4:]}"


def mask_email(match: re.Match) -> str:
    return f"{match.group(1)}***{match.group(2)}"


def mask_text(text: str) -> tuple[str, int]:
    """返回 (脱敏后的文本, 替换处数)。顺序很重要：先身份证，再手机号，最后银行卡。"""
    if not text:
        return text, 0

    total = 0
    for pattern, repl in (
        (_ID_CARD, mask_id_card),
        (_PHONE, mask_phone),
        (_BANK_CARD, mask_bank_card),
        (_EMAIL, mask_email),
    ):
        text, count = pattern.subn(repl, text)
        total += count
    return text, total


def contains_pii(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (_PHONE, _ID_CARD, _BANK_CARD, _EMAIL)
    )
