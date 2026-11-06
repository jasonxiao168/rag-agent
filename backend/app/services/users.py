"""演示用身份目录。

生产环境应接入 SSO / LDAP，这里用固定用户模拟不同部门的访问权限。
- `departments`：决定能检索到哪些文档（文档在 frontmatter 里声明 acl）；
- `permissions`：额外的字段级权限，`pii` 表示可以查看未脱敏的手机号/身份证。
"""
from __future__ import annotations

DEMO_USERS: dict[str, dict] = {
    "guest": {
        "id": "guest",
        "name": "访客",
        "department_label": "仅公共文档",
        "departments": ["public"],
        "permissions": [],
    },
    "alice": {
        "id": "alice",
        "name": "艾丽",
        "department_label": "人力资源",
        "departments": ["hr"],
        "permissions": [],
    },
    "bob": {
        "id": "bob",
        "name": "博文",
        "department_label": "财务",
        "departments": ["finance"],
        "permissions": [],
    },
    "carol": {
        "id": "carol",
        "name": "卡罗",
        "department_label": "渠道运营",
        "departments": ["channel"],
        "permissions": [],
    },
    "dave": {
        "id": "dave",
        "name": "大卫",
        "department_label": "市场",
        "departments": ["marketing"],
        "permissions": [],
    },
    "erin": {
        "id": "erin",
        "name": "艾琳",
        "department_label": "客服主管",
        "departments": ["service"],
        "permissions": ["pii"],
    },
}

DEFAULT_USER = "guest"


def resolve_user(user_id: str | None) -> dict:
    """未知用户一律按访客处理，避免「传错 id 反而拿到更多权限」。"""
    return DEMO_USERS.get(user_id or DEFAULT_USER, DEMO_USERS[DEFAULT_USER])


def list_users() -> list[dict]:
    return list(DEMO_USERS.values())
