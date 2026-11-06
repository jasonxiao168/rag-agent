"""模拟业务数据源：订单库 + 工单库。

真实项目里这里换成 SQL / 内部 API；对 Agent 来说接口形状一致即可：
入参是订单号，出参是结构化状态 + 一段人类可读的描述。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.config import settings

ORDERS: dict[str, dict] = {
    "ORD-2026-1001": {
        "order_id": "ORD-2026-1001",
        "customer": "王先生",
        "sku": "冰感控油洁面乳 120g × 2",
        "amount": 178.0,
        "status": "shipped",
        "status_text": "已发货",
        "paid_at": "2026-11-03 20:14",
        "shipped_at": "2026-11-04 09:30",
        "carrier": "顺丰速运",
        "tracking_no": "SF1234567890",
        "eta": "2026-11-08",
        "note": "包裹已到达上海转运中心，预计 11 月 8 日前送达。",
    },
    "ORD-2026-1002": {
        "order_id": "ORD-2026-1002",
        "customer": "李女士",
        "sku": "烟酰胺焕亮精华 30ml × 1",
        "amount": 129.0,
        "status": "pending_shipment",
        "status_text": "待发货",
        "paid_at": "2026-11-05 11:02",
        "shipped_at": None,
        "carrier": None,
        "tracking_no": None,
        "eta": "2026-11-07",
        "note": "仓库已接单，预计 11 月 7 日出库，出库后 48 小时内可查物流。",
    },
    "ORD-2026-1003": {
        "order_id": "ORD-2026-1003",
        "customer": "张先生",
        "sku": "男士素颜霜 SPF30 × 1",
        "amount": 99.0,
        "status": "delivered",
        "status_text": "已签收",
        "paid_at": "2026-10-29 16:40",
        "shipped_at": "2026-10-30 10:00",
        "carrier": "京东物流",
        "tracking_no": "JDVA00012345678",
        "eta": "2026-11-02",
        "note": "2026-11-02 14:20 由本人签收。七天无理由退货截止日为 2026-11-09。",
    },
    "ORD-2026-1004": {
        "order_id": "ORD-2026-1004",
        "customer": "赵女士",
        "sku": "须后舒缓水 100ml × 2",
        "amount": 138.0,
        "status": "refunding",
        "status_text": "退款中",
        "paid_at": "2026-11-01 09:15",
        "shipped_at": "2026-11-01 18:00",
        "carrier": "中通快递",
        "tracking_no": "ZT7788990011",
        "eta": None,
        "note": "退货已签收入库，退款审核通过，预计 1 至 3 个工作日原路退回。",
    },
    "ORD-2026-1005": {
        "order_id": "ORD-2026-1005",
        "customer": "陈先生",
        "sku": "控油净透面膜 5 片 × 3",
        "amount": 177.0,
        "status": "out_of_stock",
        "status_text": "缺货待补",
        "paid_at": "2026-11-04 21:08",
        "shipped_at": None,
        "carrier": None,
        "tracking_no": None,
        "eta": "2026-11-12",
        "note": "该批次面膜质检未通过已退回工厂，预计 11 月 12 日补货后优先发出。可选择等待或申请全额退款。",
    },
    "ORD-2026-1006": {
        "order_id": "ORD-2026-1006",
        "customer": "周先生",
        "sku": "氨基酸温和洁面 100g × 1",
        "amount": 79.0,
        "status": "cancelled",
        "status_text": "已取消",
        "paid_at": "2026-10-28 08:30",
        "shipped_at": None,
        "carrier": None,
        "tracking_no": None,
        "eta": None,
        "note": "用户于 2026-10-28 09:00 主动取消，款项已于当日原路退回。",
    },
}


def query_user_order(order_id: str) -> dict:
    """返回订单结构化数据；查不到时返回 ok=False，由 Agent 如实告知用户。"""
    key = (order_id or "").strip().upper().replace("_", "-")
    order = ORDERS.get(key)
    if not order:
        return {
            "ok": False,
            "order_id": key,
            "message": f"未查询到订单 {key or '（空）'}，请核对订单号后重试。",
        }
    return {"ok": True, **order}


def format_order(order: dict) -> str:
    if not order.get("ok"):
        return order["message"]
    parts = [
        f"订单号：{order['order_id']}",
        f"商品：{order['sku']}",
        f"金额：¥{order['amount']:.2f}",
        f"状态：{order['status_text']}",
        f"下单时间：{order['paid_at']}",
    ]
    if order.get("tracking_no"):
        parts.append(f"物流：{order['carrier']} {order['tracking_no']}")
    if order.get("eta"):
        parts.append(f"预计送达/完成：{order['eta']}")
    parts.append(f"说明：{order['note']}")
    return "；".join(parts)


def _ticket_file() -> Path:
    return settings.data_dir / "tickets.json"


def _load_tickets() -> list[dict]:
    path = _ticket_file()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def create_ticket(order_id: str, reason: str, priority: str = "normal") -> dict:
    """创建售后工单（写本地 JSON，演示「Agent 能做事」而不是只会回答）。"""
    order = query_user_order(order_id)
    ticket = {
        "ticket_id": f"TK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "order_id": (order_id or "").strip().upper(),
        "reason": reason,
        "priority": priority if priority in {"low", "normal", "high"} else "normal",
        "order_found": order.get("ok", False),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "open",
    }
    tickets = _load_tickets()
    tickets.append(ticket)
    _ticket_file().parent.mkdir(parents=True, exist_ok=True)
    _ticket_file().write_text(json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "ticket": ticket,
        "message": (
            f"已创建工单 {ticket['ticket_id']}（优先级 {ticket['priority']}），"
            f"客服将在 24 小时内联系用户。"
        ),
    }
