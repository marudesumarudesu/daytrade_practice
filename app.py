from __future__ import annotations

import time
from datetime import datetime, timedelta
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "Daytrade Desk Trainer"
SESSION_SECONDS = 90 * 60
SESSION_START = datetime(2026, 4, 8, 9, 0, 0)
BOARD_LEVELS = 10
NIKKEI_LABEL = "日経平均先物"

SYMBOLS: dict[str, dict[str, float | int | str | list[tuple[int, int, float]]]] = {
    "7011.T": {
        "label": "7011.M 二菱重工",
        "start_price": 2428,
        "beta": 0.86,
        "correlation": 0.82,
        "noise": 0.34,
        "wave_amp": 0.000028,
        "wave_period": 48,
        "swing_amp": 0.000045,
        "swing_period": 220,
        "phase": 0.35,
        "bias": 0.0000015,
        "volume_base": 1800,
        "depth_base": 4800,
        "seed_offset": 11,
        "shocks": [(520, 45, 0.00016), (1680, 95, -0.00013), (3180, 60, 0.00018)],
    },
    "5803.T": {
        "label": "5803.M プジクラ",
        "start_price": 2865,
        "beta": 1.02,
        "correlation": 0.78,
        "noise": 0.4,
        "wave_amp": 0.000031,
        "wave_period": 40,
        "swing_amp": 0.000055,
        "swing_period": 180,
        "phase": 1.2,
        "bias": -0.000001,
        "volume_base": 2200,
        "depth_base": 5300,
        "seed_offset": 29,
        "shocks": [(860, 55, 0.0002), (2410, 75, -0.00018), (3890, 90, 0.00015)],
    },
    "8035.T": {
        "label": "8035.M 東京エレクトーン",
        "start_price": 31800,
        "beta": 1.15,
        "correlation": 0.74,
        "noise": 0.44,
        "wave_amp": 0.00003,
        "wave_period": 36,
        "swing_amp": 0.00006,
        "swing_period": 160,
        "phase": 2.1,
        "bias": 0.0000008,
        "volume_base": 700,
        "depth_base": 1800,
        "seed_offset": 47,
        "shocks": [(640, 50, 0.00022), (2140, 80, -0.00016), (3540, 70, 0.00021)],
    },
    "9984.T": {
        "label": "9984.M ソフトパンクG",
        "start_price": 8250,
        "beta": 0.93,
        "correlation": 0.8,
        "noise": 0.37,
        "wave_amp": 0.000026,
        "wave_period": 44,
        "swing_amp": 0.00005,
        "swing_period": 190,
        "phase": 0.8,
        "bias": 0.000001,
        "volume_base": 1300,
        "depth_base": 3600,
        "seed_offset": 63,
        "shocks": [(980, 60, 0.00014), (2840, 90, -0.00019), (4210, 80, 0.00012)],
    },
}


def tick_size(price: float) -> int:
    if price < 3000:
        return 1
    if price < 5000:
        return 5
    if price < 30000:
        return 10
    if price < 50000:
        return 50
    return 100


def round_to_tick(price: float, tick: int) -> int:
    return max(tick, int(round(price / tick) * tick))


def format_price(value: float | int) -> str:
    return f"{int(round(value)):,}"


def format_signed(value: float | int) -> str:
    number = float(value)
    if number > 0:
        return f"+{number:,.0f}"
    if number < 0:
        return f"{number:,.0f}"
    return "0"


def gaussian_pulse(index: np.ndarray, center: int, width: int, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((index - center) / width) ** 2)


def html_fragment(content: str) -> str:
    return dedent(content).strip()


@st.cache_data(show_spinner=False)
def build_market(seed: int = 21) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
    nikkei = build_nikkei_path(seed)
    symbol_paths: dict[str, pd.DataFrame] = {}
    for offset, (code, meta) in enumerate(SYMBOLS.items(), start=1):
        symbol_paths[code] = build_symbol_path(code, meta, nikkei, seed + (offset * 37))
    return {"nikkei": nikkei, "symbols": symbol_paths}


def build_nikkei_path(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seconds = np.arange(SESSION_SECONDS)
    timestamps = pd.date_range(SESSION_START, periods=SESSION_SECONDS, freq="s")

    regime = np.select(
        [
            seconds < 900,
            (seconds >= 900) & (seconds < 2100),
            (seconds >= 2100) & (seconds < 3600),
            seconds >= 3600,
        ],
        [0.0000035, -0.0000015, 0.000003, -0.0000008],
        default=0.0,
    )
    wave = (
        0.000055 * np.sin(seconds / 84)
        + 0.000032 * np.sin(seconds / 260 + 0.6)
        + 0.000018 * np.cos(seconds / 35)
    )
    shocks = (
        gaussian_pulse(seconds, 420, 42, 0.0002)
        + gaussian_pulse(seconds, 1440, 100, -0.00016)
        + gaussian_pulse(seconds, 2760, 70, 0.00018)
        + gaussian_pulse(seconds, 4320, 85, -0.00014)
    )
    noise = rng.normal(0.0, 0.000014, SESSION_SECONDS)
    smooth_noise = pd.Series(noise).ewm(alpha=0.16).mean().to_numpy()
    returns = regime + wave + shocks + smooth_noise

    fair = np.empty(SESSION_SECONDS)
    fair[0] = 38640
    for idx in range(1, SESSION_SECONDS):
        fair[idx] = fair[idx - 1] * (1 + returns[idx])

    last = (np.round(fair / 10) * 10).astype(int)
    gradient = np.abs(np.gradient(last))
    size = (80 + gradient * 4 + rng.integers(0, 70, SESSION_SECONDS)).astype(int)

    df = pd.DataFrame(
        {
            "step": seconds,
            "timestamp": timestamps,
            "last": last,
            "size": size,
        }
    )
    df["return"] = df["last"].pct_change().fillna(0.0)
    turnover = (df["last"] * df["size"]).cumsum()
    volume = df["size"].cumsum().replace(0, 1)
    df["vwap"] = turnover / volume
    return df


def build_symbol_path(
    code: str,
    meta: dict[str, float | int | str | list[tuple[int, int, float]]],
    nikkei: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seconds = np.arange(SESSION_SECONDS)
    timestamps = pd.date_range(SESSION_START, periods=SESSION_SECONDS, freq="s")
    nikkei_returns = nikkei["return"].to_numpy()

    ar_component = np.zeros(SESSION_SECONDS)
    for idx in range(1, SESSION_SECONDS):
        ar_component[idx] = 0.86 * ar_component[idx - 1] + rng.normal(0.0, float(meta["noise"]))

    micro_wave = float(meta["wave_amp"]) * np.sin(seconds / float(meta["wave_period"]) + float(meta["phase"]))
    swing_wave = float(meta["swing_amp"]) * np.sin(
        seconds / float(meta["swing_period"]) + float(meta["phase"]) * 0.4
    )
    shocks = np.zeros(SESSION_SECONDS)
    for center, width, amplitude in meta["shocks"]:
        shocks += gaussian_pulse(seconds, center, width, amplitude)

    returns = (
        float(meta["beta"]) * float(meta["correlation"]) * nikkei_returns
        + micro_wave
        + swing_wave
        + shocks
        + ar_component * 0.00011
        + float(meta["bias"])
    )

    fair = np.empty(SESSION_SECONDS)
    last = np.empty(SESSION_SECONDS, dtype=int)
    best_bid = np.empty(SESSION_SECONDS, dtype=int)
    best_ask = np.empty(SESSION_SECONDS, dtype=int)
    spread = np.empty(SESSION_SECONDS, dtype=int)
    size = np.empty(SESSION_SECONDS, dtype=int)
    tick = np.empty(SESSION_SECONDS, dtype=int)
    pressure = np.empty(SESSION_SECONDS)
    side: list[str] = []

    start_price = float(meta["start_price"])
    floor_price = start_price * 0.65
    fair[0] = start_price
    initial_tick = tick_size(start_price)
    last[0] = round_to_tick(start_price, initial_tick)
    best_bid[0] = last[0] - initial_tick
    best_ask[0] = last[0]
    spread[0] = initial_tick
    size[0] = int(meta["volume_base"])
    tick[0] = initial_tick
    pressure[0] = 0.18
    side.append("BUY")

    for idx in range(1, SESSION_SECONDS):
        fair[idx] = max(floor_price, fair[idx - 1] * (1 + returns[idx]))
        current_tick = tick_size(fair[idx])
        desired = round_to_tick(fair[idx], current_tick)
        delta_ticks = int(np.clip(round((desired - last[idx - 1]) / current_tick), -3, 3))
        bias = float(meta["beta"]) * nikkei_returns[idx] * 9000 + ar_component[idx] * 1.6 + rng.normal(0.0, 0.28)

        if delta_ticks == 0 and abs(bias) > 0.46:
            delta_ticks = 1 if bias > 0 else -1
        elif delta_ticks == 0 and rng.random() < 0.28:
            delta_ticks = 1 if bias >= 0 else -1

        trade_price = round_to_tick(last[idx - 1] + delta_ticks * current_tick, current_tick)
        trade_price = max(current_tick, trade_price)
        trade_side = "BUY" if (delta_ticks > 0 or (delta_ticks == 0 and bias >= 0)) else "SELL"
        spread_ticks = 1 + int(abs(delta_ticks) >= 2 or abs(nikkei_returns[idx]) > 0.0002)
        spread_value = spread_ticks * current_tick

        if trade_side == "BUY":
            best_ask[idx] = trade_price
            best_bid[idx] = max(current_tick, trade_price - spread_value)
        else:
            best_bid[idx] = trade_price
            best_ask[idx] = trade_price + spread_value

        last[idx] = trade_price
        spread[idx] = spread_value
        tick[idx] = current_tick
        pressure[idx] = float(
            np.clip(
                0.62 * np.sign(delta_ticks if delta_ticks != 0 else (1 if trade_side == "BUY" else -1))
                + 0.38 * np.tanh(bias),
                -1.0,
                1.0,
            )
        )
        raw_size = int(
            float(meta["volume_base"])
            * (1 + (0.34 * abs(delta_ticks)) + rng.uniform(-0.16, 0.24))
        )
        size[idx] = max(100, (raw_size // 100) * 100)
        side.append(trade_side)

    df = pd.DataFrame(
        {
            "symbol": code,
            "step": seconds,
            "timestamp": timestamps,
            "last": last,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "tick": tick,
            "size": size,
            "side": side,
            "pressure": pressure,
        }
    )
    df["return"] = df["last"].pct_change().fillna(0.0)
    turnover = (df["last"] * df["size"]).cumsum()
    volume = df["size"].cumsum().replace(0, 1)
    df["vwap"] = turnover / volume
    return df


def blank_position() -> dict[str, float | int]:
    return {"qty": 0, "avg_price": 0.0, "realized": 0.0}


def ensure_state() -> None:
    if "market" not in st.session_state:
        st.session_state.market = build_market()
    if "clock" not in st.session_state:
        st.session_state.clock = {"running": True, "anchor": time.time(), "elapsed_before": 0}
    if "positions" not in st.session_state:
        st.session_state.positions = {symbol: blank_position() for symbol in SYMBOLS}
    if "fills" not in st.session_state:
        st.session_state.fills = []
    if "orders" not in st.session_state:
        st.session_state.orders = []
    if "last_processed_step" not in st.session_state:
        st.session_state.last_processed_step = -1
    if "order_seq" not in st.session_state:
        st.session_state.order_seq = 1
    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = next(iter(SYMBOLS))
    if "order_qty" not in st.session_state:
        st.session_state.order_qty = 100
    if "order_type" not in st.session_state:
        st.session_state.order_type = "成行"
    if "limit_price" not in st.session_state:
        st.session_state.limit_price = int(SYMBOLS[st.session_state.selected_symbol]["start_price"])
    if "order_symbol_bound" not in st.session_state:
        st.session_state.order_symbol_bound = st.session_state.selected_symbol
    if "notice" not in st.session_state:
        st.session_state.notice = "板は1秒ごとに更新されます。表示される価格と出来高はすべて学習用のシミュレーションです。"


def current_step() -> int:
    clock = st.session_state.clock
    elapsed = int(time.time() - clock["anchor"]) + clock["elapsed_before"] if clock["running"] else clock["elapsed_before"]
    return min(elapsed, SESSION_SECONDS - 1)


def pause_clock() -> None:
    if st.session_state.clock["running"]:
        st.session_state.clock["elapsed_before"] = current_step()
        st.session_state.clock["running"] = False


def resume_clock() -> None:
    if not st.session_state.clock["running"] and st.session_state.clock["elapsed_before"] < SESSION_SECONDS - 1:
        st.session_state.clock["anchor"] = time.time()
        st.session_state.clock["running"] = True


def reset_session() -> None:
    st.session_state.clock = {"running": True, "anchor": time.time(), "elapsed_before": 0}
    st.session_state.positions = {symbol: blank_position() for symbol in SYMBOLS}
    st.session_state.fills = []
    st.session_state.orders = []
    st.session_state.last_processed_step = -1
    st.session_state.order_seq = 1
    st.session_state.notice = "シミュレーションをリセットしました。"


def position_label(qty: int) -> str:
    if qty > 0:
        return "買い建"
    if qty < 0:
        return "売り建"
    return "フラット"


def unrealized_pnl(position: dict[str, float | int], last_price: int) -> float:
    qty = int(position["qty"])
    avg = float(position["avg_price"])
    if qty > 0:
        return qty * (last_price - avg)
    if qty < 0:
        return abs(qty) * (avg - last_price)
    return 0.0


def order_action_label(action: str) -> str:
    labels = {
        "OPEN_BUY": "新規買い",
        "OPEN_SELL": "新規売り",
        "CLOSE_BUY": "返済買い",
        "CLOSE_SELL": "返済売り",
    }
    return labels[action]


def order_side_for_action(action: str) -> str:
    return "BUY" if action in {"OPEN_BUY", "CLOSE_BUY"} else "SELL"


def order_action_from_record(order: dict[str, str | int]) -> str:
    action = order.get("action")
    if isinstance(action, str) and action:
        return action
    return "OPEN_BUY" if order.get("side") == "BUY" else "OPEN_SELL"


def validate_order_action(symbol: str, action: str, qty: int) -> tuple[bool, str]:
    position_qty = int(st.session_state.positions[symbol]["qty"])
    label = SYMBOLS[symbol]["label"]

    if qty <= 0:
        return False, "株数は1以上で指定してください"

    if action == "OPEN_BUY":
        if position_qty < 0:
            return False, f"{label} は売り建玉があるため新規買いできません"
        return True, ""
    if action == "OPEN_SELL":
        if position_qty > 0:
            return False, f"{label} は買い建玉があるため新規売りできません"
        return True, ""
    if action == "CLOSE_BUY":
        if position_qty >= 0:
            return False, f"{label} は返済買いできる売り建玉がありません"
        if qty > abs(position_qty):
            return False, f"{label} の返済買い数量は売り建玉 {abs(position_qty):,} 株以下で指定してください"
        return True, ""
    if action == "CLOSE_SELL":
        if position_qty <= 0:
            return False, f"{label} は返済売りできる買い建玉がありません"
        if qty > position_qty:
            return False, f"{label} の返済売り数量は買い建玉 {position_qty:,} 株以下で指定してください"
        return True, ""
    return False, "未対応の注文アクションです"


def session_pnl(step: int) -> dict[str, float]:
    total_realized = sum(float(position["realized"]) for position in st.session_state.positions.values())
    total_unrealized = 0.0
    for symbol, position in st.session_state.positions.items():
        last_price = int(st.session_state.market["symbols"][symbol].iloc[step]["last"])
        total_unrealized += unrealized_pnl(position, last_price)
    return {
        "realized": total_realized,
        "unrealized": total_unrealized,
        "total": total_realized + total_unrealized,
    }


def apply_fill(symbol: str, side: str, qty: int, price: int, step: int, source: str, action: str | None = None) -> str:
    position = st.session_state.positions[symbol]
    current_qty = int(position["qty"])
    avg_price = float(position["avg_price"])
    realized = float(position["realized"])
    signed_qty = qty if side == "BUY" else -qty

    if current_qty == 0 or (current_qty > 0 and signed_qty > 0) or (current_qty < 0 and signed_qty < 0):
        new_qty = current_qty + signed_qty
        total_cost = abs(current_qty) * avg_price + qty * price
        position["qty"] = new_qty
        position["avg_price"] = total_cost / abs(new_qty) if new_qty else 0.0
    else:
        closing_qty = min(abs(current_qty), qty)
        if current_qty > 0:
            realized += closing_qty * (price - avg_price)
        else:
            realized += closing_qty * (avg_price - price)
        new_qty = current_qty + signed_qty
        position["qty"] = new_qty
        if new_qty == 0:
            position["avg_price"] = 0.0
        elif (current_qty > 0 and new_qty < 0) or (current_qty < 0 and new_qty > 0):
            position["avg_price"] = float(price)
        else:
            position["avg_price"] = avg_price

    position["realized"] = realized
    fill_record = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "source": source,
        "action": action,
        "timestamp": SESSION_START + timedelta(seconds=step),
    }
    st.session_state.fills.insert(0, fill_record)
    action_label = order_action_label(action) if action else ("買い" if side == "BUY" else "売り")
    source_label = "成行" if source == "MARKET" else "指値"
    return f"{SYMBOLS[symbol]['label']} {action_label} {qty:,}株 @{format_price(price)} ({source_label})"


def execute_market_order(symbol: str, action: str, qty: int, step: int) -> str:
    is_valid, message = validate_order_action(symbol, action, qty)
    if not is_valid:
        return message
    snapshot = st.session_state.market["symbols"][symbol].iloc[step]
    side = order_side_for_action(action)
    fill_price = int(snapshot["best_ask"] if side == "BUY" else snapshot["best_bid"])
    return apply_fill(symbol, side, qty, fill_price, step, "MARKET", action)


def submit_limit_order(symbol: str, action: str, qty: int, limit_price: int, step: int) -> str:
    is_valid, message = validate_order_action(symbol, action, qty)
    if not is_valid:
        return message
    snapshot = st.session_state.market["symbols"][symbol].iloc[step]
    limit_price = round_to_tick(limit_price, int(snapshot["tick"]))
    side = order_side_for_action(action)

    immediately_fillable = (side == "BUY" and limit_price >= int(snapshot["best_ask"])) or (
        side == "SELL" and limit_price <= int(snapshot["best_bid"])
    )
    if immediately_fillable:
        fill_price = int(snapshot["best_ask"] if side == "BUY" else snapshot["best_bid"])
        return apply_fill(symbol, side, qty, fill_price, step, "LIMIT", action)

    order_id = f"ORD-{st.session_state.order_seq:04d}"
    st.session_state.order_seq += 1
    st.session_state.orders.append(
        {
            "id": order_id,
            "symbol": symbol,
            "action": action,
            "side": side,
            "qty": qty,
            "limit_price": limit_price,
            "created_step": step,
        }
    )
    return f"{SYMBOLS[symbol]['label']} {order_action_label(action)}指値 {qty:,}株 @{format_price(limit_price)} を受付"


def process_limit_orders_until(step: int) -> None:
    last_processed = st.session_state.last_processed_step
    if step <= last_processed:
        return

    notices: list[str] = []
    open_orders = st.session_state.orders
    market = st.session_state.market["symbols"]

    for current in range(last_processed + 1, step + 1):
        still_open: list[dict[str, str | int]] = []
        for order in open_orders:
            if int(order["created_step"]) > current:
                still_open.append(order)
                continue

            symbol = str(order["symbol"])
            action = order_action_from_record(order)
            is_valid, message = validate_order_action(symbol, action, int(order["qty"]))
            if not is_valid:
                notices.append(f"{order_action_label(action)}注文を取消: {message}")
                continue

            snapshot = market[symbol].iloc[current]
            if order["side"] == "BUY" and int(order["limit_price"]) >= int(snapshot["best_ask"]):
                notices.append(
                    apply_fill(
                        symbol,
                        "BUY",
                        int(order["qty"]),
                        int(snapshot["best_ask"]),
                        current,
                        "LIMIT",
                        action,
                    )
                )
            elif order["side"] == "SELL" and int(order["limit_price"]) <= int(snapshot["best_bid"]):
                notices.append(
                    apply_fill(
                        symbol,
                        "SELL",
                        int(order["qty"]),
                        int(snapshot["best_bid"]),
                        current,
                        "LIMIT",
                        action,
                    )
                )
            else:
                still_open.append(order)
        open_orders = still_open

    st.session_state.orders = open_orders
    st.session_state.last_processed_step = step
    if notices:
        st.session_state.notice = notices[-1]


def cancel_symbol_orders(symbol: str) -> str:
    before = len(st.session_state.orders)
    st.session_state.orders = [order for order in st.session_state.orders if order["symbol"] != symbol]
    canceled = before - len(st.session_state.orders)
    if canceled == 0:
        return f"{SYMBOLS[symbol]['label']} の待機中注文はありません"
    return f"{SYMBOLS[symbol]['label']} の待機中注文を {canceled} 件取消しました"


def build_board_snapshot(snapshot: pd.Series, meta: dict[str, float | int | str | list[tuple[int, int, float]]]) -> pd.DataFrame:
    rng = np.random.default_rng(int(snapshot["step"]) + int(meta["seed_offset"]) * 100)
    pressure = float(snapshot["pressure"])
    tick = int(snapshot["tick"])
    depth_base = int(meta["depth_base"])
    rows: list[dict[str, int | bool]] = []

    for distance in range(BOARD_LEVELS - 1, -1, -1):
        price = int(snapshot["best_ask"] + (distance * tick))
        volume = max(
            100,
            int(depth_base * (0.75**distance) * (1 - 0.24 * pressure) * rng.uniform(0.85, 1.15)),
        )
        rows.append(
            {
                "price": price,
                "ask_size": (volume // 100) * 100,
                "bid_size": 0,
                "is_best_ask": price == int(snapshot["best_ask"]),
                "is_best_bid": False,
                "is_last": price == int(snapshot["last"]),
            }
        )

    for distance in range(0, BOARD_LEVELS):
        price = int(snapshot["best_bid"] - (distance * tick))
        volume = max(
            100,
            int(depth_base * (0.75**distance) * (1 + 0.24 * pressure) * rng.uniform(0.85, 1.15)),
        )
        rows.append(
            {
                "price": price,
                "ask_size": 0,
                "bid_size": (volume // 100) * 100,
                "is_best_ask": False,
                "is_best_bid": price == int(snapshot["best_bid"]),
                "is_last": price == int(snapshot["last"]),
            }
        )

    return pd.DataFrame(rows)


def render_board_html(board: pd.DataFrame) -> str:
    max_size = max(int(board["ask_size"].max()), int(board["bid_size"].max()), 1)
    rows: list[str] = []
    for row in board.itertuples():
        ask_pct = (row.ask_size / max_size) * 100 if row.ask_size else 0
        bid_pct = (row.bid_size / max_size) * 100 if row.bid_size else 0
        price_class = "price-cell"
        if row.is_best_ask:
            price_class += " best-ask"
        if row.is_best_bid:
            price_class += " best-bid"
        if row.is_last:
            price_class += " last-price"
        rows.append(
            html_fragment(
                f"""
                <tr>
                    <td class="qty-cell ask-side">
                        <div class="depth-track">
                            <div class="depth-fill ask-fill" style="width:{ask_pct:.1f}%"></div>
                            <span>{format_price(row.ask_size) if row.ask_size else ""}</span>
                        </div>
                    </td>
                    <td class="{price_class}">{format_price(row.price)}</td>
                    <td class="qty-cell bid-side">
                        <div class="depth-track">
                            <div class="depth-fill bid-fill" style="width:{bid_pct:.1f}%"></div>
                            <span>{format_price(row.bid_size) if row.bid_size else ""}</span>
                        </div>
                    </td>
                </tr>
                """
            )
        )

    return html_fragment(
        f"""
        <div class="panel-shell board-shell">
            <table class="market-table">
                <thead>
                    <tr>
                        <th>売数量</th>
                        <th>値段</th>
                        <th>買数量</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
    )


def render_tape_html(trades: pd.DataFrame) -> str:
    if trades.empty:
        return html_fragment(
            """
            <div class="panel-shell tape-shell empty-state">
                約定データの生成待ちです。
            </div>
            """
        )

    rows: list[str] = []
    recent = trades.sort_values("step", ascending=False).head(18)
    for row in recent.itertuples():
        rows.append(
            html_fragment(
                f"""
                <tr>
                    <td>{row.timestamp.strftime('%H:%M:%S')}</td>
                    <td>{format_price(row.last)}</td>
                    <td>{format_price(row.size)}</td>
                </tr>
                """
            )
        )

    return html_fragment(
        f"""
        <div class="panel-shell tape-shell">
            <table class="market-table tape-table">
                <thead>
                    <tr>
                        <th>時刻</th>
                        <th>約定値</th>
                        <th>株数</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
    )


def render_orders_html(orders: list[dict[str, str | int]]) -> str:
    if not orders:
        return html_fragment(
            """
            <div class="subpanel-shell empty-state">
                待機中の指値注文はありません。
            </div>
            """
        )

    rows: list[str] = []
    for order in orders[:4]:
        rows.append(
            html_fragment(
                f"""
                <tr>
                    <td>{order_action_label(order_action_from_record(order))}</td>
                    <td>{format_price(order["qty"])}</td>
                    <td>{format_price(order["limit_price"])}</td>
                </tr>
                """
            )
        )
    return html_fragment(
        f"""
        <div class="subpanel-shell">
            <table class="mini-table">
                <thead>
                    <tr>
                        <th>注文</th>
                        <th>株数</th>
                        <th>指値</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
    )


def render_status_html(
    symbol: str,
    snapshot: pd.Series,
    position: dict[str, float | int],
    open_orders: list[dict[str, str | int]],
    pnl_summary: dict[str, float],
) -> str:
    qty = int(position["qty"])
    avg_price = float(position["avg_price"])
    realized = float(position["realized"])
    vwap_gap = int(snapshot["last"]) - float(snapshot["vwap"])
    unrealized = unrealized_pnl(position, int(snapshot["last"]))
    direction = position_label(qty)
    last_fill = next((fill for fill in st.session_state.fills if fill["symbol"] == symbol), None)
    last_fill_text = (
        f"{order_action_label(str(last_fill['action'])) if last_fill.get('action') else ('買い' if last_fill['side'] == 'BUY' else '売り')} "
        f"{format_price(last_fill['qty'])}株 @ {format_price(last_fill['price'])}"
        if last_fill
        else "未約定"
    )

    cards = [
        ("通算損益", format_signed(pnl_summary["total"]), f"実現 {format_signed(pnl_summary['realized'])} / 含み {format_signed(pnl_summary['unrealized'])}"),
        ("現在値", format_price(snapshot["last"]), f"VWAP差 {format_signed(vwap_gap)}"),
        ("VWAP", format_price(snapshot["vwap"]), f"最良売 {format_price(snapshot['best_ask'])}"),
        ("保有株数", f"{abs(qty):,}株", direction),
        ("平均単価", format_price(avg_price) if avg_price else "-", f"実現 {format_signed(realized)}"),
        ("含み損益", format_signed(unrealized), f"最終約定 {last_fill_text}"),
    ]

    card_html = "".join(
        html_fragment(
            f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-sub">{sub}</div>
            </div>
            """
        )
        for label, value, sub in cards
    )

    return html_fragment(
        f"""
        <div class="panel-shell status-shell">
            <div class="stat-grid">{card_html}</div>
            <div class="subpanel-title">待機中の注文</div>
            {render_orders_html(open_orders)}
        </div>
        """
    )


def build_candles(df: pd.DataFrame, step: int) -> pd.DataFrame:
    view = df.iloc[: step + 1].copy()
    indexed = view.set_index("timestamp")
    ohlc = indexed["last"].resample("1min").ohlc()
    volume = indexed["size"].resample("1min").sum()
    turnover = (indexed["last"] * indexed["size"]).resample("1min").sum()
    candles = ohlc.join(volume.rename("volume")).join((turnover / volume.replace(0, np.nan)).rename("vwap"))
    return candles.dropna().reset_index().tail(24)


def build_chart(candles: pd.DataFrame, title: str, show_vwap: bool = False) -> go.Figure:
    fig = go.Figure()

    if candles.empty:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(7,14,26,0.96)",
            height=260,
            margin=dict(l=8, r=8, t=28, b=6),
            title=dict(text=title, x=0.02, font=dict(size=13)),
            font=dict(color="#E8EEFF", family="IBM Plex Sans JP"),
        )
        fig.add_annotation(
            text="データを待っています",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="#A9B6D6"),
        )
        return fig

    fig.add_trace(
        go.Candlestick(
            x=candles["timestamp"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="1分足",
            increasing_line_color="#FF6B7A",
            increasing_fillcolor="#FF6B7A",
            decreasing_line_color="#35D07F",
            decreasing_fillcolor="#35D07F",
        )
    )

    if show_vwap:
        fig.add_trace(
            go.Scatter(
                x=candles["timestamp"],
                y=candles["vwap"],
                name="VWAP",
                mode="lines",
                line=dict(color="#F2C86B", width=1.8),
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,14,26,0.96)",
        height=260,
        margin=dict(l=8, r=8, t=28, b=6),
        title=dict(text=title, x=0.02, font=dict(size=13)),
        font=dict(color="#E8EEFF", family="IBM Plex Sans JP"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False), showline=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(112,133,171,0.14)", side="right", zeroline=False),
    )
    return fig


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(21, 77, 145, 0.18), transparent 32%),
                radial-gradient(circle at top left, rgba(0, 194, 203, 0.10), transparent 22%),
                linear-gradient(180deg, #060B16 0%, #07101D 52%, #040814 100%);
            color: #EDF3FF;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.4rem;
            max-width: 1500px;
        }

        h1, h2, h3, h4, p, div, span, label {
            font-family: 'IBM Plex Sans JP', sans-serif;
        }

        [data-testid="stSegmentedControl"] {
            background: rgba(9, 18, 32, 0.82);
            border: 1px solid rgba(100, 132, 184, 0.24);
            border-radius: 18px;
            padding: 0.2rem;
        }

        [data-testid="stSegmentedControl"] button {
            border-radius: 14px;
            color: #D6E1FA !important;
            background: linear-gradient(180deg, rgba(18, 28, 48, 0.98), rgba(9, 16, 31, 0.98)) !important;
            border: 1px solid rgba(95, 120, 168, 0.22) !important;
            -webkit-text-fill-color: #D6E1FA !important;
        }

        [data-testid="stSegmentedControl"] button * {
            color: #D6E1FA !important;
            -webkit-text-fill-color: #D6E1FA !important;
        }

        [data-testid="stSegmentedControl"] button:not([aria-pressed="true"]) {
            background: linear-gradient(180deg, rgba(18, 28, 48, 0.98), rgba(9, 16, 31, 0.98)) !important;
        }

        [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            background: linear-gradient(180deg, rgba(47, 85, 146, 0.95), rgba(20, 43, 80, 0.98)) !important;
            border: 1px solid rgba(143, 176, 232, 0.4) !important;
            color: #F8FBFF !important;
            -webkit-text-fill-color: #F8FBFF !important;
        }

        [data-testid="stSegmentedControl"] button[aria-pressed="true"] * {
            color: #F8FBFF !important;
            -webkit-text-fill-color: #F8FBFF !important;
        }

        [data-testid="stButton"] button {
            border-radius: 14px;
            border: 1px solid rgba(108, 133, 179, 0.28);
            background: linear-gradient(180deg, rgba(24, 39, 66, 0.96), rgba(10, 18, 30, 0.96));
            color: #EEF4FF;
            font-weight: 600;
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: rgba(7, 15, 28, 0.96);
            border-radius: 12px;
            color: #F2F6FF !important;
        }

        [data-testid="stNumberInput"] input {
            -webkit-text-fill-color: #F2F6FF !important;
            caret-color: #F2F6FF !important;
        }

        [data-testid="stSelectbox"] label p,
        [data-testid="stNumberInput"] label p {
            color: #D7E2F9 !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] * {
            color: #F2F6FF !important;
        }

        [data-baseweb="popover"] [role="listbox"] *,
        [data-baseweb="select"] [role="option"] * {
            color: #F2F6FF !important;
            background-color: #0A1323 !important;
        }

        [data-baseweb="popover"] [role="listbox"] {
            background: #0A1323 !important;
            border: 1px solid rgba(107, 132, 176, 0.28) !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(8, 14, 28, 0.62);
            border: 1px solid rgba(98, 131, 184, 0.22);
            border-radius: 18px;
            box-shadow: 0 16px 50px rgba(0, 0, 0, 0.25);
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(14, 23, 42, 0.92), rgba(8, 14, 28, 0.94));
            border: 1px solid rgba(97, 125, 176, 0.22);
            border-radius: 20px;
            padding: 1.15rem 1.2rem 1.05rem;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.25rem;
        }

        .hero-sub {
            color: #A6B6D6;
            font-size: 0.95rem;
        }

        .top-ribbon {
            display: flex;
            gap: 0.8rem;
            flex-wrap: wrap;
            margin: 0.9rem 0 0.3rem;
        }

        .ribbon-chip {
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(0, 176, 196, 0.12);
            border: 1px solid rgba(0, 176, 196, 0.24);
            color: #D7F9FB;
            font-size: 0.88rem;
        }

        .status-banner {
            margin-top: 0.65rem;
            padding: 0.75rem 0.95rem;
            border-radius: 14px;
            background: rgba(12, 23, 39, 0.92);
            border: 1px solid rgba(104, 129, 176, 0.22);
            color: #C6D4F1;
        }

        .panel-heading {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.5rem;
            margin-bottom: 0.55rem;
        }

        .panel-title {
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }

        .panel-subtitle {
            font-size: 0.8rem;
            color: #95A8CE;
        }

        .panel-shell {
            background: rgba(7, 14, 26, 0.96);
            border-radius: 14px;
            border: 1px solid rgba(100, 126, 173, 0.18);
            overflow: auto;
        }

        .board-shell {
            max-height: 660px;
        }

        .tape-shell {
            max-height: 970px;
        }

        .status-shell {
            padding: 0.75rem;
        }

        .subpanel-shell {
            margin-top: 0.45rem;
            background: rgba(10, 17, 31, 0.92);
            border: 1px solid rgba(101, 126, 171, 0.15);
            border-radius: 12px;
            padding: 0.25rem 0.4rem 0.4rem;
        }

        .subpanel-title {
            margin-top: 0.7rem;
            margin-bottom: 0.15rem;
            color: #9FB3DD;
            font-size: 0.82rem;
        }

        .market-table, .mini-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
        }

        .market-table thead th, .mini-table thead th {
            position: sticky;
            top: 0;
            background: rgba(12, 20, 36, 0.98);
            color: #96A8D1;
            text-align: right;
            padding: 0.6rem 0.7rem;
            border-bottom: 1px solid rgba(107, 128, 173, 0.18);
        }

        .market-table thead th:nth-child(2),
        .mini-table thead th:nth-child(2) {
            text-align: center;
        }

        .market-table tbody td,
        .mini-table tbody td {
            padding: 0.5rem 0.7rem;
            border-bottom: 1px solid rgba(107, 128, 173, 0.12);
            text-align: right;
        }

        .market-table tbody td:nth-child(2),
        .mini-table tbody td:nth-child(2) {
            text-align: center;
        }

        .qty-cell {
            width: 38%;
            position: relative;
        }

        .price-cell {
            width: 24%;
            font-weight: 700;
            color: #EAF0FF;
        }

        .best-ask {
            color: #FF919B;
        }

        .best-bid {
            color: #56D391;
        }

        .last-price {
            background: linear-gradient(90deg, rgba(242, 200, 107, 0.12), rgba(242, 200, 107, 0.02));
        }

        .depth-track {
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.02);
            min-height: 1.55rem;
        }

        .depth-fill {
            position: absolute;
            top: 0;
            bottom: 0;
            opacity: 0.22;
        }

        .ask-fill {
            right: 0;
            background: linear-gradient(90deg, rgba(255, 90, 118, 0.0), rgba(255, 90, 118, 0.92));
        }

        .bid-fill {
            left: 0;
            background: linear-gradient(90deg, rgba(61, 211, 125, 0.92), rgba(61, 211, 125, 0.0));
        }

        .depth-track span {
            position: relative;
            z-index: 1;
            display: block;
            padding: 0.2rem 0.1rem;
        }

        .buy-row td:last-child {
            color: #4DE08E;
            font-weight: 700;
        }

        .sell-row td:last-child {
            color: #FF8D9C;
            font-weight: 700;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
        }

        .stat-card {
            padding: 0.7rem 0.8rem;
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(13, 23, 41, 0.98), rgba(8, 15, 28, 0.98));
            border: 1px solid rgba(103, 130, 178, 0.18);
        }

        .stat-label {
            font-size: 0.78rem;
            color: #96A9D2;
            margin-bottom: 0.35rem;
        }

        .stat-value {
            font-size: 1.12rem;
            font-weight: 700;
            color: #F1F5FF;
            line-height: 1.15;
        }

        .stat-sub {
            margin-top: 0.28rem;
            font-size: 0.78rem;
            color: #B6C6E8;
        }

        .empty-state {
            color: #9AAED6;
            padding: 1rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def panel_header(title: str, subtitle: str) -> str:
    return html_fragment(
        f"""
        <div class="panel-heading">
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
        </div>
        """
    )


@st.fragment(run_every=1)
def render_live_terminal() -> None:
    step = current_step()
    if step >= SESSION_SECONDS - 1:
        pause_clock()
        st.session_state.clock["elapsed_before"] = SESSION_SECONDS - 1

    process_limit_orders_until(step)

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-title">Daytrade Desk Trainer</div>
            <div class="hero-sub">フル板・歩み値・1分足を同じシミュレーションから生成した、デイトレ学習用の疑似トレード端末です。</div>
            <div class="top-ribbon">
                <div class="ribbon-chip">板更新: 1秒ごと</div>
                <div class="ribbon-chip">日経平均先物: あらかじめ設定したシナリオ</div>
                <div class="ribbon-chip">個別銘柄: 日経平均と緩やかな相関あり</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_symbol = st.segmented_control(
        "銘柄切替",
        options=list(SYMBOLS.keys()),
        default=st.session_state.selected_symbol,
        format_func=lambda code: str(SYMBOLS[code]["label"]),
        key="symbol_tabs",
        label_visibility="collapsed",
    )
    if selected_symbol is None:
        selected_symbol = st.session_state.selected_symbol
    st.session_state.selected_symbol = selected_symbol

    symbol_df = st.session_state.market["symbols"][selected_symbol]
    symbol_snapshot = symbol_df.iloc[step]
    nikkei_df = st.session_state.market["nikkei"]
    nikkei_snapshot = nikkei_df.iloc[step]

    if st.session_state.order_symbol_bound != selected_symbol:
        st.session_state.order_symbol_bound = selected_symbol
        st.session_state.limit_price = int(symbol_snapshot["last"])

    controls = st.columns([1.2, 1.0, 0.8, 0.8, 0.8])
    with controls[0]:
        running_label = "稼働中" if st.session_state.clock["running"] else "停止中"
        st.markdown(
            f"""
            <div class="status-banner">
                <strong>セッション時刻</strong><br>
                {(SESSION_START + timedelta(seconds=step)).strftime('%H:%M:%S')} / {running_label}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with controls[1]:
        st.markdown(
            f"""
            <div class="status-banner">
                <strong>{NIKKEI_LABEL}</strong><br>
                {format_price(nikkei_snapshot['last'])} / VWAP {format_price(nikkei_snapshot['vwap'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with controls[2]:
        if st.button("一時停止" if st.session_state.clock["running"] else "再開", use_container_width=True):
            if st.session_state.clock["running"]:
                pause_clock()
                st.session_state.notice = "シミュレーションを一時停止しました。"
            else:
                resume_clock()
                st.session_state.notice = "シミュレーションを再開しました。"
            st.rerun()
    with controls[3]:
        if st.button("リセット", use_container_width=True):
            reset_session()
            st.rerun()
    with controls[4]:
        st.markdown(
            f"""
            <div class="status-banner">
                <strong>選択中</strong><br>
                {SYMBOLS[selected_symbol]["label"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="status-banner">{st.session_state.notice}</div>', unsafe_allow_html=True)

    left_col, middle_col, right_col = st.columns(3, gap="small")

    with left_col:
        with st.container(border=True):
            st.markdown(panel_header("タイムアンドセールス", "同一銘柄の約定フロー"), unsafe_allow_html=True)
            st.markdown(render_tape_html(symbol_df.iloc[: step + 1]), unsafe_allow_html=True)

    with middle_col:
        with st.container(border=True):
            st.markdown(panel_header("フル板", "歩み値と整合する深さ10本"), unsafe_allow_html=True)
            board = build_board_snapshot(symbol_snapshot, SYMBOLS[selected_symbol])
            st.markdown(render_board_html(board), unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(panel_header("注文画面", "新規・返済を分けた売買練習"), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                qty = st.number_input("株数", min_value=100, step=100, key="order_qty")
            with c2:
                order_type = st.selectbox("注文種別", ["成行", "指値"], key="order_type")
            with c3:
                st.number_input(
                    "指値",
                    min_value=int(symbol_snapshot["tick"]),
                    step=int(symbol_snapshot["tick"]),
                    key="limit_price",
                    disabled=order_type == "成行",
                )

            st.caption(
                f"最良買気配: {format_price(symbol_snapshot['best_bid'])} / "
                f"最良売気配: {format_price(symbol_snapshot['best_ask'])} / "
                f"現在値: {format_price(symbol_snapshot['last'])}"
            )
            position_qty = int(st.session_state.positions[selected_symbol]["qty"])
            close_buy_disabled = position_qty >= 0
            close_sell_disabled = position_qty <= 0
            qty_too_large_for_close_buy = position_qty < 0 and int(qty) > abs(position_qty)
            qty_too_large_for_close_sell = position_qty > 0 and int(qty) > position_qty

            if close_buy_disabled:
                st.caption("返済買いは売り建玉があるときのみ可能です。")
            elif qty_too_large_for_close_buy:
                st.caption(f"返済買いは現在の売り建玉 {abs(position_qty):,} 株までです。")

            if close_sell_disabled:
                st.caption("返済売りは買い建玉があるときのみ可能です。")
            elif qty_too_large_for_close_sell:
                st.caption(f"返済売りは現在の買い建玉 {position_qty:,} 株までです。")

            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                if st.button("新規買い", use_container_width=True, disabled=position_qty < 0):
                    message = (
                        execute_market_order(selected_symbol, "OPEN_BUY", int(qty), step)
                        if order_type == "成行"
                        else submit_limit_order(selected_symbol, "OPEN_BUY", int(qty), int(st.session_state.limit_price), step)
                    )
                    st.session_state.notice = message
                    st.rerun()
            with b2:
                if st.button("新規売り", use_container_width=True, disabled=position_qty > 0):
                    message = (
                        execute_market_order(selected_symbol, "OPEN_SELL", int(qty), step)
                        if order_type == "成行"
                        else submit_limit_order(selected_symbol, "OPEN_SELL", int(qty), int(st.session_state.limit_price), step)
                    )
                    st.session_state.notice = message
                    st.rerun()
            with b3:
                if st.button(
                    "返済買い",
                    use_container_width=True,
                    disabled=close_buy_disabled or qty_too_large_for_close_buy,
                ):
                    message = (
                        execute_market_order(selected_symbol, "CLOSE_BUY", int(qty), step)
                        if order_type == "成行"
                        else submit_limit_order(selected_symbol, "CLOSE_BUY", int(qty), int(st.session_state.limit_price), step)
                    )
                    st.session_state.notice = message
                    st.rerun()
            with b4:
                if st.button(
                    "返済売り",
                    use_container_width=True,
                    disabled=close_sell_disabled or qty_too_large_for_close_sell,
                ):
                    message = (
                        execute_market_order(selected_symbol, "CLOSE_SELL", int(qty), step)
                        if order_type == "成行"
                        else submit_limit_order(selected_symbol, "CLOSE_SELL", int(qty), int(st.session_state.limit_price), step)
                    )
                    st.session_state.notice = message
                    st.rerun()
            with b5:
                if st.button("注文取消", use_container_width=True):
                    st.session_state.notice = cancel_symbol_orders(selected_symbol)
                    st.rerun()

    with right_col:
        symbol_orders = [order for order in st.session_state.orders if order["symbol"] == selected_symbol]
        pnl_summary = session_pnl(step)
        with st.container(border=True):
            st.markdown(panel_header("VWAP / 保有状況", "選択銘柄の約定と建玉"), unsafe_allow_html=True)
            st.markdown(
                render_status_html(
                    selected_symbol,
                    symbol_snapshot,
                    st.session_state.positions[selected_symbol],
                    symbol_orders,
                    pnl_summary,
                ),
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown(panel_header("1分足", "同一銘柄"), unsafe_allow_html=True)
            st.plotly_chart(
                build_chart(build_candles(symbol_df, step), f"{SYMBOLS[selected_symbol]['label']} 1分足", show_vwap=True),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with st.container(border=True):
            st.markdown(panel_header("日経平均先物 1分足", "事前設定シナリオ"), unsafe_allow_html=True)
            st.plotly_chart(
                build_chart(build_candles(nikkei_df, step), f"{NIKKEI_LABEL} 1分足", show_vwap=False),
                use_container_width=True,
                config={"displayModeBar": False},
            )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_state()
    inject_css()
    render_live_terminal()


if __name__ == "__main__":
    main()
