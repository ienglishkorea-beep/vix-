from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import yfinance as yf


# =========================================================
# CONFIG
# =========================================================

STATE_FILE = Path(os.getenv("STATE_FILE", "vix_reversal_state.json"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Core signal
BREADTH_THRESHOLD = float(os.getenv("BREADTH_THRESHOLD", "40.0"))
REGIME_USE_STRICT = os.getenv("REGIME_USE_STRICT", "1") == "1"

# VIX jump filters
VIX_MIN_LEVEL = float(os.getenv("VIX_MIN_LEVEL", "20.0"))
VIX_1D_JUMP_PCT = float(os.getenv("VIX_1D_JUMP_PCT", "12.0"))
VIX_3D_JUMP_PCT = float(os.getenv("VIX_3D_JUMP_PCT", "25.0"))

# Extra filters
SPY_DROP_TRIGGER = float(os.getenv("SPY_DROP_TRIGGER", "-1.2"))
CANDIDATE_MIN_DROP = float(os.getenv("CANDIDATE_MIN_DROP", "-2.0"))
USE_VIX3M_FILTER = os.getenv("USE_VIX3M_FILTER", "1") == "1"

# Trade plan
TOP_N = int(os.getenv("TOP_N", "2"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.06"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.10"))
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "5"))

# Universe
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"
VIX3M_TICKER = "^VIX3M"

ETF_UNIVERSE = [
    ("SMH", "VanEck Semiconductor ETF"),
    ("SOXX", "iShares Semiconductor ETF"),
    ("IGV", "iShares Expanded Tech-Software Sector ETF"),
    ("QQQ", "Invesco QQQ Trust"),
    ("XLK", "Technology Select Sector SPDR Fund"),
    ("IWM", "iShares Russell 2000 ETF"),
    ("XBI", "SPDR S&P Biotech ETF"),
    ("ARKK", "ARK Innovation ETF"),
    ("FDN", "First Trust Dow Jones Internet Index Fund"),
    ("MTUM", "iShares MSCI USA Momentum Factor ETF"),
]

HISTORY_DAYS = 320
TIMEOUT = 20


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class MarketRegime:
    close: float
    sma50: float
    sma200: float
    close_above_200: bool
    sma50_above_200: bool
    passed: bool


@dataclass
class SpyShock:
    close: float
    prev_close: float
    day_return_pct: float
    threshold: float
    triggered: bool


@dataclass
class VixSignal:
    close: float
    prev_close: float
    close_3d_ago: float
    jump_1d_pct: float
    jump_3d_pct: float
    min_level_passed: bool
    jump_1d_passed: bool
    jump_3d_passed: bool
    triggered: bool


@dataclass
class VixTermStructure:
    vix: float
    vix3m: float
    backwardation: bool
    filter_used: bool
    passed: bool


@dataclass
class BreadthSignal:
    total_count: int
    valid_count: int
    above_50_count: int
    pct_above_50: float
    threshold: float
    triggered: bool


@dataclass
class CandidateETF:
    ticker: str
    name: str
    close: float
    day_return_pct: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    score_rank: int


@dataclass
class SignalReport:
    date_utc: str
    signal_triggered: bool
    regime: MarketRegime
    spy_shock: SpyShock
    vix: VixSignal
    term_structure: VixTermStructure
    breadth: BreadthSignal
    selected: List[CandidateETF]
    reason: str


# =========================================================
# UTIL
# =========================================================

def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_state() -> Dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(data: Dict) -> None:
    STATE_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram env not set. Skipping send.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()


def safe_pct(a: float, b: float) -> float:
    if b == 0 or pd.isna(a) or pd.isna(b):
        return float("nan")
    return (a / b - 1.0) * 100.0


def latest_two(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna()
    if len(s) < 2:
        raise RuntimeError("Not enough price history.")
    return float(s.iloc[-1]), float(s.iloc[-2])


# =========================================================
# MARKET DATA
# =========================================================

def get_sp500_tickers() -> List[str]:
    csv_path = Path("sp500.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "sp500.csv not found. Create a file named sp500.csv with one column named ticker."
        )

    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns:
        raise ValueError("sp500.csv must contain a column named ticker.")

    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not tickers:
        raise ValueError("sp500.csv ticker list is empty.")

    return sorted(set(tickers))


def download_close_history(
    tickers: List[str],
    period_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in data.columns:
                closes[t] = data[(t, "Close")]
        df = pd.DataFrame(closes)
    else:
        if "Close" not in data.columns:
            raise RuntimeError("Close column missing from price download.")
        df = pd.DataFrame({tickers[0]: data["Close"]})

    return df.sort_index()


# =========================================================
# SIGNALS
# =========================================================

def compute_regime(spy_close: pd.Series) -> MarketRegime:
    s = spy_close.dropna()
    if len(s) < 220:
        raise RuntimeError("Not enough SPY history for regime calculation.")

    close = float(s.iloc[-1])
    sma50 = float(s.rolling(50).mean().iloc[-1])
    sma200 = float(s.rolling(200).mean().iloc[-1])

    close_above_200 = close > sma200
    sma50_above_200 = sma50 > sma200
    passed = close_above_200 and sma50_above_200 if REGIME_USE_STRICT else close_above_200

    return MarketRegime(
        close=round(close, 2),
        sma50=round(sma50, 2),
        sma200=round(sma200, 2),
        close_above_200=close_above_200,
        sma50_above_200=sma50_above_200,
        passed=passed,
    )


def compute_spy_shock(spy_close: pd.Series, threshold: float) -> SpyShock:
    close, prev_close = latest_two(spy_close)
    day_return_pct = safe_pct(close, prev_close)
    triggered = day_return_pct <= threshold

    return SpyShock(
        close=round(close, 2),
        prev_close=round(prev_close, 2),
        day_return_pct=round(day_return_pct, 2),
        threshold=threshold,
        triggered=triggered,
    )


def compute_vix_signal(vix_close: pd.Series) -> VixSignal:
    s = vix_close.dropna()
    if len(s) < 4:
        raise RuntimeError("Not enough VIX history.")

    close = float(s.iloc[-1])
    prev_close = float(s.iloc[-2])
    close_3d_ago = float(s.iloc[-4])

    jump_1d_pct = safe_pct(close, prev_close)
    jump_3d_pct = safe_pct(close, close_3d_ago)

    min_level_passed = close >= VIX_MIN_LEVEL
    jump_1d_passed = jump_1d_pct >= VIX_1D_JUMP_PCT
    jump_3d_passed = jump_3d_pct >= VIX_3D_JUMP_PCT
    triggered = min_level_passed and (jump_1d_passed or jump_3d_passed)

    return VixSignal(
        close=round(close, 2),
        prev_close=round(prev_close, 2),
        close_3d_ago=round(close_3d_ago, 2),
        jump_1d_pct=round(jump_1d_pct, 2),
        jump_3d_pct=round(jump_3d_pct, 2),
        min_level_passed=min_level_passed,
        jump_1d_passed=jump_1d_passed,
        jump_3d_passed=jump_3d_passed,
        triggered=triggered,
    )


def compute_vix_term_structure(vix_close: pd.Series, vix3m_close: pd.Series) -> VixTermStructure:
    s1 = vix_close.dropna()
    s2 = vix3m_close.dropna()

    if len(s1) < 1 or len(s2) < 1:
        raise RuntimeError("Not enough VIX/VIX3M history.")

    vix = float(s1.iloc[-1])
    vix3m = float(s2.iloc[-1])
    backwardation = vix > vix3m
    passed = backwardation if USE_VIX3M_FILTER else True

    return VixTermStructure(
        vix=round(vix, 2),
        vix3m=round(vix3m, 2),
        backwardation=backwardation,
        filter_used=USE_VIX3M_FILTER,
        passed=passed,
    )


def compute_breadth_signal(all_closes: pd.DataFrame, threshold: float) -> BreadthSignal:
    valid_count = 0
    above_50 = 0

    for col in all_closes.columns:
        s = all_closes[col].dropna()
        if len(s) < 60:
            continue

        sma50 = s.rolling(50).mean().iloc[-1]
        close = s.iloc[-1]

        if pd.isna(sma50):
            continue

        valid_count += 1
        if close > sma50:
            above_50 += 1

    total_count = len(all_closes.columns)
    pct = 0.0 if valid_count == 0 else above_50 / valid_count * 100.0

    return BreadthSignal(
        total_count=total_count,
        valid_count=valid_count,
        above_50_count=above_50,
        pct_above_50=round(pct, 2),
        threshold=threshold,
        triggered=pct <= threshold,
    )


def rank_candidates(
    etf_closes: pd.DataFrame,
    top_n: int,
    min_drop: float,
) -> List[CandidateETF]:
    candidates: List[CandidateETF] = []

    for ticker, name in ETF_UNIVERSE:
        if ticker not in etf_closes.columns:
            continue

        s = etf_closes[ticker].dropna()
        if len(s) < 2:
            continue

        close, prev_close = latest_two(s)
        day_return_pct = safe_pct(close, prev_close)

        if pd.isna(day_return_pct):
            continue

        if day_return_pct > min_drop:
            continue

        entry_price = round(close * 1.002, 2)
        stop_price = round(close * (1.0 - STOP_LOSS_PCT), 2)
        take_profit_price = round(close * (1.0 + TAKE_PROFIT_PCT), 2)

        candidates.append(
            CandidateETF(
                ticker=ticker,
                name=name,
                close=round(close, 2),
                day_return_pct=round(day_return_pct, 2),
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                score_rank=0,
            )
        )

    candidates.sort(key=lambda x: x.day_return_pct)

    for i, c in enumerate(candidates, start=1):
        c.score_rank = i

    return candidates[:top_n]


# =========================================================
# REPORTING
# =========================================================

def build_reason(
    regime: MarketRegime,
    spy_shock: SpyShock,
    vix: VixSignal,
    term_structure: VixTermStructure,
    breadth: BreadthSignal,
    selected: List[CandidateETF],
) -> str:
    parts = []

    parts.append("레짐 통과" if regime.passed else "레짐 미통과")

    if spy_shock.triggered:
        parts.append(f"SPY {spy_shock.day_return_pct}% <= {spy_shock.threshold}%")
    else:
        parts.append(f"SPY 급락 미충족 ({spy_shock.day_return_pct}%)")

    if vix.min_level_passed:
        parts.append(f"VIX >= {VIX_MIN_LEVEL}")
    else:
        parts.append(f"VIX < {VIX_MIN_LEVEL}")

    parts.append(f"VIX 1일 {vix.jump_1d_pct}% / 3일 {vix.jump_3d_pct}%")

    if vix.jump_1d_passed or vix.jump_3d_passed:
        parts.append("jump 통과")
    else:
        parts.append("jump 미통과")

    if term_structure.filter_used:
        if term_structure.passed:
            parts.append(f"term 통과 (VIX {term_structure.vix} > VIX3M {term_structure.vix3m})")
        else:
            parts.append(f"term 미통과 (VIX {term_structure.vix} <= VIX3M {term_structure.vix3m})")
    else:
        parts.append("term 필터 미사용")

    if breadth.triggered:
        parts.append(f"breadth {breadth.pct_above_50}% <= {breadth.threshold}%")
    else:
        parts.append(f"breadth {breadth.pct_above_50}% > {breadth.threshold}%")

    parts.append(f"후보 {len(selected)}개" if selected else "후보 없음")

    return " | ".join(parts)


def build_message(report: SignalReport) -> str:
    lines = []
    lines.append("VIX 역추세 시스템 체크")
    lines.append(f"시각: {report.date_utc}")
    lines.append("")
    lines.append(f"신호 발생: {'YES' if report.signal_triggered else 'NO'}")
    lines.append(report.reason)
    lines.append("")
    lines.append(
        f"SPY 레짐: close {report.regime.close:.2f} | 50MA {report.regime.sma50:.2f} | 200MA {report.regime.sma200:.2f}"
    )
    lines.append(
        f"SPY 일간변화: {report.spy_shock.day_return_pct:.2f}% | 트리거 {report.spy_shock.threshold:.2f}%"
    )
    lines.append(
        f"VIX: {report.vix.close:.2f} | 1일변화 {report.vix.jump_1d_pct:.2f}% | 3일변화 {report.vix.jump_3d_pct:.2f}% | 절대레벨 기준 {VIX_MIN_LEVEL:.2f}"
    )
    lines.append(
        f"VIX3M: {report.term_structure.vix3m:.2f} | backwardation: {'YES' if report.term_structure.backwardation else 'NO'}"
    )
    lines.append(
        f"Breadth: {report.breadth.pct_above_50:.2f}% above 50DMA ({report.breadth.above_50_count}/{report.breadth.valid_count})"
    )
    lines.append("")

    if report.signal_triggered and report.selected:
        lines.append("매수 후보")
        for c in report.selected:
            lines.append(
                f"- {c.ticker} {c.name} | 종가 {c.close:.2f} | 당일수익률 {c.day_return_pct:.2f}% | 진입 {c.entry_price:.2f} | 손절 {c.stop_price:.2f} | 익절 {c.take_profit_price:.2f}"
            )
        lines.append("")
        lines.append(
            f"보유 계획: 기본 {MAX_HOLD_DAYS}거래일, 손절 -{int(STOP_LOSS_PCT * 100)}%, 익절 +{int(TAKE_PROFIT_PCT * 100)}%"
        )
    else:
        lines.append("오늘은 거래 없음")

    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    sp500 = get_sp500_tickers()
    base_tickers = [SPY_TICKER, VIX_TICKER, VIX3M_TICKER] + [x[0] for x in ETF_UNIVERSE]
    all_tickers = sorted(set(sp500 + base_tickers))

    print(f"[INFO] Downloading data for {len(all_tickers)} tickers...")
    closes = download_close_history(all_tickers, HISTORY_DAYS)

    required = [SPY_TICKER, VIX_TICKER, VIX3M_TICKER]
    for ticker in required:
        if ticker not in closes.columns:
            raise RuntimeError(f"{ticker} data missing.")

    spy_close = closes[SPY_TICKER]
    vix_close = closes[VIX_TICKER]
    vix3m_close = closes[VIX3M_TICKER]

    etf_cols = [x[0] for x in ETF_UNIVERSE if x[0] in closes.columns]
    etf_closes = closes[etf_cols]
    breadth_source = closes[[c for c in sp500 if c in closes.columns]]

    regime = compute_regime(spy_close)
    spy_shock = compute_spy_shock(spy_close, SPY_DROP_TRIGGER)
    vix = compute_vix_signal(vix_close)
    term_structure = compute_vix_term_structure(vix_close, vix3m_close)
    breadth = compute_breadth_signal(breadth_source, BREADTH_THRESHOLD)

    pre_selected = rank_candidates(etf_closes, TOP_N, CANDIDATE_MIN_DROP)

    signal_triggered = (
        regime.passed
        and spy_shock.triggered
        and vix.triggered
        and term_structure.passed
        and breadth.triggered
        and len(pre_selected) > 0
    )

    selected = pre_selected if signal_triggered else []

    report = SignalReport(
        date_utc=utc_now_str(),
        signal_triggered=signal_triggered,
        regime=regime,
        spy_shock=spy_shock,
        vix=vix,
        term_structure=term_structure,
        breadth=breadth,
        selected=selected,
        reason=build_reason(regime, spy_shock, vix, term_structure, breadth, selected),
    )

    state = load_state()
    signal_key = f"{datetime.now(timezone.utc).date()}_{int(signal_triggered)}"
    if selected:
        signal_key += "_" + "_".join([c.ticker for c in selected])

    message = build_message(report)

    print(message)
    print("")

    if state.get("last_signal_key") != signal_key:
        if signal_triggered:
            send_telegram(message)
            print("[INFO] Telegram alert sent.")
        else:
            print("[INFO] No trade signal. Telegram skipped.")
        state["last_signal_key"] = signal_key
        state["last_report"] = asdict(report)
        save_state(state)
    else:
        print("[INFO] Duplicate signal. Skipped sending.")

    out_path = Path("vix_reversal_last_report.json")
    out_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Report saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
