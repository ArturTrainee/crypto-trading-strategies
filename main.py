import time
import os

import ccxt
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


class LongTrailingStrategyBacktest:
    """
    Entry:
      - close > EMA200, EMA50, EMA20
      - EMA200, EMA50, EMA20 strictly increasing for the last 5 candles

    Exit:
      - Trailing stop with positive offset (offset must be reached before trailing activates)
      - Hard stoploss

    Parameters (aligned with your intent):
      stoploss = -0.9                       # -90%
      trailing_stop = True
      trailing_stop_positive = 0.1          # 10%
      trailing_stop_positive_offset = 0.5   # 50% profit threshold before trailing activates
      trailing_only_offset_is_reached = False  # If False, trailing is active immediately; True = after offset
    """

    def __init__(
            self,
            stoploss: float = -0.9,
            trailing_stop: bool = True,
            trailing_stop_positive: float = 0.1,
            trailing_stop_positive_offset: float = 0.5,
            trailing_only_offset_is_reached: bool = True,  # your code implied this, though it had a typo
    ):
        self.stoploss = stoploss
        self.trailing_stop = trailing_stop
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.trailing_only_offset_is_reached = trailing_only_offset_is_reached

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def strictly_increasing_last_n(series: pd.Series, n: int) -> pd.Series:
        """
        Returns True at index i if series[i] > series[i-1] > ... > series[i-(n-1)]
        """
        cond = pd.Series(True, index=series.index)
        for k in range(1, n):
            cond &= series.shift(k - 1) > series.shift(k)
        return cond.fillna(False)

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # EMAs
        df["ema20"] = self.ema(df["close"], 20)
        df["ema50"] = self.ema(df["close"], 50)
        df["ema200"] = self.ema(df["close"], 200)

        # "up_5" = strictly increasing for last 5 candles
        df["ema20_up_5"] = self.strictly_increasing_last_n(df["ema20"], 5)
        df["ema50_up_5"] = self.strictly_increasing_last_n(df["ema50"], 5)
        df["ema200_up_5"] = self.strictly_increasing_last_n(df["ema200"], 5)
        return df

    def populate_entry_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["enter_long"] = (
                (df["close"] > df["ema200"])
                & (df["close"] > df["ema50"])
                & (df["close"] > df["ema20"])
                & (df["ema200_up_5"])
                & (df["ema50_up_5"])
                & (df["ema20_up_5"])
        )
        return df


def fetch_ohlcv_ccxt(
        exchange_name: str,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: Optional[int] = None,
        limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch OHLCV data via CCXT in a loop to cover the requested timerange.
    """
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({"enableRateLimit": True})
    all_rows: List[List[Any]] = []
    fetch_since = since_ms

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        # Stop if we reached the desired end
        if until_ms is not None and last_ts >= until_ms:
            break
        # Advance since to last_ts + 1ms to avoid duplicates
        fetch_since = last_ts + 1
        # Be gentle with rate limits
        time.sleep(exchange.rateLimit / 1000.0)

    if not all_rows:
        raise RuntimeError("No data returned. Check symbol/timeframe/exchange.")

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("date", inplace=True)
    df = df.drop(columns=["timestamp"])
    df = df.sort_index()
    if until_ms is not None:
        end_dt = pd.to_datetime(until_ms, unit="ms", utc=True)
        df = df[df.index < end_dt]
    return df


def backtest_strategy(
        df: pd.DataFrame,
        fee_rate: float = 0.001,  # 0.1% per side (spot typical)
        starting_cash: float = 10_000.0,
        strategy: Optional[LongTrailingStrategyBacktest] = None,
        dca_step_pct: float = 0.05,
        dca_max_adds: int = 0,
) -> Dict[str, Any]:
    """
    Simple long-only backtest with optional position averaging (DCA):
      - Enters at next candle open when signal true.
      - Optional DCA: buy one equal tranche on every dca_step_pct drop from last fill price (intrabar),
        up to dca_max_adds additional fills. Tranche size is cash available at entry divided by (dca_max_adds+1).
      - Trailing stop (with optional activation offset) + hard stoploss.
      - One position at a time.
      - Uses bar data; stop/TSL execution assumed at stop level if within bar's range.
    """
    if strategy is None:
        strategy = LongTrailingStrategyBacktest()

    # Ensure numeric
    cash = float(starting_cash)

    df = strategy.populate_indicators(df)
    df = strategy.populate_entry_trend(df)

    # We will enter at next bar open, so we need shift of enter signal
    df["enter_next"] = df["enter_long"].shift(1).fillna(False)

    position_size = 0.0
    entry_price = None  # this will represent the current average entry price
    max_price_since_entry = None
    trailing_active = False

    stoploss_level = None
    trail_level = None

    # DCA state
    tranche_cash = 0.0
    dca_adds_done = 0
    last_fill_price = None
    next_add_price = None
    total_buy_fees = 0.0
    total_spent_cash = 0.0  # includes buy fees
    net_invested = 0.0      # excludes buy fees

    trades: List[Dict[str, Any]] = []

    # For equity curve
    equity_curve: List[float] = []

    # iterate row by row
    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # Update equity (cash + position value)
        equity_curve.append(cash + (position_size * c))

        # If in position, manage stoploss/trailing
        if position_size > 0:
            # Update max price
            max_price_since_entry = max(max_price_since_entry, h)

            # Activation of trailing if applicable
            if strategy.trailing_stop:
                if strategy.trailing_only_offset_is_reached:
                    # Activate trailing when profit >= offset
                    if not trailing_active:
                        if (max_price_since_entry - entry_price) / entry_price >= strategy.trailing_stop_positive_offset:
                            trailing_active = True
                    if trailing_active:
                        trail_level = max(
                            trail_level or 0.0,
                            max_price_since_entry * (1.0 - strategy.trailing_stop_positive),
                        )
                else:
                    # Trailing always active
                    trailing_active = True
                    trail_level = max(
                        trail_level or 0.0,
                        max_price_since_entry * (1.0 - strategy.trailing_stop_positive),
                    )

            # Compute stop levels based on average entry price
            stoploss_level = entry_price * (1.0 + strategy.stoploss)  # negative stoploss e.g. -0.9 -> 10% of entry
            effective_stop = stoploss_level
            if trailing_active and trail_level is not None:
                effective_stop = max(effective_stop, trail_level)

            exit_reason = None
            exit_price = None

            # If today's low breaches the effective stop, exit at stop level
            if l <= effective_stop <= h:
                exit_price = effective_stop
                exit_reason = "trailing_stop" if trailing_active and effective_stop == trail_level else "stoploss"
            # else hold

            # If exit triggered
            if exit_price is not None:
                # Sell entire position; deduct fee on selling
                gross_value = position_size * exit_price
                sell_fee = gross_value * fee_rate
                cash = cash + gross_value - sell_fee
                # trade record
                trades[-1]["exit_time"] = date
                trades[-1]["exit_price"] = exit_price
                trades[-1]["exit_reason"] = exit_reason
                trades[-1]["sell_fee"] = sell_fee
                trades[-1]["pnl"] = cash - trades[-1]["cash_before_entry"]
                spent = trades[-1].get("total_spent_cash", total_spent_cash)
                trades[-1]["return_pct"] = (trades[-1]["pnl"] / spent * 100.0) if spent > 0 else 0.0

                # Reset position & DCA state
                position_size = 0.0
                entry_price = None
                max_price_since_entry = None
                trailing_active = False
                stoploss_level = None
                trail_level = None
                tranche_cash = 0.0
                dca_adds_done = 0
                last_fill_price = None
                next_add_price = None
                total_buy_fees = 0.0
                total_spent_cash = 0.0
                net_invested = 0.0

                continue  # move to next bar

            # If not exited, process DCA adds (intrabar) if enabled
            if dca_max_adds > 0 and next_add_price is not None:
                while dca_adds_done < dca_max_adds and cash > 0 and l <= next_add_price:
                    # execute DCA buy at the grid price
                    spend = min(tranche_cash, cash)
                    if spend <= 0:
                        break
                    buy_fee = spend * fee_rate
                    net_spend = spend - buy_fee
                    add_price = next_add_price
                    size_added = net_spend / add_price
                    position_size += size_added
                    net_invested += net_spend
                    total_spent_cash += spend
                    total_buy_fees += buy_fee
                    cash -= spend

                    # Sync trade record totals
                    trades[-1]["total_spent_cash"] = total_spent_cash

                    # Update average entry and next grid
                    entry_price = net_invested / position_size
                    dca_adds_done += 1
                    last_fill_price = add_price
                    next_add_price = last_fill_price * (1.0 - dca_step_pct)

                    # Update trade record
                    trades[-1]["buy_fee_total"] = trades[-1].get("buy_fee_total", 0.0) + buy_fee
                    trades[-1]["dca_adds"] = dca_adds_done
                    trades[-1]["entry_price"] = entry_price

        # If flat, consider entry at next candle open (we're at current candle close)
        if position_size == 0.0 and i + 1 < len(df):
            # If today's signal says enter on next bar
            if df.iloc[i + 1]["enter_next"]:
                next_open = df.iloc[i + 1]["open"]
                # Allocate tranche size for DCA (if enabled)
                cash_before_entry_val = cash
                if dca_max_adds > 0:
                    tranche_cash = cash / float(dca_max_adds + 1)
                else:
                    tranche_cash = cash

                # First buy at next open
                spend = min(tranche_cash, cash)
                buy_fee = spend * fee_rate
                net_cash_to_position = spend - buy_fee
                position_size = net_cash_to_position / next_open
                entry_price = next_open if position_size == 0 else (net_cash_to_position / position_size)
                max_price_since_entry = next_open
                trailing_active = False
                stoploss_level = entry_price * (1.0 + strategy.stoploss)
                trail_level = None

                # Track investment stats
                total_spent_cash = spend
                total_buy_fees = buy_fee
                net_invested = net_cash_to_position
                cash -= spend

                # Init DCA grid
                dca_adds_done = 0
                last_fill_price = next_open
                next_add_price = last_fill_price * (1.0 - dca_step_pct) if dca_max_adds > 0 else None

                # record trade
                trades.append(
                    {
                        "entry_time": df.index[i + 1],
                        "entry_price": entry_price,
                        "cash_before_entry": cash_before_entry_val,
                        "buy_fee": buy_fee,
                        "buy_fee_total": buy_fee,
                        "dca_adds": 0,
                        "tranche_cash": tranche_cash,
                        "exit_time": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "sell_fee": None,
                        "pnl": None,
                        "return_pct": None,
                        "total_spent_cash": total_spent_cash,
                    }
                )

    # If still in position at the end, close at last close
    if position_size > 0:
        last_row = df.iloc[-1]
        last_date = df.index[-1]
        exit_price = last_row["close"]
        gross_value = position_size * exit_price
        sell_fee = gross_value * fee_rate
        cash = cash + gross_value - sell_fee
        trades[-1]["exit_time"] = last_date
        trades[-1]["exit_price"] = exit_price
        trades[-1]["exit_reason"] = "end_of_test"
        trades[-1]["sell_fee"] = sell_fee
        trades[-1]["pnl"] = cash - trades[-1]["cash_before_entry"]
        spent = trades[-1].get("total_spent_cash", total_spent_cash)
        trades[-1]["return_pct"] = (trades[-1]["pnl"] / spent * 100.0) if spent > 0 else 0.0

    # Final equity is cash (no open position)
    final_equity = cash
    equity_series = pd.Series(equity_curve, index=df.index)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min() if len(drawdown) else 0.0

    # KPI summary
    total_return = (final_equity / float(starting_cash) - 1.0) * 100.0
    days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 1
    years = max(days / 365.25, 1e-9)
    # Compound Annual Growth Rate
    cagr = ((final_equity / float(starting_cash)) ** (1 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

    wins = sum(1 for t in trades if t.get("pnl") is not None and t["pnl"] > 0)
    losses = sum(1 for t in trades if t.get("pnl") is not None and t["pnl"] <= 0)
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0

    result = {
        "trades": trades,
        "summary": {
            "starting_cash": float(starting_cash),
            "final_equity": final_equity,
            "total_return_pct": total_return,
            "CAGR_pct": cagr,
            "num_trades": len(trades),
            "win_rate_pct": win_rate,
            "max_drawdown_pct": max_dd * 100.0,
            "fee_rate_per_side": fee_rate,
        },
        "equity_curve": equity_series,
    }
    return result


class BearishShortStrategyBacktest:
    """
    Bearish market short strategy (mirror of long version):

    Entry (enter_short):
      - close < EMA200, EMA50, EMA20
      - EMA200, EMA50, EMA20 strictly decreasing for the last 5 candles

    Exit:
      - Trailing stop with positive offset (for shorts: activates after sufficient profit, i.e., price drops)
      - Hard stoploss (for shorts: price rising above a threshold)

    Parameters are the same semantics as in LongTrailingStrategyBacktest.
    """

    def __init__(
        self,
        stoploss: float = -0.9,
        trailing_stop: bool = True,
        trailing_stop_positive: float = 0.1,
        trailing_stop_positive_offset: float = 0.5,
        trailing_only_offset_is_reached: bool = True,
    ):
        self.stoploss = stoploss
        self.trailing_stop = trailing_stop
        self.trailing_stop_positive = trailing_stop_positive
        self.trailing_stop_positive_offset = trailing_stop_positive_offset
        self.trailing_only_offset_is_reached = trailing_only_offset_is_reached

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def strictly_decreasing_last_n(series: pd.Series, n: int) -> pd.Series:
        """
        Returns True at index i if series[i] < series[i-1] < ... < series[i-(n-1)]
        """
        cond = pd.Series(True, index=series.index)
        for k in range(1, n):
            cond &= series.shift(k - 1) < series.shift(k)
        return cond.fillna(False)

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema20"] = self.ema(df["close"], 20)
        df["ema50"] = self.ema(df["close"], 50)
        df["ema200"] = self.ema(df["close"], 200)

        df["ema20_down_5"] = self.strictly_decreasing_last_n(df["ema20"], 5)
        df["ema50_down_5"] = self.strictly_decreasing_last_n(df["ema50"], 5)
        df["ema200_down_5"] = self.strictly_decreasing_last_n(df["ema200"], 5)
        return df

    def populate_entry_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["enter_short"] = (
            (df["close"] < df["ema200"]) &
            (df["close"] < df["ema50"]) &
            (df["close"] < df["ema20"]) &
            (df["ema200_down_5"]) &
            (df["ema50_down_5"]) &
            (df["ema20_down_5"]) 
        )
        return df


def backtest_short_strategy(
    df: pd.DataFrame,
    fee_rate: float = 0.001,
    starting_cash: float = 10_000.0,
    strategy: Optional[BearishShortStrategyBacktest] = None,
    dca_step_pct: float = 0.05,
    dca_max_adds: int = 0,
) -> Dict[str, Any]:
    """
    Simple short-only backtest with optional position averaging on adverse moves (DCA adds when price rises):
      - Enters at next candle open when signal true.
      - DCA: allocate equal tranches of margin; each add increases size at higher prices.
      - Trailing stop for shorts (trail above the lowest price reached after activation).
      - Hard stoploss for shorts (price rising sufficiently above entry).
      - One position at a time. Uses bar data with stop execution at stop level if within the bar.
    """
    if strategy is None:
        strategy = BearishShortStrategyBacktest()

    cash = float(starting_cash)

    df = strategy.populate_indicators(df)
    df = strategy.populate_entry_trend(df)
    df["enter_next"] = df.get("enter_short", False).shift(1).fillna(False)

    position_size = 0.0  # in asset units, positive number representing shorted amount
    entry_price = None
    min_price_since_entry = None
    trailing_active = False

    # For shorts, stoploss above entry; with stoploss negative, use (1 - stoploss)
    stoploss_level = None
    trail_level = None  # for shorts: trail price above min_price_since_entry

    # DCA state
    tranche_cash = 0.0
    dca_adds_done = 0
    last_fill_price = None
    next_add_price = None
    total_open_fees = 0.0
    total_reserved_margin = 0.0
    net_invested_margin = 0.0

    trades: List[Dict[str, Any]] = []
    equity_curve: List[float] = []

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # Unrealized PnL for short if in position
        unrealized = 0.0
        if position_size > 0 and entry_price is not None:
            unrealized = position_size * (entry_price - c)
        # Equity = free cash + reserved margin + unrealized PnL
        equity_curve.append(cash + total_reserved_margin + unrealized)

        if position_size > 0:
            # Update min price since entry
            min_price_since_entry = min(min_price_since_entry, l)

            # Trailing activation and level (for shorts)
            if strategy.trailing_stop:
                if strategy.trailing_only_offset_is_reached:
                    if not trailing_active:
                        if (entry_price - min_price_since_entry) / entry_price >= strategy.trailing_stop_positive_offset:
                            trailing_active = True
                    if trailing_active:
                        trail_level = min(
                            trail_level if trail_level is not None else float('inf'),
                            min_price_since_entry * (1.0 + strategy.trailing_stop_positive),
                        )
                else:
                    trailing_active = True
                    trail_level = min(
                        trail_level if trail_level is not None else float('inf'),
                        min_price_since_entry * (1.0 + strategy.trailing_stop_positive),
                    )

            # Compute hard stop (for shorts: above entry)
            stoploss_level = entry_price * (1.0 - strategy.stoploss)  # with negative stoploss, e.g. -0.9 -> 1.9x entry
            effective_stop = stoploss_level
            if trailing_active and trail_level is not None:
                # For shorts, stop should be the minimum of (higher price is worse)
                effective_stop = min(effective_stop, trail_level)

            exit_reason = None
            exit_price = None

            # If today's high breaches the effective stop, exit at stop level
            if l <= effective_stop <= h:
                exit_price = effective_stop
                exit_reason = "trailing_stop" if trailing_active and effective_stop == trail_level else "stoploss"

            if exit_price is not None:
                # Close the short: PnL = size * (avg_entry - exit)
                pnl = position_size * (entry_price - exit_price)
                gross_cover = position_size * exit_price
                close_fee = gross_cover * fee_rate

                cash = cash + total_reserved_margin + pnl - close_fee

                trades[-1]["exit_time"] = date
                trades[-1]["exit_price"] = exit_price
                trades[-1]["exit_reason"] = exit_reason
                trades[-1]["close_fee"] = close_fee
                trades[-1]["pnl"] = cash - trades[-1]["cash_before_entry"]
                spent = trades[-1].get("total_reserved_margin", total_reserved_margin)
                trades[-1]["return_pct"] = (trades[-1]["pnl"] / spent * 100.0) if spent > 0 else 0.0

                # Reset state
                position_size = 0.0
                entry_price = None
                min_price_since_entry = None
                trailing_active = False
                stoploss_level = None
                trail_level = None
                tranche_cash = 0.0
                dca_adds_done = 0
                last_fill_price = None
                next_add_price = None
                total_open_fees = 0.0
                total_reserved_margin = 0.0
                net_invested_margin = 0.0
                continue

            # Process DCA adds on adverse move up
            if dca_max_adds > 0 and next_add_price is not None:
                while dca_adds_done < dca_max_adds and cash > 0 and h >= next_add_price:
                    spend = min(tranche_cash, cash)
                    if spend <= 0:
                        break
                    open_fee = spend * fee_rate
                    net_margin = spend - open_fee
                    add_price = next_add_price
                    size_added = net_margin / add_price
                    position_size += size_added
                    net_invested_margin += net_margin
                    total_reserved_margin += spend
                    total_open_fees += open_fee
                    cash -= spend

                    trades[-1]["total_reserved_margin"] = total_reserved_margin

                    # Update average entry (by size weighting)
                    entry_price = net_invested_margin / position_size
                    dca_adds_done += 1
                    last_fill_price = add_price
                    next_add_price = last_fill_price * (1.0 + dca_step_pct)

                    trades[-1]["open_fee_total"] = trades[-1].get("open_fee_total", 0.0) + open_fee
                    trades[-1]["dca_adds"] = dca_adds_done
                    trades[-1]["entry_price"] = entry_price

        # If flat, consider next bar entry
        if position_size == 0.0 and i + 1 < len(df):
            if df.iloc[i + 1]["enter_next"]:
                next_open = df.iloc[i + 1]["open"]

                cash_before_entry_val = cash
                if dca_max_adds > 0:
                    tranche_cash = cash / float(dca_max_adds + 1)
                else:
                    tranche_cash = cash

                # Open initial short using tranche_cash as margin
                spend = min(tranche_cash, cash)
                open_fee = spend * fee_rate
                net_margin = spend - open_fee

                size = net_margin / next_open
                position_size = size
                entry_price = next_open if position_size == 0 else (net_margin / position_size)
                min_price_since_entry = next_open
                trailing_active = False
                stoploss_level = entry_price * (1.0 - strategy.stoploss)
                trail_level = None

                total_reserved_margin = spend
                total_open_fees = open_fee
                net_invested_margin = net_margin
                cash -= spend

                dca_adds_done = 0
                last_fill_price = next_open
                next_add_price = last_fill_price * (1.0 + dca_step_pct) if dca_max_adds > 0 else None

                trades.append(
                    {
                        "entry_time": df.index[i + 1],
                        "entry_price": entry_price,
                        "cash_before_entry": cash_before_entry_val,
                        "open_fee": open_fee,
                        "open_fee_total": open_fee,
                        "dca_adds": 0,
                        "tranche_cash": tranche_cash,
                        "exit_time": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "close_fee": None,
                        "pnl": None,
                        "return_pct": None,
                        "total_reserved_margin": total_reserved_margin,
                    }
                )

    # Close at end if open: buy to cover at last close
    if position_size > 0:
        last_row = df.iloc[-1]
        last_date = df.index[-1]
        exit_price = last_row["close"]
        pnl = position_size * (entry_price - exit_price)
        gross_cover = position_size * exit_price
        close_fee = gross_cover * fee_rate
        cash = cash + total_reserved_margin + pnl - close_fee

        trades[-1]["exit_time"] = last_date
        trades[-1]["exit_price"] = exit_price
        trades[-1]["exit_reason"] = "end_of_test"
        trades[-1]["close_fee"] = close_fee
        trades[-1]["pnl"] = cash - trades[-1]["cash_before_entry"]
        spent = trades[-1].get("total_reserved_margin", total_reserved_margin)
        trades[-1]["return_pct"] = (trades[-1]["pnl"] / spent * 100.0) if spent > 0 else 0.0

    final_equity = cash
    equity_series = pd.Series(equity_curve, index=df.index)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min() if len(drawdown) else 0.0

    total_return = (final_equity / float(starting_cash) - 1.0) * 100.0
    days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 1
    years = max(days / 365.25, 1e-9)
    cagr = ((final_equity / float(starting_cash)) ** (1 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

    wins = sum(1 for t in trades if t.get("pnl") is not None and t["pnl"] > 0)
    losses = sum(1 for t in trades if t.get("pnl") is not None and t["pnl"] <= 0)
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0

    result = {
        "trades": trades,
        "summary": {
            "starting_cash": float(starting_cash),
            "final_equity": final_equity,
            "total_return_pct": total_return,
            "CAGR_pct": cagr,
            "num_trades": len(trades),
            "win_rate_pct": win_rate,
            "max_drawdown_pct": max_dd * 100.0,
            "fee_rate_per_side": fee_rate,
        },
        "equity_curve": equity_series,
    }
    return result


def prepare_data_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure data has the required columns and clean NaNs from long EMAs (e.g., ema200 warm-up).
    """
    # Nothing additional here; we keep full DF for indicator warm-up and backtest entry shift handles NaNs.
    return df

EXCHANGE_NAME = "bybit"

def main():
    symbol = input("Enter symbol (default: BTC/USDT): ") or "BTC/USDT"
    timeframe = input("Enter timeframe (default: 1d): ") or "1d"
    market_mode = (input("Market mode: bullish or bearish? (default: bullish): ") or "bullish").strip().lower()
    start_dt_str = input("Enter start date (YYYY-MM-DD, default: 2024-01-01): ") or "2024-01-01"
    end_dt_str = input("Enter end date (YYYY-MM-DD, default: 2025-01-01): ") or "2025-01-01"
    starting_cash = float(input("Enter starting position size in USD (default: 1000): ") or "1000")
    start_dt = datetime.strptime(start_dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)  # exclusive end
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    print(f"Downloading {symbol} {timeframe} data from {start_dt_str} to {end_dt_str}...")
    df = fetch_ohlcv_ccxt(
        exchange_name=EXCHANGE_NAME,
        symbol=symbol,
        timeframe=timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    if df.empty:
        raise RuntimeError("No data fetched.")

    # Prepare data
    df = prepare_data_with_indicators(df)

    # Run backtest
    print("Running backtest...")
    if market_mode == "bearish":
        strat = BearishShortStrategyBacktest(
            stoploss=-0.9,
            trailing_stop=True,
            trailing_stop_positive=0.1,
            trailing_stop_positive_offset=0.5,
            trailing_only_offset_is_reached=True,
        )
        result = backtest_short_strategy(
            df=df,
            fee_rate=0.001,
            starting_cash=starting_cash,
            strategy=strat,
            dca_step_pct=0.05,
            dca_max_adds=10,
        )
    else:
        strat = LongTrailingStrategyBacktest(
            stoploss=-0.9,
            trailing_stop=True,
            trailing_stop_positive=0.1,
            trailing_stop_positive_offset=0.5,
            trailing_only_offset_is_reached=True,  # activates trailing after +50% profit
        )
        result = backtest_strategy(
            df=df,
            fee_rate=0.001,  # 0.1% per side
            starting_cash=starting_cash,
            strategy=strat,
            dca_step_pct=0.05,
            dca_max_adds=10,
        )

    # Output summary
    summ = result["summary"]
    print("\n===== Backtest Summary =====")
    print(f"Starting cash               : ${summ['starting_cash']:.2f}")
    print(f"Final equity                : ${summ['final_equity']:.2f}")
    print(f"Total return                : {summ['total_return_pct']:.2f}%")
    print(f"Compound Annual Growth Rate : {summ['CAGR_pct']:.2f}%")
    print(f"Number of trades            : {summ['num_trades']}")
    print(f"Win rate                    : {summ['win_rate_pct']:.2f}%")
    print(f"Max drawdown                : {summ['max_drawdown_pct']:.2f}%")
    print(f"Fee rate per side           : {summ['fee_rate_per_side'] * 100:.3f}%")

    # Save trades
    trades_df = pd.DataFrame(result["trades"])
    mode_suffix = "bearish" if market_mode == "bearish" else "bullish"
    trades_file_name = f"trades/{symbol.replace('/', '')}_{timeframe}_{start_dt_str}_{end_dt_str}_{mode_suffix}.csv"
    parent = os.path.dirname(trades_file_name)
    if parent:
        os.makedirs(parent, exist_ok=True)
    trades_df.to_csv(trades_file_name, index=False)
    print(f"\nTrades saved to {trades_file_name}")


if __name__ == "__main__":
    main()
