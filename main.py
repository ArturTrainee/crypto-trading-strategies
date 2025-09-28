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
) -> Dict[str, Any]:
    """
    Simple long-only backtest:
      - Enters at next candle open when signal true.
      - Trailing stop (with optional activation offset) + stoploss.
      - One position at a time, all-in position sizing.
      - Uses daily bars; stop/TSL execution assumed at stop level if within day's range.
    """
    if strategy is None:
        strategy = LongTrailingStrategyBacktest()

    df = strategy.populate_indicators(df)
    df = strategy.populate_entry_trend(df)

    # We will enter at next bar open, so we need shift of enter signal
    df["enter_next"] = df["enter_long"].shift(1).fillna(False)

    cash = starting_cash
    position_size = 0.0
    entry_price = None
    max_price_since_entry = None
    trailing_active = False

    stoploss_level = None
    trail_level = None

    trades: List[Dict[str, Any]] = []

    # For equity curve
    equity_curve: List[float] = []

    # iterate row by row
    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # Update equity
        if position_size > 0:
            equity_curve.append(position_size * c)
        else:
            equity_curve.append(cash)

        # If in position, manage stoploss/trailing
        if position_size > 0:
            # Update max price
            max_price_since_entry = max(max_price_since_entry, h)

            # Activation of trailing if applicable
            if strategy.trailing_stop:
                if strategy.trailing_only_offset_is_reached:
                    # Activate trailing when profit >= offset
                    if not trailing_active:
                        if (
                                max_price_since_entry - entry_price) / entry_price >= strategy.trailing_stop_positive_offset:
                            trailing_active = True
                    if trailing_active:
                        trail_level = max(trail_level or 0.0,
                                          max_price_since_entry * (1.0 - strategy.trailing_stop_positive))
                else:
                    # Trailing always active
                    trailing_active = True
                    trail_level = max(trail_level or 0.0,
                                      max_price_since_entry * (1.0 - strategy.trailing_stop_positive))

            # Compute stop levels
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
                cash = gross_value - sell_fee
                # trade record
                trades[-1]["exit_time"] = date
                trades[-1]["exit_price"] = exit_price
                trades[-1]["exit_reason"] = exit_reason
                trades[-1]["sell_fee"] = sell_fee
                trades[-1]["pnl"] = cash - trades[-1]["cash_before_entry"]
                trades[-1]["return_pct"] = (exit_price / trades[-1]["entry_price"] - 1.0) * 100.0 - (
                            fee_rate * 100.0) - (fee_rate * 100.0)

                # Reset position
                position_size = 0.0
                entry_price = None
                max_price_since_entry = None
                trailing_active = False
                stoploss_level = None
                trail_level = None

                continue  # move to next bar

        # If flat, consider entry at next candle open (we're at current candle close)
        if position_size == 0.0 and i + 1 < len(df):
            # If today's signal says enter on next bar
            if df.iloc[i + 1]["enter_next"]:
                next_open = df.iloc[i + 1]["open"]
                # Buy with all cash at next open minus fees
                buy_cost = cash
                buy_fee = buy_cost * fee_rate
                net_cash_to_position = buy_cost - buy_fee
                position_size = net_cash_to_position / next_open
                entry_price = next_open
                max_price_since_entry = entry_price
                trailing_active = False
                stoploss_level = entry_price * (1.0 + strategy.stoploss)
                trail_level = None

                # record trade
                trades.append(
                    {
                        "entry_time": df.index[i + 1],
                        "entry_price": entry_price,
                        "cash_before_entry": cash,
                        "buy_fee": buy_fee,
                        "exit_time": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "sell_fee": None,
                        "pnl": None,
                        "return_pct": None,
                    }
                )
                cash = 0.0  # fully invested

    # If still in position at the end, close at last close
    if position_size > 0:
        last_row = df.iloc[-1]
        last_date = df.index[-1]
        exit_price = last_row["close"]
        gross_value = position_size * exit_price
        sell_fee = gross_value * fee_rate
        cash = gross_value - sell_fee
        trades[-1]["exit_time"] = last_date
        trades[-1]["exit_price"] = exit_price
        trades[-1]["exit_reason"] = "end_of_test"
        trades[-1]["sell_fee"] = sell_fee
        trades[-1]["pnl"] = cash - trades[-1]["cash_before_entry"]
        trades[-1]["return_pct"] = (exit_price / trades[-1]["entry_price"] - 1.0) * 100.0 - (fee_rate * 100.0) - (
                    fee_rate * 100.0)

    # Final equity is cash (no open position)
    final_equity = cash
    equity_series = pd.Series(equity_curve, index=df.index)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min() if len(drawdown) else 0.0

    # KPI summary
    total_return = (final_equity / starting_cash - 1.0) * 100.0
    days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 1
    years = max(days / 365.25, 1e-9)
    # Compound Annual Growth Rate
    cagr = ((final_equity / starting_cash) ** (1 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

    wins = sum(1 for t in trades if t["pnl"] is not None and t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] is not None and t["pnl"] <= 0)
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0

    result = {
        "trades": trades,
        "summary": {
            "starting_cash": starting_cash,
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
    start_dt_str = input("Enter start date (YYYY-MM-DD, default: 2024-01-01): ") or "2024-01-01"
    end_dt_str = input("Enter end date (YYYY-MM-DD, default: 2025-01-01): ") or "2025-01-01"
    starting_cash = input("Enter starting position size in USD (default: 1000): ") or "1000"
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
    trades_file_name = f"trades/{symbol.replace('/', '')}_{timeframe}_{start_dt_str}_{end_dt_str}.csv"
    parent = os.path.dirname(trades_file_name)
    if parent:
        os.makedirs(parent, exist_ok=True)
    trades_df.to_csv(trades_file_name, index=False)
    print(f"\nTrades saved to {trades_file_name}")


if __name__ == "__main__":
    main()
