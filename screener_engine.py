"""
Screener Engine — Smart Money Accumulation Detector
Scans Nifty 500 weekly charts for:
  1. Accumulation (BB Squeeze + OBV rising + Volume building)
  2. Breakout Initiated (Squeeze released + Volume spike + BB breakout)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta


def fetch_weekly(ticker, period="2y"):
    """Fetch weekly OHLCV data for a ticker."""
    try:
        if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            ticker = f"{ticker}.NS"
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1wk")
        if df.empty or len(df) < 30:
            return None
        return df
    except:
        return None


def compute_indicators(df):
    """Calculate all indicators needed for accumulation detection."""
    if df is None or len(df) < 26:
        return None

    # Bollinger Bands (20, 2)
    bb = ta.bbands(df['Close'], length=20, std=2)
    if bb is None or bb.empty:
        return None
    bbu = next((c for c in bb.columns if c.startswith("BBU")), None)
    bbl = next((c for c in bb.columns if c.startswith("BBL")), None)
    bbm = next((c for c in bb.columns if c.startswith("BBM")), None)
    if not bbu or not bbl:
        return None
    df['BB_Upper'] = bb[bbu]
    df['BB_Lower'] = bb[bbl]
    df['BB_Mid'] = bb[bbm] if bbm else df['Close'].rolling(20).mean()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    # Keltner Channel (20, 1.5 × ATR)
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=20)
    if atr is None:
        return None
    df['ATR'] = atr
    kc_mid = df['Close'].rolling(20).mean()
    df['KC_Upper'] = kc_mid + 1.5 * atr
    df['KC_Lower'] = kc_mid - 1.5 * atr

    # Squeeze: BB inside KC
    df['Squeeze'] = (df['BB_Lower'] > df['KC_Lower']) & (df['BB_Upper'] < df['KC_Upper'])

    # Volume indicators
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

    # OBV (On-Balance Volume)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_MA10'] = df['OBV'].rolling(10).mean()

    # RSI
    rsi = ta.rsi(df['Close'], length=14)
    df['RSI'] = rsi

    # EMA Filters
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['EMA_40'] = ta.ema(df['Close'], length=40)

    return df


def count_squeeze_weeks(df):
    """Count consecutive squeeze weeks ending at the most recent bar."""
    squeeze = df['Squeeze'].values
    count = 0
    for i in range(len(squeeze) - 1, -1, -1):
        if squeeze[i]:
            count += 1
        else:
            break
    return count


def compute_obv_slope(df, lookback=6):
    """Compute OBV slope over last N weeks. Positive = rising."""
    obv = df['OBV'].dropna().values
    if len(obv) < lookback:
        return 0
    recent = obv[-lookback:]
    x = np.arange(len(recent))
    try:
        slope = np.polyfit(x, recent, 1)[0]
        return slope
    except:
        return 0


def volume_trend_building(df, lookback=8):
    """Check if recent volume trend is building vs prior period."""
    vols = df['Volume'].values
    if len(vols) < lookback:
        return False, 0
    half = lookback // 2
    recent_avg = np.mean(vols[-half:])
    prior_avg = np.mean(vols[-lookback:-half])
    if prior_avg == 0:
        return False, 0
    ratio = recent_avg / prior_avg
    return ratio > 1.05, ratio


def scan_accumulating(df, min_squeeze_weeks=4, require_trend=True, min_up_down_ratio=1.2):
    """
    Scan Mode 1: Accumulating Now
    - BB Squeeze active for >= min_squeeze_weeks
    - OBV rising (positive slope)
    - Trend Filter: Price above 40-week EMA (optional)
    - Squeeze volume dry-up: Average volume inside squeeze <= 20w average (optional)
    - Buying pressure: Up/Down Volume Ratio during squeeze >= min_up_down_ratio
    - RSI < 60
    """
    if df is None or 'Squeeze' not in df.columns:
        return None

    squeeze_weeks = count_squeeze_weeks(df)
    if squeeze_weeks < min_squeeze_weeks:
        return None

    w_curr = df.iloc[-1]
    
    # 1. Trend Filter
    if require_trend and w_curr['Close'] < w_curr['EMA_40']:
        return None

    # 2. RSI check
    rsi = w_curr['RSI']
    if np.isnan(rsi) or rsi >= 60:
        return None

    # 3. OBV slope
    obv_slope = compute_obv_slope(df, lookback=min_squeeze_weeks + 2)
    if obv_slope <= 0:
        return None

    # 4. Volume dry-up check inside squeeze
    squeeze_slice = df.iloc[-squeeze_weeks:]
    avg_squeeze_vol = squeeze_slice['Volume'].mean()
    avg_20w_vol = w_curr['Vol_MA20'] if not np.isnan(w_curr['Vol_MA20']) else w_curr['Volume']
    
    # Volume must be contracting during squeeze
    if avg_squeeze_vol > avg_20w_vol * 1.1:
        return None

    # 5. Up/Down Volume Pressure Ratio
    green_weeks = squeeze_slice[squeeze_slice['Close'] >= squeeze_slice['Open']]
    red_weeks = squeeze_slice[squeeze_slice['Close'] < squeeze_slice['Open']]
    green_vol = green_weeks['Volume'].sum() if not green_weeks.empty else 0
    red_vol = red_weeks['Volume'].sum() if not red_weeks.empty else 0
    
    up_down_ratio = green_vol / red_vol if red_vol > 0 else 2.0 if green_vol > 0 else 1.0
    if up_down_ratio < min_up_down_ratio:
        return None

    # Volume building indicator (retained for scoring)
    vol_building, vol_ratio = volume_trend_building(df, lookback=min_squeeze_weeks * 2)
    last_vol_above_ma = w_curr['Vol_Ratio'] if not np.isnan(w_curr['Vol_Ratio']) else 0

    # Accumulation Strength Score (1-10)
    score = 0
    score += min(3.0, squeeze_weeks / 3.0)       # Up to 3 points for squeeze duration
    score += 2.0 if obv_slope > 0 else 0          # 2 points for OBV rising
    score += 1.5 if vol_building else 0            # 1.5 points for volume trend
    score += min(1.5, up_down_ratio * 0.5)        # Up to 1.5 points for Up/Down volume ratio
    score += 1.0 if rsi < 45 else 0.5 if rsi < 55 else 0  # 1 point for low RSI
    score += 1.0 if last_vol_above_ma < 1.0 else 0.5  # Under MA = dry volume (good)
    score = min(10, round(score, 1))

    # Breakout level
    bb_upper = w_curr['BB_Upper']
    current_price = w_curr['Close']
    bb_width = w_curr['BB_Width']

    # Entry Levels
    # Buy Zone: BB Mid to BB Lower
    buy_zone_low = w_curr['BB_Lower']
    buy_zone_high = w_curr['BB_Mid']
    stop_loss = buy_zone_low * 0.985 # 1.5% below BB Lower
    
    risk = buy_zone_high - stop_loss
    if risk <= 0:
        risk = buy_zone_high * 0.03
        stop_loss = buy_zone_high - risk
        
    target_price = buy_zone_high + 2 * risk
    rr_ratio = (target_price - buy_zone_high) / risk if risk > 0 else 0

    return {
        'mode': 'accumulating',
        'score': score,
        'price': current_price,
        'squeeze_weeks': squeeze_weeks,
        'rsi': rsi,
        'obv_slope': obv_slope,
        'vol_ratio': vol_ratio,
        'up_down_ratio': up_down_ratio,
        'vol_above_ma': last_vol_above_ma,
        'breakout_level': bb_upper,
        'buy_zone_low': buy_zone_low,
        'buy_zone_high': buy_zone_high,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'risk_reward': rr_ratio,
        'bb_width': bb_width,
        'trend_status': 'Uptrend' if current_price >= w_curr['EMA_40'] else 'Downtrend/Base'
    }


def scan_breakout(df, require_trend=True):
    """
    Scan Mode 2: Breakout Initiated
    - Squeeze was active within last 3 weeks but now released
    - Price closed above Upper BB recently
    - Volume spike (last week > 1.2× avg)
    - Current week above prior week close
    - Green candle
    """
    if df is None or 'Squeeze' not in df.columns:
        return None

    # Check squeeze was recently active (within last 5 bars) but now released
    squeeze = df['Squeeze'].values
    current_squeeze = squeeze[-1]

    # If still in squeeze, not a breakout
    if current_squeeze:
        return None

    # Check squeeze was active in recent history (within last 5 weeks)
    recent_squeeze_count = sum(squeeze[-6:-1])
    if recent_squeeze_count < 2:
        return None

    # Current or last week: close > Upper BB
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. Trend Filter
    if require_trend and curr['Close'] < curr['EMA_40']:
        return None

    bb_breakout = curr['Close'] > curr['BB_Upper'] or prev['Close'] > prev['BB_Upper']
    if not bb_breakout:
        return None

    # Volume spike
    vol_ratio = curr['Vol_Ratio'] if not np.isnan(curr['Vol_Ratio']) else 0
    if vol_ratio < 1.2:
        return None

    # Green candle and above prior close
    is_green = curr['Close'] > curr['Open']
    above_prev = curr['Close'] > prev['Close']
    if not (is_green or above_prev):
        return None

    # RSI should be > 50 (momentum starting)
    rsi = curr['RSI']
    if np.isnan(rsi) or rsi < 45:
        return None

    # OBV confirmation
    obv_slope = compute_obv_slope(df, lookback=6)

    # Find how many weeks squeeze lasted before release
    squeeze_duration = 0
    for i in range(len(squeeze) - 2, -1, -1):
        if squeeze[i]:
            squeeze_duration += 1
        elif squeeze_duration > 0:
            break

    # Breakout Strength Score (1-10)
    score = 0
    score += min(2.5, squeeze_duration / 2.0)  # Longer squeeze = stronger breakout
    score += min(2.5, vol_ratio * 1.0)           # Volume spike strength
    score += 2.0 if obv_slope > 0 else 0         # OBV confirmation
    score += 1.0 if is_green and above_prev else 0.5 if is_green else 0
    score += 1.0 if rsi > 55 else 0.5            # Momentum building
    score += 1.0 if curr['Close'] > curr['BB_Upper'] else 0.5
    score = min(10, round(score, 1))

    current_price = curr['Close']
    bb_width = curr['BB_Width']
    target_price = curr['BB_Upper'] + bb_width
    stop_loss = curr['BB_Mid']  # Mid BB as stop for breakout trades
    risk = current_price - stop_loss
    reward = target_price - current_price
    rr_ratio = reward / risk if risk > 0 else 0

    return {
        'mode': 'breakout',
        'score': score,
        'price': current_price,
        'squeeze_weeks': squeeze_duration,
        'rsi': rsi,
        'obv_slope': obv_slope,
        'vol_ratio': vol_ratio,
        'vol_above_ma': vol_ratio,
        'breakout_level': curr['BB_Upper'],
        'target_price': target_price,
        'stop_loss': stop_loss,
        'risk_reward': rr_ratio,
        'bb_width': bb_width,
        'trend_status': 'Uptrend' if current_price >= curr['EMA_40'] else 'Downtrend/Base'
    }


def scan_stock(ticker, min_squeeze_weeks=4, require_trend=True, min_up_down_ratio=1.2):
    """
    Scan a single stock for both accumulation and breakout signals.
    Returns (ticker, df_with_indicators, result_dict) or None.
    """
    df = fetch_weekly(ticker)
    if df is None:
        return None

    df = compute_indicators(df)
    if df is None:
        return None

    # Try accumulation first
    acc = scan_accumulating(df, min_squeeze_weeks, require_trend, min_up_down_ratio)
    if acc:
        return (ticker, df, acc)

    # Try breakout
    brk = scan_breakout(df, require_trend)
    if brk:
        return (ticker, df, brk)

    return None


def run_full_scan(stock_list, min_squeeze_weeks=4, require_trend=True, min_up_down_ratio=1.2, progress_callback=None):
    """
    Scan all stocks in the list in parallel using ThreadPoolExecutor.
    Returns two sorted lists: accumulating and breakout.
    """
    accumulating = []
    breakouts = []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_stocks = len(stock_list)
    completed = 0

    def scan_worker(ticker):
        return scan_stock(ticker, min_squeeze_weeks, require_trend, min_up_down_ratio)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(scan_worker, ticker): ticker for ticker in stock_list}
        for future in as_completed(futures):
            completed += 1
            ticker = futures[future]
            if progress_callback:
                progress_callback(ticker, completed - 1, total_stocks)
                
            try:
                result = future.result()
                if result:
                    t, df, info = result
                    rec = {'ticker': t, 'df': df, **info}
                    if info['mode'] == 'accumulating':
                        accumulating.append(rec)
                    else:
                        breakouts.append(rec)
            except Exception as e:
                pass

    # Sort by score descending
    accumulating.sort(key=lambda x: x['score'], reverse=True)
    breakouts.sort(key=lambda x: x['score'], reverse=True)

    return accumulating, breakouts
