"""
Valuation Engine - Complete Stock Fair Value Analyzer
5 Models: DCF, Graham, P/E Band, EV/EBITDA, DDM
+ Technical Entry Zones + Composite Scoring + Report
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io


# =============================================================================
# 1. DATA PARSERS
# =============================================================================

def parse_screener_datasheet(pasted_text):
    """
    Parses pasted text from Screener.in 'Data Sheet' tab.
    Returns a standardized dict with all financial data.
    """
    # Try tab-separated first, then fall back to comma or auto-detect
    try:
        df = pd.read_csv(io.StringIO(pasted_text), sep='\t', header=None)
    except:
        df = pd.read_csv(io.StringIO(pasted_text), header=None)
    
    # If only 1 column was parsed, the separator was wrong
    if df.shape[1] <= 2:
        # Try with different separators
        for sep in [',', '\s{2,}', None]:
            try:
                df = pd.read_csv(io.StringIO(pasted_text), sep=sep, header=None, engine='python')
                if df.shape[1] > 2:
                    break
            except:
                continue
    
    def find_row(keyword):
        """Flexible row finder — handles & vs AND, partial matches."""
        keyword_lower = keyword.lower()
        # Also create variant: replace & with AND and vice versa
        variants = [keyword_lower]
        if '&' in keyword_lower:
            variants.append(keyword_lower.replace('&', 'and'))
            variants.append(keyword_lower.replace('& ', '').replace(' &', ''))
        if 'and' in keyword_lower:
            variants.append(keyword_lower.replace('and', '&'))
        
        for idx, row in df.iterrows():
            val = str(row.iloc[0]).strip().lower()
            for v in variants:
                if val.startswith(v) or v.startswith(val.rstrip(':')):
                    return idx
        return None

    def get_values(row_idx):
        if row_idx is None:
            return None
        vals = df.iloc[row_idx, 1:].values
        return pd.to_numeric(vals, errors='coerce')

    def get_dates(row_idx):
        """Parse dates from Report Date row — handles multiple formats."""
        if row_idx is None:
            return None
        vals = df.iloc[row_idx, 1:].values
        dates = []
        for v in vals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                dates.append(pd.NaT)
                continue
            v_str = str(v).strip()
            if not v_str or v_str.lower() == 'nan':
                dates.append(pd.NaT)
                continue
            try:
                dates.append(pd.to_datetime(v_str))
            except:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%b-%y', '%B %Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        dates.append(datetime.strptime(v_str, fmt))
                        break
                    except:
                        continue
                else:
                    dates.append(pd.NaT)
        return dates

    # META
    meta = {}
    r = find_row('COMPANY NAME')
    meta['company'] = str(df.iloc[r, 1]).strip() if r is not None else 'Unknown'
    
    # Try to find Current Price
    cp = 0
    r_cp = find_row('Current Price')
    if r_cp is None:
        r_cp = find_row('PRICE') # Fallback
        
    if r_cp is not None:
        try:
            # Screener often has current price in the second column of the 'Current Price' or 'PRICE' row
            val = df.iloc[r_cp, 1]
            cp = float(str(val).replace(',', ''))
        except:
            pass
            
    meta['current_price'] = cp
    
    # Try to find Market Cap
    mc = 0
    r_mc = find_row('Market Cap')
    if r_mc is not None:
        try:
            val = df.iloc[r_mc, 1]
            mc = float(str(val).replace(',', ''))
        except:
            pass
    meta['market_cap'] = mc
    
    # Try to find Face Value
    fv = 1
    r_fv = find_row('Face Value')
    if r_fv is not None:
        try:
            val = df.iloc[r_fv, 1]
            fv = float(str(val).replace(',', ''))
        except:
            pass
    meta['face_value'] = fv
    r = find_row('Market Capitalization')
    try:
        meta['market_cap'] = float(df.iloc[r, 1]) if r is not None else 0
    except:
        meta['market_cap'] = 0
    r = find_row('Face Value')
    try:
        meta['face_value'] = float(df.iloc[r, 1]) if r is not None else 1
    except:
        meta['face_value'] = 1

    # PROFIT & LOSS section — try multiple variants
    pl_idx = find_row('PROFIT & LOSS')
    if pl_idx is None:
        pl_idx = find_row('PROFIT AND LOSS')
    if pl_idx is None:
        pl_idx = find_row('PROFIT')
    
    date_row = pl_idx + 1 if pl_idx is not None else None
    dates = get_dates(date_row)
    
    if dates is None:
        dates = []
    
    valid_mask = [pd.notna(d) for d in dates]
    valid_dates = [d for d, m in zip(dates, valid_mask) if m]
    n_years = len(valid_dates)

    def get_valid(row_idx):
        vals = get_values(row_idx)
        if vals is None or len(valid_dates) == 0:
            return np.full(max(1, len(valid_dates)), np.nan)
        filtered = []
        for v, m in zip(vals, valid_mask):
            if m:
                filtered.append(v if not (isinstance(v, str)) else np.nan)
        if not filtered:
            return np.full(max(1, len(valid_dates)), np.nan)
        return np.array(filtered, dtype=float)

    sales = get_valid(find_row('Sales'))
    net_profit = get_valid(find_row('Net profit'))
    depreciation = get_valid(find_row('Depreciation'))
    interest = get_valid(find_row('Interest'))
    tax = get_valid(find_row('Tax'))
    pbt = get_valid(find_row('Profit before tax'))
    other_income = get_valid(find_row('Other Income'))
    dividend = get_valid(find_row('Dividend Amount'))

    # Operating Profit = PBT + Interest + Depreciation - Other Income
    n = max(1, len(valid_dates))
    op_profit = np.nan_to_num(pbt, nan=0.0) + np.nan_to_num(interest, nan=0.0) + np.nan_to_num(depreciation, nan=0.0) - np.nan_to_num(other_income, nan=0.0)
    opm = np.where(sales > 0, (op_profit / sales) * 100, np.nan)
    ebitda = op_profit + np.nan_to_num(other_income, nan=0.0)

    # BALANCE SHEET
    equity_capital = get_valid(find_row('Equity Share Capital'))
    reserves = get_valid(find_row('Reserves'))
    borrowings = get_valid(find_row('Borrowings'))
    bs_idx = find_row('BALANCE SHEET')
    total_assets = np.full(n, np.nan)
    other_liabilities = get_valid(find_row('Other Liabilities'))
    receivables = get_valid(find_row('Receivables'))
    cash_bank = get_valid(find_row('Cash & Bank'))
    if np.all(np.isnan(cash_bank)):
        cash_bank = get_valid(find_row('Cash and Bank'))
    
    # Find Total rows in balance sheet section
    if bs_idx is not None:
        for idx in range(bs_idx + 2, min(bs_idx + 20, len(df))):
            val = str(df.iloc[idx, 0]).strip()
            if val == 'Total':
                total_assets = get_valid(idx)
                break

    shares_row = find_row('No. of Equity Shares')
    shares_outstanding = get_valid(shares_row) if shares_row else np.full(n, np.nan)
    adj_shares_row = find_row('Adjusted Equity Shares')
    adj_shares = get_valid(adj_shares_row) if adj_shares_row else None

    # Shareholders equity
    shareholders_equity = np.nan_to_num(equity_capital, nan=0.0) + np.nan_to_num(reserves, nan=0.0)
    book_value_ps = np.where(shares_outstanding > 0, 
                             shareholders_equity / (shares_outstanding / 1e7),
                             np.nan)

    # CASH FLOW
    cfo = get_valid(find_row('Cash from Operating Activity'))
    cfi = get_valid(find_row('Cash from Investing Activity'))
    cff = get_valid(find_row('Cash from Financing Activity'))
    fcf = np.nan_to_num(cfo, nan=0.0) + np.nan_to_num(cfi, nan=0.0)

    # PRICE
    price_row = find_row('PRICE')
    prices = get_valid(price_row) if price_row is not None else np.full(n, np.nan)

    # EPS
    eps = np.where(shares_outstanding > 0,
                   net_profit / (shares_outstanding / 1e7),
                   np.nan)

    # Dividends per share
    dps = np.where(shares_outstanding > 0,
                   dividend / (shares_outstanding / 1e7),
                   np.nan)

    # Build result
    data = {
        'meta': meta,
        'dates': valid_dates,
        'n_years': len(valid_dates),
        'sales': sales, 'net_profit': net_profit, 'depreciation': depreciation,
        'interest': interest, 'tax': tax, 'pbt': pbt,
        'other_income': other_income, 'dividend': dividend,
        'op_profit': op_profit, 'opm': opm, 'ebitda': ebitda,
        'equity_capital': equity_capital, 'reserves': reserves,
        'borrowings': borrowings, 'total_assets': total_assets,
        'other_liabilities': other_liabilities,
        'shareholders_equity': shareholders_equity,
        'book_value_ps': book_value_ps,
        'receivables': receivables, 'cash_bank': cash_bank,
        'shares_outstanding': shares_outstanding,
        'cfo': cfo, 'cfi': cfi, 'cff': cff, 'fcf': fcf,
        'prices': prices, 'eps': eps, 'dps': dps,
        'source': 'screener'
    }
    return data


def fetch_yahoo_data(ticker_symbol):
    """Fetches data from Yahoo Finance and returns standardized dict."""
    if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO"):
        ticker_symbol = f"{ticker_symbol}.NS"

    stock = yf.Ticker(ticker_symbol)
    info = stock.info or {}
    history = stock.history(period="10y")

    fin = stock.financials.T if stock.financials is not None else pd.DataFrame()
    bs = stock.balance_sheet.T if stock.balance_sheet is not None else pd.DataFrame()
    cf = stock.cashflow.T if stock.cashflow is not None else pd.DataFrame()

    if fin.empty:
        return None

    fin = fin.sort_index()
    bs = bs.sort_index()
    cf = cf.sort_index()

    dates = list(fin.index)
    n = len(dates)

    def safe_col(df, names, default=np.nan):
        # Align with master dates
        result = np.full(n, default)
        merged = pd.DataFrame(index=dates)
        
        target_col = None
        for name in (names if isinstance(names, list) else [names]):
            if name in df.columns:
                target_col = name
                break
        
        if target_col:
            # Join data to align dates correctly
            temp = df[[target_col]].copy()
            # Ensure index is datetime for proper joining
            temp.index = pd.to_datetime(temp.index)
            merged = merged.join(temp)
            result = merged[target_col].values
            
        return result

    sales = safe_col(fin, ['Total Revenue'])
    net_profit = safe_col(fin, ['Net Income'])
    op_income = safe_col(fin, ['Operating Income'])
    interest = safe_col(fin, ['Interest Expense'])
    tax = safe_col(fin, ['Tax Provision', 'Income Tax Expense'])
    depreciation = safe_col(fin, ['Depreciation And Amortization In Income Statement', 'Depreciation'])
    other_income = safe_col(fin, ['Other Income'])
    ebitda = safe_col(fin, ['EBITDA'])
    if np.all(np.isnan(ebitda)):
        ebitda = np.nan_to_num(op_income) + np.nan_to_num(depreciation)

    equity = safe_col(bs, ['Stockholders Equity', 'Total Equity Gross Minority Interest'])
    total_assets_val = safe_col(bs, ['Total Assets'])
    borrowings_val = safe_col(bs, ['Total Debt', 'Long Term Debt'])
    cash_val = safe_col(bs, ['Cash And Cash Equivalents'])
    shares = safe_col(bs, ['Share Issued', 'Ordinary Shares Number'])

    cfo_val = safe_col(cf, ['Operating Cash Flow'])
    capex_col = next((c for c in cf.columns if 'Capital Expenditure' in c), None)
    
    # Align capex manually to ensure shape (n,)
    if capex_col:
        temp_capex = cf[[capex_col]].copy()
        temp_capex.index = pd.to_datetime(temp_capex.index)
        merged_capex = pd.DataFrame(index=dates).join(temp_capex)
        capex = merged_capex[capex_col].values
    else:
        capex = np.zeros(n)
        
    fcf_val = np.nan_to_num(cfo_val) + np.nan_to_num(capex)  # capex usually negative

    # Use a helper to avoid repeated np.where for division
    def safe_div(a, b):
        return np.divide(a, b, out=np.full(n, np.nan), where=b > 0)

    opm = safe_div(op_income, sales) * 100
    eps_val = safe_div(net_profit, shares)
    bvps = safe_div(equity, shares)

    # Dividends
    div = safe_col(cf, ['Common Stock Dividend Paid'])
    div = np.abs(np.nan_to_num(div))
    dps = safe_div(div, shares)

    # Prices at report dates
    prices_arr = []
    for d in dates:
        try:
            mask = history.index.tz_localize(None) <= d.replace(tzinfo=None)
            if mask.any():
                prices_arr.append(history.loc[mask].iloc[-1]['Close'])
            else:
                prices_arr.append(np.nan)
        except:
            prices_arr.append(np.nan)

    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    if current_price == 0 and not history.empty:
        current_price = history['Close'].iloc[-1]

    data = {
        'meta': {
            'company': info.get('longName', ticker_symbol),
            'current_price': current_price,
            'market_cap': info.get('marketCap', 0),
            'face_value': info.get('faceValue', 1),
        },
        'dates': dates, 'n_years': n,
        'sales': sales, 'net_profit': net_profit, 'depreciation': depreciation,
        'interest': interest, 'tax': tax, 'pbt': np.nan_to_num(op_income),
        'other_income': other_income, 'dividend': div,
        'op_profit': op_income, 'opm': opm, 'ebitda': ebitda,
        'equity_capital': np.zeros(n), 'reserves': np.zeros(n),
        'borrowings': borrowings_val, 'total_assets': total_assets_val,
        'other_liabilities': np.zeros(n),
        'shareholders_equity': equity,
        'book_value_ps': bvps,
        'receivables': np.zeros(n), 'cash_bank': cash_val,
        'shares_outstanding': shares,
        'cfo': cfo_val, 'cfi': np.zeros(n), 'cff': np.zeros(n), 'fcf': fcf_val,
        'prices': np.array(prices_arr), 'eps': eps_val, 'dps': dps,
        'source': 'yahoo',
        'history': history,
        'ticker': ticker_symbol,
    }
    return data


# =============================================================================
# 2. VALUATION MODELS
# =============================================================================

def _calculate_cagr(values):
    """CAGR from first to last non-NaN positive value."""
    clean = [v for v in values if not np.isnan(v) and v > 0]
    if len(clean) < 2:
        return 0
    n = len(clean) - 1
    return ((clean[-1] / clean[0]) ** (1 / n) - 1) * 100


def model_dcf(data, discount_rate=0.12, terminal_growth=0.04, projection_years=5):
    """
    Discounted Cash Flow Model.
    Projects FCF forward, discounts back. Adds terminal value.
    """
    try:
        fcf = data['fcf']
        clean_fcf = [v for v in fcf if not np.isnan(v) and v != 0]
        if len(clean_fcf) < 2:
            return {'fair_price': 0, 'confidence': 0, 'error': 'Insufficient FCF data'}

        latest_fcf = clean_fcf[-1]
        fcf_growth = _calculate_cagr(clean_fcf) / 100
        
        # Cap growth rate
        fcf_growth = max(0.02, min(fcf_growth, 0.25))
        
        # Ensure discount > terminal growth
        if discount_rate <= terminal_growth:
            terminal_growth = discount_rate * 0.5

        # Project FCF
        projected_fcf = []
        for yr in range(1, projection_years + 1):
            proj = latest_fcf * (1 + fcf_growth) ** yr
            pv = proj / (1 + discount_rate) ** yr
            projected_fcf.append(pv)

        # Terminal Value (Gordon Growth)
        terminal_fcf = latest_fcf * (1 + fcf_growth) ** projection_years * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** projection_years

        total_value = sum(projected_fcf) + pv_terminal  # in Cr

        # Per share
        shares = data['shares_outstanding']
        clean_shares = [s for s in shares if not np.isnan(s) and s > 0]
        if not clean_shares:
            return {'fair_price': 0, 'confidence': 0, 'error': 'No shares data'}

        latest_shares = clean_shares[-1]
        # If source is screener, shares are in units; value is in Cr
        if data['source'] == 'screener':
            fair_price = (total_value * 1e7) / latest_shares
        else:
            fair_price = total_value / latest_shares

        # Confidence based on FCF consistency
        fcf_std = np.std(clean_fcf) / np.mean(clean_fcf) if np.mean(clean_fcf) != 0 else 1
        confidence = max(0.2, min(0.95, 1 - fcf_std))

        return {
            'fair_price': max(0, fair_price),
            'confidence': confidence,
            'fcf_growth': fcf_growth * 100,
            'discount_rate': discount_rate * 100,
            'terminal_growth': terminal_growth * 100,
            'projected_fcf_pv': sum(projected_fcf),
            'terminal_value_pv': pv_terminal,
            'total_value': total_value,
            'projection_years': projection_years,
        }
    except Exception as e:
        return {'fair_price': 0, 'confidence': 0, 'error': str(e)}


def model_graham(data):
    """
    Graham's Intrinsic Value = sqrt(22.5 × EPS × Book Value per Share)
    """
    try:
        eps = data['eps']
        bvps = data['book_value_ps']
        
        clean_eps = [v for v in eps if not np.isnan(v)]
        clean_bvps = [v for v in bvps if not np.isnan(v)]
        
        if not clean_eps or not clean_bvps:
            return {'fair_price': 0, 'confidence': 0, 'error': 'Missing EPS or Book Value'}

        latest_eps = clean_eps[-1]
        latest_bvps = clean_bvps[-1]

        if latest_eps <= 0 or latest_bvps <= 0:
            return {'fair_price': 0, 'confidence': 0.3, 'error': 'Negative EPS or BV — Graham not applicable'}

        fair_price = np.sqrt(22.5 * latest_eps * latest_bvps)
        
        # Confidence: higher if EPS is growing consistently
        eps_growth = _calculate_cagr(clean_eps)
        confidence = 0.7 if eps_growth > 5 else 0.5 if eps_growth > 0 else 0.3

        return {
            'fair_price': fair_price,
            'confidence': confidence,
            'eps': latest_eps,
            'book_value': latest_bvps,
            'eps_growth': eps_growth,
        }
    except Exception as e:
        return {'fair_price': 0, 'confidence': 0, 'error': str(e)}


def model_pe_band(data):
    """
    P/E Band Analysis — Historical P/E range × Current EPS.
    Shows if stock is at historical high/low P/E.
    """
    try:
        eps = data['eps']
        prices = data['prices']
        
        pe_ratios = []
        for e, p in zip(eps, prices):
            if not np.isnan(e) and not np.isnan(p) and e > 0:
                pe_ratios.append(p / e)

        if len(pe_ratios) < 2:
            return {'fair_price': 0, 'confidence': 0, 'error': 'Insufficient P/E data'}

        median_pe = np.median(pe_ratios)
        mean_pe = np.mean(pe_ratios)
        min_pe = np.min(pe_ratios)
        max_pe = np.max(pe_ratios)

        latest_eps_val = [v for v in eps if not np.isnan(v) and v > 0]
        if not latest_eps_val:
            return {'fair_price': 0, 'confidence': 0, 'error': 'No positive EPS'}

        current_eps = latest_eps_val[-1]
        current_price = data['meta']['current_price']
        current_pe = current_price / current_eps if current_eps > 0 else 0

        fair_price_median = median_pe * current_eps
        fair_price_mean = mean_pe * current_eps
        fair_price_low = min_pe * current_eps
        fair_price_high = max_pe * current_eps

        # Confidence based on spread
        pe_cv = np.std(pe_ratios) / mean_pe if mean_pe != 0 else 1
        confidence = max(0.3, min(0.9, 1 - pe_cv))

        return {
            'fair_price': fair_price_median,
            'confidence': confidence,
            'current_pe': current_pe,
            'median_pe': median_pe,
            'mean_pe': mean_pe,
            'min_pe': min_pe,
            'max_pe': max_pe,
            'fair_low': fair_price_low,
            'fair_high': fair_price_high,
            'pe_history': pe_ratios,
        }
    except Exception as e:
        return {'fair_price': 0, 'confidence': 0, 'error': str(e)}


def model_ev_ebitda(data):
    """
    EV/EBITDA Valuation.
    Enterprise Value = Market Cap + Debt - Cash
    Fair Price = (Median EV/EBITDA × Current EBITDA - Debt + Cash) / Shares
    """
    try:
        ebitda = data['ebitda']
        borrowings = data['borrowings']
        cash = data['cash_bank']
        shares = data['shares_outstanding']
        prices = data['prices']

        ev_ebitda_ratios = []
        for i in range(len(prices)):
            e = ebitda[i] if i < len(ebitda) else np.nan
            p = prices[i] if i < len(prices) else np.nan
            b = borrowings[i] if i < len(borrowings) else 0
            c = cash[i] if i < len(cash) else 0
            s = shares[i] if i < len(shares) else np.nan

            if np.isnan(e) or np.isnan(p) or np.isnan(s) or e <= 0 or s <= 0:
                continue

            if data['source'] == 'screener':
                mcap = p * (s / 1e7)  # price × shares in Cr
            else:
                mcap = p * s

            ev = mcap + np.nan_to_num(b) - np.nan_to_num(c)
            ratio = ev / e
            if ratio > 0:
                ev_ebitda_ratios.append(ratio)

        if len(ev_ebitda_ratios) < 2:
            return {'fair_price': 0, 'confidence': 0, 'error': 'Insufficient EV/EBITDA data'}

        median_ratio = np.median(ev_ebitda_ratios)
        latest_ebitda = [v for v in ebitda if not np.isnan(v) and v > 0][-1]
        latest_debt = np.nan_to_num(borrowings[-1])
        latest_cash = np.nan_to_num(cash[-1])
        latest_shares = [v for v in shares if not np.isnan(v) and v > 0][-1]

        fair_ev = median_ratio * latest_ebitda
        fair_equity = fair_ev - latest_debt + latest_cash

        if data['source'] == 'screener':
            fair_price = (fair_equity * 1e7) / latest_shares
        else:
            fair_price = fair_equity / latest_shares

        confidence = max(0.3, min(0.85, 1 - np.std(ev_ebitda_ratios) / np.mean(ev_ebitda_ratios)))

        return {
            'fair_price': max(0, fair_price),
            'confidence': confidence,
            'median_ev_ebitda': median_ratio,
            'current_ev_ebitda': ev_ebitda_ratios[-1] if ev_ebitda_ratios else 0,
            'history': ev_ebitda_ratios,
        }
    except Exception as e:
        return {'fair_price': 0, 'confidence': 0, 'error': str(e)}


def model_ddm(data, required_return=0.12):
    """
    Dividend Discount Model (Gordon Growth).
    Fair Price = DPS × (1 + g) / (r - g)
    """
    try:
        dps = data['dps']
        clean_dps = [v for v in dps if not np.isnan(v) and v > 0]

        if len(clean_dps) < 2:
            return {'fair_price': 0, 'confidence': 0, 'error': 'Insufficient dividend data (stock may not pay dividends)'}

        latest_dps = clean_dps[-1]
        div_growth = _calculate_cagr(clean_dps) / 100
        div_growth = max(0.01, min(div_growth, required_return - 0.01))

        fair_price = latest_dps * (1 + div_growth) / (required_return - div_growth)

        # Payout ratio check
        eps_vals = [v for v in data['eps'] if not np.isnan(v) and v > 0]
        payout = (latest_dps / eps_vals[-1] * 100) if eps_vals else 0

        confidence = 0.6 if div_growth > 0.03 else 0.4
        if payout > 80:
            confidence *= 0.7  # unsustainable payout

        return {
            'fair_price': max(0, fair_price),
            'confidence': confidence,
            'latest_dps': latest_dps,
            'div_growth': div_growth * 100,
            'payout_ratio': payout,
            'required_return': required_return * 100,
        }
    except Exception as e:
        return {'fair_price': 0, 'confidence': 0, 'error': str(e)}


# =============================================================================
# 3. TECHNICAL ENTRY ZONE
# =============================================================================

def analyze_technical_entry(data):
    """
    Calculates support/resistance, Fibonacci, and MA zones.
    Uses Yahoo Finance history or constructs from Screener prices.
    """
    try:
        history = data.get('history')
        ticker = data.get('ticker')
        current_price = data['meta']['current_price']

        if history is None or history.empty:
            # For screener source, fetch price history from yfinance
            symbol = data['meta'].get('company', '').split()[0]
            if not symbol.endswith('.NS'):
                symbol += '.NS'
            try:
                stock = yf.Ticker(symbol)
                history = stock.history(period="1y")
            except:
                return {'error': 'Cannot fetch price history for technical analysis'}

        if history.empty:
            return {'error': 'No price history available'}

        # 1. Moving Averages (use daily data)
        daily = history.copy()
        daily['MA50'] = daily['Close'].rolling(50).mean()
        daily['MA100'] = daily['Close'].rolling(100).mean()
        daily['MA200'] = daily['Close'].rolling(200).mean()

        ma50 = daily['MA50'].iloc[-1] if not np.isnan(daily['MA50'].iloc[-1]) else None
        ma100 = daily['MA100'].iloc[-1] if not np.isnan(daily['MA100'].iloc[-1]) else None
        ma200 = daily['MA200'].iloc[-1] if not np.isnan(daily['MA200'].iloc[-1]) else None

        # 2. 52-Week High/Low
        one_year = daily.tail(252)
        high_52w = one_year['High'].max()
        low_52w = one_year['Low'].min()

        # 3. Fibonacci Retracement from 52w high to low
        diff = high_52w - low_52w
        fib_levels = {
            '0% (High)': high_52w,
            '23.6%': high_52w - 0.236 * diff,
            '38.2%': high_52w - 0.382 * diff,
            '50.0%': high_52w - 0.500 * diff,
            '61.8%': high_52w - 0.618 * diff,
            '100% (Low)': low_52w,
        }

        # 4. Support/Resistance from pivot points
        recent = daily.tail(60)
        pivots = (recent['High'] + recent['Low'] + recent['Close']) / 3
        
        # Find nearest support (below price) and resistance (above price)
        support_candidates = [p for p in pivots if p < current_price]
        resistance_candidates = [p for p in pivots if p > current_price]
        
        support = np.percentile(support_candidates, 25) if support_candidates else low_52w
        resistance = np.percentile(resistance_candidates, 75) if resistance_candidates else high_52w

        # 5. Best Entry Zone
        entry_levels = [v for v in [ma200, fib_levels.get('61.8%'), support] if v is not None]
        if entry_levels:
            best_entry_low = min(entry_levels)
            best_entry_high = max(min(entry_levels), min([v for v in [ma100, fib_levels.get('50.0%')] if v is not None], default=current_price))
        else:
            best_entry_low = low_52w
            best_entry_high = current_price * 0.9

        return {
            'ma50': ma50, 'ma100': ma100, 'ma200': ma200,
            'high_52w': high_52w, 'low_52w': low_52w,
            'fib_levels': fib_levels,
            'support': support, 'resistance': resistance,
            'best_entry_low': best_entry_low,
            'best_entry_high': best_entry_high,
            'history': daily,
        }
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# 4. COMPOSITE SCORE & VERDICT
# =============================================================================

def calculate_composite(data, dcf, graham, pe_band, ev_ebitda, ddm, technical, margin_of_safety=0.15):
    """
    Weighted average of all model prices, applies margin of safety,
    and generates a Buy/Wait/Avoid verdict with score.
    """
    current_price = data['meta']['current_price']
    
    models = {
        'DCF': dcf, 'Graham': graham, 'P/E Band': pe_band,
        'EV/EBITDA': ev_ebitda, 'DDM': ddm,
    }
    
    # Weighted average (weight = confidence)
    total_weight = 0
    weighted_sum = 0
    model_prices = {}
    
    for name, m in models.items():
        fp = m.get('fair_price', 0)
        conf = m.get('confidence', 0)
        if fp > 0 and conf > 0:
            weighted_sum += fp * conf
            total_weight += conf
            model_prices[name] = {'price': fp, 'confidence': conf}

    if total_weight == 0:
        return {
            'composite_fair_price': 0, 'buy_below_price': 0,
            'verdict': 'INSUFFICIENT DATA', 'score': 0,
            'model_prices': {}, 'upside': 0,
        }

    composite_fair = weighted_sum / total_weight
    buy_below = composite_fair * (1 - margin_of_safety)
    
    # Upside/downside
    upside = ((composite_fair - current_price) / current_price) * 100
    upside_mos = ((buy_below - current_price) / current_price) * 100

    # Score (1-10)
    if upside > 30:
        score = 9
    elif upside > 20:
        score = 8
    elif upside > 10:
        score = 7
    elif upside > 0:
        score = 6
    elif upside > -10:
        score = 5
    elif upside > -20:
        score = 4
    else:
        score = 3

    # Adjust for technical alignment
    if technical and 'ma200' in technical and technical['ma200']:
        if current_price > technical['ma200']:
            score = min(10, score + 0.5)
        else:
            score = max(1, score - 0.5)

    # Verdict
    if current_price <= buy_below:
        verdict = 'STRONG BUY 🟢'
    elif current_price <= composite_fair * 0.95:
        verdict = 'BUY 🟡'
    elif current_price <= composite_fair * 1.05:
        verdict = 'WAIT ⏳'
    else:
        verdict = 'AVOID 🔴'

    return {
        'composite_fair_price': composite_fair,
        'buy_below_price': buy_below,
        'verdict': verdict,
        'score': round(score, 1),
        'model_prices': model_prices,
        'upside': upside,
        'upside_with_mos': upside_mos,
        'margin_of_safety': margin_of_safety * 100,
        'current_price': current_price,
    }


# =============================================================================
# 5. REPORT GENERATOR
# =============================================================================

def generate_valuation_report(data, dcf, graham, pe_band, ev_ebitda, ddm, technical, composite):
    """Generates a detailed markdown valuation report."""
    cp = data['meta']['current_price']
    company = data['meta']['company']
    src = data['source'].upper()
    
    r = []
    r.append(f"# 📊 Valuation Report: {company}")
    r.append(f"**Data Source:** {src} | **Analysis Date:** {datetime.now().strftime('%d-%b-%Y')}")
    r.append(f"**Current Market Price:** ₹{cp:,.2f}")
    r.append("")

    # Executive Summary
    r.append("## 🎯 Executive Summary")
    r.append(f"| Metric | Value |")
    r.append(f"|--------|-------|")
    r.append(f"| **Verdict** | **{composite['verdict']}** |")
    r.append(f"| **Composite Score** | **{composite['score']}/10** |")
    r.append(f"| **Fair Value (Weighted)** | **₹{composite['composite_fair_price']:,.2f}** |")
    r.append(f"| **Buy Below (15% MoS)** | **₹{composite['buy_below_price']:,.2f}** |")
    r.append(f"| **Upside to Fair Value** | **{composite['upside']:+.1f}%** |")
    r.append(f"| **Current Price** | ₹{cp:,.2f} |")
    r.append("")

    # Growth Metrics
    r.append("## 📈 Growth Analysis")
    sales_cagr = _calculate_cagr(data['sales'])
    profit_cagr = _calculate_cagr(data['net_profit'])
    fcf_cagr = _calculate_cagr(data['fcf'])
    r.append(f"- **Sales CAGR ({data['n_years']}yr):** {sales_cagr:.1f}%")
    r.append(f"- **Net Profit CAGR:** {profit_cagr:.1f}%")
    r.append(f"- **FCF CAGR:** {fcf_cagr:.1f}%")
    
    opm_arr = data['opm']
    if len(opm_arr) > 0 and not np.isnan(opm_arr[-1]):
        latest_opm = opm_arr[-1]
    else:
        latest_opm = 0
    r.append(f"- **Latest OPM:** {latest_opm:.1f}%")
    r.append("")

    # Model Breakdown
    r.append("## 🧮 Model-wise Fair Value Estimates")
    r.append("")
    r.append("| Model | Fair Price | Confidence | vs CMP |")
    r.append("|-------|-----------|------------|--------|")

    def model_row(name, m):
        fp = m.get('fair_price', 0)
        conf = m.get('confidence', 0)
        err = m.get('error', '')
        if fp > 0:
            diff = ((fp - cp) / cp) * 100
            return f"| **{name}** | ₹{fp:,.2f} | {conf:.0%} | {diff:+.1f}% |"
        else:
            return f"| **{name}** | N/A ({err[:40]}) | — | — |"

    r.append(model_row('DCF', dcf))
    r.append(model_row('Graham', graham))
    r.append(model_row('P/E Band (Median)', pe_band))
    r.append(model_row('EV/EBITDA', ev_ebitda))
    r.append(model_row('DDM', ddm))
    r.append("")

    # DCF Details
    if dcf.get('fair_price', 0) > 0:
        r.append("### 💰 DCF Model Details")
        r.append(f"- FCF Growth Rate: {dcf.get('fcf_growth', 0):.1f}%")
        r.append(f"- Discount Rate: {dcf.get('discount_rate', 12):.1f}%")
        r.append(f"- Terminal Growth: {dcf.get('terminal_growth', 4):.1f}%")
        r.append(f"- Projection: {dcf.get('projection_years', 5)} years")
        r.append("")

    # P/E Details
    if pe_band.get('fair_price', 0) > 0:
        r.append("### 📊 P/E Band Details")
        r.append(f"- Current P/E: {pe_band.get('current_pe', 0):.1f}x")
        r.append(f"- Median P/E: {pe_band.get('median_pe', 0):.1f}x")
        r.append(f"- P/E Range: {pe_band.get('min_pe', 0):.1f}x – {pe_band.get('max_pe', 0):.1f}x")
        r.append(f"- Fair Range: ₹{pe_band.get('fair_low', 0):,.0f} – ₹{pe_band.get('fair_high', 0):,.0f}")
        r.append("")

    # DDM Details
    if ddm.get('fair_price', 0) > 0:
        r.append("### 💎 DDM Details")
        r.append(f"- Latest DPS: ₹{ddm.get('latest_dps', 0):.2f}")
        r.append(f"- Dividend Growth: {ddm.get('div_growth', 0):.1f}%")
        r.append(f"- Payout Ratio: {ddm.get('payout_ratio', 0):.1f}%")
        r.append("")

    # Technical
    if technical and 'error' not in technical:
        r.append("## 📐 Technical Entry Zone")
        r.append(f"| Level | Price |")
        r.append(f"|-------|-------|")
        if technical.get('ma50'):
            r.append(f"| 50 DMA | ₹{technical['ma50']:,.2f} |")
        if technical.get('ma100'):
            r.append(f"| 100 DMA | ₹{technical['ma100']:,.2f} |")
        if technical.get('ma200'):
            r.append(f"| 200 DMA | ₹{technical['ma200']:,.2f} |")
        r.append(f"| 52W High | ₹{technical.get('high_52w', 0):,.2f} |")
        r.append(f"| 52W Low | ₹{technical.get('low_52w', 0):,.2f} |")
        r.append(f"| Support | ₹{technical.get('support', 0):,.2f} |")
        r.append(f"| Resistance | ₹{technical.get('resistance', 0):,.2f} |")
        r.append("")
        r.append("**Fibonacci Retracement Levels:**")
        for level, price in technical.get('fib_levels', {}).items():
            r.append(f"- {level}: ₹{price:,.2f}")
        r.append("")
        r.append(f"**🎯 Best Entry Zone: ₹{technical.get('best_entry_low', 0):,.2f} – ₹{technical.get('best_entry_high', 0):,.2f}**")
        r.append("")

    # Final Verdict
    r.append("## ✅ Final Verdict")
    r.append(f"**{composite['verdict']}** — Score: **{composite['score']}/10**")
    r.append("")
    if composite['upside'] > 10:
        r.append(f"The stock appears **undervalued** by ~{composite['upside']:.0f}%. ")
        r.append(f"Consider buying below **₹{composite['buy_below_price']:,.2f}** for a 15% margin of safety.")
    elif composite['upside'] > -5:
        r.append(f"The stock is trading near its **fair value**. Wait for a dip to ₹{composite['buy_below_price']:,.2f}.")
    else:
        r.append(f"The stock appears **overvalued** by ~{abs(composite['upside']):.0f}%. Avoid buying at current levels.")
    r.append("")
    r.append("---")
    r.append("*Disclaimer: This analysis is for educational purposes. Always do your own research before investing.*")

    return "\n".join(r)


# =============================================================================
# 6. MAIN ORCHESTRATOR
# =============================================================================

def run_full_analysis(data, ticker_for_technical=None):
    """
    Runs all 5 models + technical + composite.
    Returns dict with all results.
    """
    dcf = model_dcf(data)
    graham = model_graham(data)
    pe_band = model_pe_band(data)
    ev_ebitda = model_ev_ebitda(data)
    ddm = model_ddm(data)

    # Technical (needs ticker for Yahoo price fetch if screener source)
    if ticker_for_technical and data['source'] == 'screener':
        if not ticker_for_technical.endswith('.NS'):
            ticker_for_technical += '.NS'
        try:
            stock = yf.Ticker(ticker_for_technical)
            data['history'] = stock.history(period="1y")
            data['ticker'] = ticker_for_technical
        except:
            pass

    technical = analyze_technical_entry(data)
    composite = calculate_composite(data, dcf, graham, pe_band, ev_ebitda, ddm, technical)
    report = generate_valuation_report(data, dcf, graham, pe_band, ev_ebitda, ddm, technical, composite)

    return {
        'dcf': dcf, 'graham': graham, 'pe_band': pe_band,
        'ev_ebitda': ev_ebitda, 'ddm': ddm,
        'technical': technical, 'composite': composite,
        'report': report, 'data': data,
    }
