
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import io


def fetch_stock_details(ticker_symbol):
    """
    Fetches comprehensive data for a stock:
    - Historic Price (10y)
    - Financials (Income, Balance Sheet, Cash Flow)
    - Calculated Metrics (OPM, FCF, ROE, ROCE)
    """
    if not ticker_symbol.endswith(".NS") and not ticker_symbol.endswith(".BO"):
        ticker_symbol = f"{ticker_symbol}.NS"
        
    stock = yf.Ticker(ticker_symbol)
    
    # 1. Fetch History (10y for Price Analysis)
    history = stock.history(period="10y")
    
    # 2. Fetch Financials (Limit of yfinance is usually 4-5 years)
    fin = stock.financials
    bs = stock.balance_sheet
    cf = stock.cashflow
    
    # Transpose to get Years as Index
    fin = fin.T if fin is not None else pd.DataFrame()
    bs = bs.T if bs is not None else pd.DataFrame()
    cf = cf.T if cf is not None else pd.DataFrame()
    
    # Merge Financials into one DataFrame based on Date
    # We use 'outer' join to keep all dates
    fundamentals = pd.concat([fin, bs, cf], axis=1)
    
    # Ensure index is datetime
    fundamentals.index = pd.to_datetime(fundamentals.index)
    fundamentals = fundamentals.sort_index(ascending=True) # Oldest first
    
    # 3. Calculate Metrics
    metrics = pd.DataFrame(index=fundamentals.index)
    
    try:
        # Sales (Total Revenue)
        # yfinance keys vary, standardizing:
        if 'Total Revenue' in fundamentals.columns:
            metrics['Sales'] = fundamentals['Total Revenue']
        elif 'Total Revenue' in fin.columns: # Sometimes duplicate column names causing issues in concat
             metrics['Sales'] = fin['Total Revenue']
        else:
             metrics['Sales'] = np.nan

        # Operating Profit Margin (OPM) = Operating Income / Total Revenue
        if 'Operating Income' in fundamentals.columns and 'Sales' in metrics.columns:
            metrics['Operating Profit'] = fundamentals['Operating Income']
            metrics['OPM'] = (metrics['Operating Profit'] / metrics['Sales']) * 100
        else:
            metrics['OPM'] = np.nan

        # Free Cash Flow (FCF) = Operating Cash Flow - CapEx
        # CapEx is often 'Capital Expenditure' or 'Capital Expenditures' (negative value usually)
        # FCF = OCF + CapEx (if CapEx is negative) or OCF - CapEx (if positive)
        # yfinance usually reports CapEx as negative.
        if 'Operating Cash Flow' in fundamentals.columns:
            metrics['OCF'] = fundamentals['Operating Cash Flow']
            
            # Find CapEx column
            capex_col = next((c for c in fundamentals.columns if 'Capital Expenditure' in c), None)
            if capex_col:
                metrics['CapEx'] = fundamentals[capex_col]
                # If CapEx is negative (cash outflow), we add it. If positive (rare for CapEx), we subtract?
                # Standard: FCF is Cash from Ops less CapEx. 
                # yfinance 'Capital Expenditure' is usually negative number. So FCF = OCF + CapEx
                metrics['FCF'] = metrics['OCF'] + metrics['CapEx'] 
            else:
                metrics['FCF'] = metrics['OCF'] # Approx
        else:
            metrics['FCF'] = np.nan
            
        # Net Income
        if 'Net Income' in fundamentals.columns:
            metrics['Net Income'] = fundamentals['Net Income']
        else: 
            metrics['Net Income'] = np.nan

        # ROE = Net Income / Stockholders Equity
        # Stockholders Equity is in Balance Sheet
        equity_col = next((c for c in fundamentals.columns if 'Stockholders Equity' in c or 'Total Equity Gross Minority Interest' in c), None)
        if 'Net Income' in metrics.columns and equity_col:
             metrics['Equity'] = fundamentals[equity_col]
             metrics['ROE'] = (metrics['Net Income'] / metrics['Equity']) * 100
        else:
             metrics['ROE'] = np.nan

        # ROCE = EBIT / (Total Assets - Current Liabilities)
        # EBIT ~ Operating Income (Close enough)
        # Capital Employed = Total Assets - Current Liabilities
        assets_col = next((c for c in fundamentals.columns if 'Total Assets' in c), None)
        cliab_col = next((c for c in fundamentals.columns if 'Current Liabilities' in c), None) # often 'Total Current Liabilities' or 'Current Liabilities'
        
        if 'Operating Profit' in metrics.columns and assets_col and cliab_col:
             metrics['Result Assets'] = fundamentals[assets_col]
             metrics['Result CL'] = fundamentals[cliab_col]
             metrics['Capital Employed'] = metrics['Result Assets'] - metrics['Result CL']
             metrics['ROCE'] = (metrics['Operating Profit'] / metrics['Capital Employed']) * 100
        else:
             metrics['ROCE'] = np.nan
             
        # EPS for PEG
        if 'Basic EPS' in fundamentals.columns:
            metrics['EPS'] = fundamentals['Basic EPS']
        else:
             metrics['EPS'] = np.nan

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        
    return history, metrics


def perform_regression_analysis(history, metrics):
    """
    Performs Cubic Regression (Degree 3) for:
    1. Price vs OPM
    2. Price vs Sales
    3. Price vs FCF
    
    Returns:
        latest_estimates (dict): Estimate Price based on *current* metrics
        regression_data (dict): Data used for plotting (x, y, fitted_y)
    """
    estimates = {}
    reg_plots = {}
    
    # 1. Align Data: We need Average Price for the Year corresponding to the Metric
    # Metrics index is Date (e.g., 2023-03-31). We'll take the avg price of that year/quarter? 
    # Or simplified: Price on that specific date (or nearest).
    
    aligned_data = []
    
    for date, row in metrics.iterrows():
        # Find price nearest to this financial statement date
        # Use tolerance of 7 days to find nearest closing price
        try:
            # We look for price around the reporting date
            # Since reports are often lagged, maybe price *after* report? 
            # For simplicity: Price *at* reporting date (Market reaction to these fundamentals)
            price_series = history.loc[history.index.tz_localize(None) <= date.replace(tzinfo=None)]
            if not price_series.empty:
                # Take the last available price on or before report date
                price = price_series.iloc[-1]['Close'] 
                
                aligned_data.append({
                    'Date': date,
                    'Price': price,
                    'Sales': row.get('Sales', np.nan),
                    'OPM': row.get('OPM', np.nan),
                    'FCF': row.get('FCF', np.nan),
                    'EPS': row.get('EPS', np.nan)
                })
        except Exception as e:
            continue
            
    if not aligned_data:
        return {}, {}
        
    df_reg = pd.DataFrame(aligned_data).dropna()
    
    if len(df_reg) < 3:
        return {"Error": "Not enough data points for Cubic Regression (Need > 3 years)"}, {}

    # Helper for regression
    def fit_cubic(X, Y, current_X_value):
        try:
            # Dynamic Degree: Prevent overfitting if few data points
            # If N=4, Degree 3 fits perfectly. Use min(3, N-2) to allow some error/trend? 
            # This ensures we don't just "connect the dots" and get Estimate = Current Price
            N = len(X)
            # If N < 3, we can't really do cubic.
            # If N=4, max degree should be 2 to avoid perfect fit (N-1=3).
            # If N=5, max degree can be 3.
            if N < 4:
                degree = 1 # Linear if very few points
            else:
                degree = min(3, N-2)
            
            # Polyfit
            coeffs = np.polyfit(X, Y, degree)
            poly = np.poly1d(coeffs)
            
            # Measure Fit (R-squared)
            y_pred = poly(X)
            y_bar = np.mean(Y)
            ss_tot = np.sum((Y - y_bar)**2)
            ss_res = np.sum((Y - y_pred)**2)
            
            if ss_tot == 0:
                r_squared = 0 # Constant line
            else:
                r_squared = 1 - (ss_res / ss_tot)

            # Estimate
            est_price = poly(current_X_value)
            
            # Generate curve for plotting
            x_line = np.linspace(X.min(), X.max(), 100)
            y_line = poly(x_line)
            
            return est_price, x_line, y_line, poly, r_squared
        except:
            return None, None, None, None, 0

    # Current/Latest Metrics (Check for NaNs)
    latest_metrics = metrics.iloc[-1]
    
    # -- Model 1: Price vs Sales --
    if not np.isnan(latest_metrics.get('Sales')):
         est_p_sales, x_sales, y_sales, _, r2_sales = fit_cubic(df_reg['Sales'], df_reg['Price'], latest_metrics['Sales'])
         if est_p_sales:
            estimates['Sales_Based_Price'] = est_p_sales
            estimates['Sales_R2'] = r2_sales
            reg_plots['Sales'] = {'x': df_reg['Sales'], 'y': df_reg['Price'], 'tx': x_sales, 'ty': y_sales}

    # -- Model 2: Price vs FCF --
    if not np.isnan(latest_metrics.get('FCF')):
        est_p_fcf, x_fcf, y_fcf, _, r2_fcf = fit_cubic(df_reg['FCF'], df_reg['Price'], latest_metrics['FCF'])
        if est_p_fcf:
            estimates['FCF_Based_Price'] = est_p_fcf
            estimates['FCF_R2'] = r2_fcf
            reg_plots['FCF'] = {'x': df_reg['FCF'], 'y': df_reg['Price'], 'tx': x_fcf, 'ty': y_fcf}
        
    # -- Model 3: Price vs OPM --
    if not np.isnan(latest_metrics.get('OPM')):
        est_p_opm, x_opm, y_opm, poly_opm, r2_opm = fit_cubic(df_reg['OPM'], df_reg['Price'], latest_metrics['OPM'])
        if est_p_opm:
            estimates['OPM_Based_Price'] = est_p_opm
            estimates['OPM_R2'] = r2_opm
            estimates['OPM_Formula'] = str(poly_opm) # Store formula string
            reg_plots['OPM'] = {'x': df_reg['OPM'], 'y': df_reg['Price'], 'tx': x_opm, 'ty': y_opm}

    # -- Model 4: Price vs EPS --
    if not np.isnan(latest_metrics.get('EPS')):
        est_p_eps, x_eps, y_eps, poly_eps, r2_eps = fit_cubic(df_reg['EPS'], df_reg['Price'], latest_metrics['EPS'])
        if est_p_eps:
             estimates['EPS_Based_Price'] = est_p_eps
             estimates['EPS_R2'] = r2_eps
             estimates['EPS_Formula'] = str(poly_eps)
             reg_plots['EPS'] = {'x': df_reg['EPS'], 'y': df_reg['Price'], 'tx': x_eps, 'ty': y_eps}

    return estimates, reg_plots

def predict_price_trend(history):
    """
    Performs Cubic Regression on Weekly Price Data to predict future trend.
    Args:
        history (pd.DataFrame): 10y daily price history
    Returns:
        dict: containing 'dates', 'actual', 'trend', 'future_dates', 'future_trend', 'r2'
    """
    try:
        # 1. Resample to Weekly 'W' (End of week) to reduce noise
        # Use 'Close' price
        weekly_df = history['Close'].resample('W').last().dropna().reset_index()
        weekly_df.columns = ['Date', 'Close']
        
        # 2. Prepare X (Time) and Y (Price)
        # Use Date Ordinal to ensure correct spacing (works for Daily, Weekly, or Annual)
        weekly_df['Ordinal'] = weekly_df['Date'].apply(lambda d: d.toordinal())
        
        X = weekly_df['Ordinal'].values
        Y = weekly_df['Close'].values
        
        # 3. Cubic Fit & Dynamic Degree
        # Check data points count
        N = len(X)
        if N < 4:
            degree = 1
        else:
            degree = min(3, N-2) # Prevent overfitting on small N (e.g. Annual data)

        coeffs = np.polyfit(X, Y, degree)
        poly = np.poly1d(coeffs)
        
        # 4. Measure Fit (R2)
        y_pred = poly(X)
        y_bar = np.mean(Y)
        ss_tot = np.sum((Y - y_bar)**2)
        ss_res = np.sum((Y - y_pred)**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 5. Future Prediction (Next 1 Year = 52 Weeks)
        last_date = weekly_df['Date'].iloc[-1]
        
        # Generate Future Dates (Weekly steps for smoothness)
        future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, 53)]
        future_X = np.array([d.toordinal() for d in future_dates])
        
        future_Y = poly(future_X)
        
        return {
            'history_dates': weekly_df['Date'],
            'history_actual': Y,
            'history_trend': y_pred,
            'future_dates': future_dates,
            'future_trend': future_Y,
            'r2': r_squared,
            'equation': poly
        }

        
    except Exception as e:
        print(f"Error in price trend prediction: {e}")
        return None


def _process_fundamental_data(df):
    """
    Internal function to process the raw DataFrame from Excel/Text.
    """
    try:
        # Helper to find row index by keyword in first column
        def find_row(keyword):
            # Check if column 0 contains keyword (case-insensitive)
            # Use fillna('') to avoid errors on NaNs
            matches = df[df[0].astype(str).fillna('').str.contains(keyword, case=False, regex=False)]
            if not matches.empty:
                return matches.index[0]
            # Try regex if plain string failed (sometimes spaces vary)
            matches = df[df[0].astype(str).fillna('').str.contains(keyword, case=False, regex=True)]
            if not matches.empty:
                return matches.index[0]
            return None

        # --- 1. LOCATE SECTIONS ---
        pl_idx = find_row("PROFIT & LOSS")
        cf_idx = find_row("CASH FLOW")
        # bs_idx = find_row("BALANCE SHEET") # Not strictly needed for now if we don't use ROE/ROCE yet from sheet
        # Looking for PRICE row. It might be "PRICE" or "PRICE:"
        price_row_idx = find_row("PRICE") 
        
        if pl_idx is None or cf_idx is None:
            return None, None # Critical sections missing
            
        # --- 2. EXTRACT DATES ---
        # Dates are usually in the row immediately after section header
        date_row_idx = pl_idx + 1 # e.g., "Report Date", "Mar-16", "Mar-17"...
        dates = df.iloc[date_row_idx, 1:].values # Skip first col ("Report Date")
        
        # Convert specific formats like 'Mar-16' to datetime
        # Assuming format is %b-%y
        parsed_dates = []
        valid_cols = [] # Track valid columns index (1-based from Excel, 0-based in dates array)
        
        for i, d in enumerate(dates):
            try:
                if isinstance(d, str):
                    # Handle 'Mar-16' -> datetime
                    # Sometimes user locale might affect this, but standard Screener is consistent
                    dt = datetime.strptime(d.strip(), "%b-%y")
                elif isinstance(d, datetime):
                    dt = d
                else: 
                     continue
                parsed_dates.append(dt)
                valid_cols.append(i + 1) # +1 because dates array started from col 1
            except:
                pass
                
        if not parsed_dates:
            return None, None
            
        # --- 3. EXTRACT METRICS ---
        metrics_dict = {'Date': parsed_dates}
        
        def get_row_data(section_start_idx, section_end_idx, row_name):
            # Limit search to section
            if section_end_idx is None: 
                section_end_idx = len(df)
                
            section_df = df.iloc[section_start_idx:section_end_idx]
            
            # Search for row_name
            # Regex start/end anchor might be strict, let's just contains
            row = section_df[section_df[0].astype(str).fillna('').str.contains(row_name, case=False, regex=False)]
            
            if not row.empty:
                # Get values for valid columns
                values = row.iloc[0, valid_cols].values
                # Clean numeric
                return pd.to_numeric(values, errors='coerce')
            return np.full(len(parsed_dates), np.nan)

        # Financials range
        pl_end = cf_idx # P&L ends where Cash Flow or Balance Sheet starts roughly
        
        # Sales
        metrics_dict['Sales'] = get_row_data(pl_idx, pl_end, "Sales")
        
        # Operating Profit (Calculate or Find)
        # Try finding explicit 'Operating Profit' first
        op_prof = get_row_data(pl_idx, pl_end, "Operating Profit")
        
        # If 'Operating Profit' row is missing or all NaNs (Screener sometimes puts it in Quarters but allows it in P&L custom cols)
        # Check if we got valid data
        if np.all(np.isnan(op_prof)):
             # Calculate: Profit before tax + Interest + Depreciation - Other Income
             pbt = get_row_data(pl_idx, pl_end, "Profit before tax")
             interest = get_row_data(pl_idx, pl_end, "Interest")
             dep = get_row_data(pl_idx, pl_end, "Depreciation")
             other_inc = get_row_data(pl_idx, pl_end, "Other Income")
             
             # Fill NaNs with 0 for calculation, but keep NaN if everything is missing
             # If PBT exists, valid calculation is possible
             if not np.all(np.isnan(pbt)):
                 op_prof = np.nan_to_num(pbt) + np.nan_to_num(interest) + np.nan_to_num(dep) - np.nan_to_num(other_inc)
        
        metrics_dict['Operating Profit'] = op_prof
        
        # OPM
        # Avoid division by zero
        sales = metrics_dict['Sales']
        opm = np.full(len(parsed_dates), np.nan)
        mask = (sales != 0) & (~np.isnan(sales)) & (~np.isnan(metrics_dict['Operating Profit']))
        opm[mask] = (metrics_dict['Operating Profit'][mask] / sales[mask]) * 100
        metrics_dict['OPM'] = opm
            
        # Net Profit
        metrics_dict['Net Profit'] = get_row_data(pl_idx, pl_end, "Net profit")
        
        # Earnings Data? EPS?
        
        # --- CASH FLOW & FCF ---
        # Cash Flow ends at end of file or Price row
        cf_end = price_row_idx if price_row_idx else len(df)
        
        cfo = get_row_data(cf_idx, cf_end, "Cash from Operating Activity")
        cfi = get_row_data(cf_idx, cf_end, "Cash from Investing Activity")
        
        # FCF = CFO + CFI (Assuming CFI is mostly Capex and is negative)
        metrics_dict['FCF'] = cfo + cfi
        
        # Reserves (Balance Sheet)
        # Search for "Reserves" or "Reserves and Surplus" or "Other Equity"
        # Usually in Balance Sheet (bs_idx might be needed if we want to be strict, but search global is safer for text)
        metrics_dict['Reserves'] = get_row_data(0, None, "Reserves")
        
        # EPS (Earnings Per Share)
        # Often in P&L as "EPS" or "Basic EPS" or "Earnings Per Share"
        metrics_dict['EPS'] = get_row_data(0, None, "EPS")
        if np.all(np.isnan(metrics_dict['EPS'])):
             metrics_dict['EPS'] = get_row_data(0, None, "Earnings Per Share")
             
        # ROE (Return on Equity) -> Usually explicitly stated in custom Screener exports
        metrics_dict['ROE'] = get_row_data(0, None, "Return on Equity")
        
        # ROCE (Return on Capital Employed)
        metrics_dict['ROCE'] = get_row_data(0, None, "Return on Capital Employed")

        # --- PRICE ---
        if price_row_idx:
             # Price row is strictly below the header "PRICE" or on the same line?
             # User image shows "PRICE:" ... values on same line? Or next line?
             # Usually Screener export has "PRICE:" in col 0 and values in col 1, 2...
             # Let's check the row found
             price_row = df.iloc[price_row_idx]
             prices = price_row.iloc[valid_cols].values
             prices = pd.to_numeric(prices, errors='coerce')
        else:
            prices = np.full(len(parsed_dates), np.nan)
            
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.set_index('Date', inplace=True)
        
        # Handle Price History (metrics_df index is annual dates)
        # Create a history DF that looks like yfinance history (for compatibility)
        history_df = pd.DataFrame(index=metrics_df.index)
        history_df['Close'] = prices
        history_df = history_df.dropna() # Remove years with no price
        
        return history_df, metrics_df

    except Exception as e:
        print(f"Error parsing dataframe: {e}")
        return None, None


def parse_uploaded_excel(uploaded_file):
    """
    Parses a Screener.in format Excel file.
    """
    try:
        # Read Excel (No header initially to find sections)
        df = pd.read_excel(uploaded_file, header=None)
        return _process_fundamental_data(df)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return None, None


def parse_pasted_data(pasted_text):
    """
    Parses pasted Excel data (tab-separated).
    """
    try:
        # Read CSV from String (Tab Separated for Excel copy-paste)
        df = pd.read_csv(io.StringIO(pasted_text), sep='\t', header=None)
        return _process_fundamental_data(df)
    except Exception as e:
        print(f"Error reading pasted text: {e}")
        return None, None




def generate_fundamental_report(metrics, estimates, future_trend):
    """
    Generates a textual report based on the analysis.
    """
    report = []
    
    # 1. Growth Analysis - DETAILED
    try:
        def calculate_cagr(series):
            try:
                if series.empty or len(series) < 2: return 0
                start_val = series.iloc[0]
                end_val = series.iloc[-1]
                years = len(series) - 1
                if start_val <= 0 or end_val <= 0: return 0 # Avoid complex numbers
                return ((end_val / start_val) ** (1/years) - 1) * 100
            except:
                return 0

        # Calculate CAGRs
        sales_cagr = calculate_cagr(metrics['Sales'])
        profit_cagr = calculate_cagr(metrics['Net Profit'])
        reserves_cagr = calculate_cagr(metrics['Reserves'])
        fcf_cagr = calculate_cagr(metrics['FCF'])
        
        # Absolute Changes
        opm_change = metrics['OPM'].iloc[-1] - metrics['OPM'].iloc[0]
        roe_change = metrics['ROE'].iloc[-1] - metrics['ROE'].iloc[0]
        roce_change = metrics['ROCE'].iloc[-1] - metrics['ROCE'].iloc[0]
        
        report.append(f"### 🚀 Growth & Performance (Last {len(metrics)} Years)")
        report.append(f"- **Sales Growth (CAGR):** {sales_cagr:.1f}%")
        report.append(f"- **Net Profit Growth (CAGR):** {profit_cagr:.1f}%")
        report.append(f"- **Reserves Growth (CAGR):** {reserves_cagr:.1f}%")
        report.append(f"- **Free Cash Flow Growth (CAGR):** {fcf_cagr:.1f}%")
        report.append(f"- **OPM Change:** {'Improved' if opm_change > 0 else 'Declined'} by {abs(opm_change):.1f}%")
        report.append(f"- **ROE Change:** {'Improved' if roe_change > 0 else 'Declined'} by {abs(roe_change):.1f}%")
        report.append(f"- **ROCE Change:** {'Improved' if roce_change > 0 else 'Declined'} by {abs(roce_change):.1f}%")

    except Exception as e:
        report.append(f"Error calculating growth metrics: {str(e)}")

    # 2. Fair Value Analysis & Formulas
    report.append("\n### ⚖️ Fair Value Assessment (Cubic Regression)")
    best_r2 = 0
    best_model = ""
    best_price = 0
    
    for key, val in estimates.items():
        if "_R2" in key:
            model_name = key.replace("_R2", "")
            r2 = val
            price = estimates.get(f"{model_name}_Based_Price", 0)
            formula = estimates.get(f"{model_name}_Formula", "N/A")
            
            # Add details for each model
            report.append(f"**{model_name} Model:**")
            report.append(f"- Estimated Fair Value: **₹{price:.2f}**")
            report.append(f"- Correlation (R²): {r2:.2f}")
            # report.append(f"- Formula: `{formula}`") # Too long/complex for normal users? Maybe just show it exists.
            
            if val > best_r2:
                best_r2 = val
                best_model = model_name
                best_price = price
    
    report.append(f"\n✅ **Conclusion:** The **{best_model} Model** is the most reliable predictor (R²: {best_r2:.2f}), estimating a fair value of **₹{best_price:.2f}**.")

    # 3. Future Trend
    if future_trend:
        report.append("\n### 🔮 Future Price Trend (1 Year Projection)")
        # Ensure we have data
        current_price = future_trend['history_actual'][-1]
        future_price = future_trend['future_trend'][-1]
        
        # Calculate percentage change
        if current_price != 0:
            change = ((future_price - current_price) / current_price) * 100
        else:
            change = 0
            
        report.append(f"- **Current Trend Price:** ₹{current_price:.2f}")
        report.append(f"- **Projected Price (1 Year):** **₹{future_price:.2f}**")
        if change > 0:
            report.append(f"- **Potential Upside:** **+{change:.1f}%** 🚀")
        else:
            report.append(f"- **Potential Downside:** **{change:.1f}%** 📉")
            
        report.append(f"- *Trend Reliability (R²):* {future_trend['r2']:.2f}")
    
    report_text = "\n".join(report)
    return report_text


def get_nifty_market_regime():
    """
    Checks the trend of the benchmark index (Nifty 50 - ^NSEI) on weekly charts.
    Returns:
        status (str): "BULLISH", "BEARISH", "CAUTION"
        details (dict): current price and EMA values
    """
    try:
        nifty = yf.Ticker("^NSEI")
        df = nifty.history(period="1y", interval="1wk")
        if df.empty or len(df) < 40:
            return "UNKNOWN", {}
        
        import pandas_ta as ta
        df['EMA10'] = ta.ema(df['Close'], length=10)
        df['EMA40'] = ta.ema(df['Close'], length=40)
        
        curr_close = df['Close'].iloc[-1]
        ema10 = df['EMA10'].iloc[-1]
        ema40 = df['EMA40'].iloc[-1]
        
        if curr_close > ema10 and curr_close > ema40:
            status = "BULLISH"
        elif curr_close < ema40:
            status = "BEARISH"
        else:
            status = "CAUTION"
            
        return status, {
            'price': curr_close,
            'ema10': ema10,
            'ema40': ema40
        }
    except Exception as e:
        return "UNKNOWN", {'error': str(e)}


def analyze_screener_strategy(df, market_regime="BULLISH", use_trend_flt=True, use_vol_spike=True, use_wick_flt=True, use_squeeze=False):
    """
    Implements the Improved Swing Breakout Strategy (Weekly):
    1. Consolidation (T-6 to T-3): Volume contraction and price stability (drop <= 2%).
    2. Momentum Buildup (T-2 to T-1): Volume increase (T-1 > T-2 > T-3) and BB Breakout at T-1 (Close > Upper BB).
    3. Confirmation (Current T): Close > Open (Green) and Close > Close[1].
    4. Trend Filter (Optional): Close > 40-week EMA.
    5. Volume Spike (Optional): T-1 volume > 1.3x 20-week Volume MA.
    6. Wick Rejection Filter (Optional): T-1 Upper Wick < 35% of total range.
    7. Bollinger Band Squeeze (Optional): Require BB squeeze in consolidation window.
    """
    try:
        # Ensure we have enough data for 40-week EMA and indicators
        if len(df) < 45:
            return False, {"msg": "Insufficient data (need >= 45 weekly bars)"}, 0

        # Calculate Indicators
        import pandas_ta as ta
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['EMA_40'] = ta.ema(df['Close'], length=40)
        
        # Bollinger Bands (20, 2)
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is None or bb.empty:
            return False, {"msg": "Bollinger Bands calculation failed"}, 0
        bbu_col = next((c for c in bb.columns if c.startswith("BBU")), None)
        bbl_col = next((c for c in bb.columns if c.startswith("BBL")), None)
        bbm_col = next((c for c in bb.columns if c.startswith("BBM")), None)
        df['BB_Upper'] = bb[bbu_col]
        df['BB_Lower'] = bb[bbl_col]
        df['BB_Mid'] = bb[bbm_col] if bbm_col else df['Close'].rolling(20).mean()
        
        # Keltner Channel (20, 1.5 x ATR)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=20)
        df['ATR'] = atr
        kc_mid = ta.sma(df['Close'], length=20)
        df['KC_Upper'] = kc_mid + 1.5 * atr
        df['KC_Lower'] = kc_mid - 1.5 * atr
        
        # Squeeze Flag (BB inside KC)
        df['Squeeze'] = (df['BB_Lower'] > df['KC_Lower']) & (df['BB_Upper'] < df['KC_Upper'])
        
        # Volume MA
        df['Vol_MA'] = ta.sma(df['Volume'], length=20)

        # Slice last 7 weeks
        if len(df) < 7:
            return False, {"msg": "Not enough recent weekly data"}, df.iloc[-1]['Close']

        # Assign Rows
        w_curr = df.iloc[-1]   # T
        w_last = df.iloc[-2]   # T-1
        w_prev = df.iloc[-3]   # T-2
        cons_window = df.iloc[-7:-3] # T-6 to T-3

        current_price = w_curr['Close']

        # === CORE SWING BREAKOUT LOGIC (screener.pine) ===
        
        # 1. Consolidation (T-6 to T-3)
        vols = cons_window['Volume'].values
        vol_first_half = (vols[0] + vols[1]) / 2.0
        vol_last_half = (vols[2] + vols[3]) / 2.0
        is_vol_falling = vol_last_half < vol_first_half
        
        prices = cons_window['Close'].values
        price_change = (prices[3] - prices[0]) / prices[0]
        is_price_stable = price_change > -0.02
        
        condition_1 = is_vol_falling and is_price_stable
        if not condition_1:
            return False, {"msg": f"Consolidation check failed: Vol falling: {is_vol_falling}, Price change: {price_change*100:.1f}%"}, current_price

        # 2. Momentum Buildup (T-2 to T-1)
        vol_increase = (w_last['Volume'] > w_prev['Volume']) and (w_prev['Volume'] > vols[3])
        bb_breakout = w_last['Close'] > w_last['BB_Upper']
        
        condition_2 = vol_increase and bb_breakout
        if not condition_2:
            return False, {"msg": f"Momentum buildup failed: Vol increase: {vol_increase}, BB Breakout: {bb_breakout}"}, current_price

        # 3. Confirmation (Current T)
        is_green = w_curr['Close'] > w_curr['Open']
        is_above_prev = w_curr['Close'] > w_last['Close']
        
        condition_3 = is_green and is_above_prev
        if not condition_3:
            return False, {"msg": f"Weekly confirmation failed: Green: {is_green}, Above prev Close: {is_above_prev}"}, current_price

        # === EXPERT FILTERS ===

        # Filter A: Trend Filter
        trend_ok = True
        if use_trend_flt:
            trend_ok = (w_last['Close'] > w_last['EMA_40']) and (w_curr['Close'] > w_curr['EMA_40'])
        if not trend_ok:
            return False, {"msg": "Trend filter failed: Close below 40-week EMA"}, current_price

        # Filter B: Volume Spike
        vol_ratio = w_last['Volume'] / w_last['Vol_MA'] if w_last['Vol_MA'] > 0 else 0.0
        vol_spike_ok = True
        if use_vol_spike:
            vol_spike_ok = vol_ratio > 1.3
        if not vol_spike_ok:
            return False, {"msg": f"Volume spike filter failed: {vol_ratio:.2f}x (required > 1.3x)"}, current_price

        # Filter C: Wick Rejection (No large upper wick on breakout week T-1)
        breakout_range = w_last['High'] - w_last['Low']
        breakout_body_high = max(w_last['Open'], w_last['Close'])
        breakout_upper_wick = w_last['High'] - breakout_body_high
        wick_ok = True
        if use_wick_flt and breakout_range > 0:
            wick_ok = (breakout_upper_wick / breakout_range) <= 0.35
        if not wick_ok:
            return False, {"msg": f"Wick filter failed: Upper wick is {breakout_upper_wick/breakout_range:.1%} of candle range"}, current_price

        # Filter D: Squeeze during consolidation window (T-6 to T-3)
        squeeze_count = int(df['Squeeze'].iloc[-7:-3].sum())
        squeeze_ok = True
        if use_squeeze:
            squeeze_ok = squeeze_count >= 1
        if not squeeze_ok:
            return False, {"msg": "Squeeze filter failed: No squeeze active during consolidation phase"}, current_price

        # === TRADE PARAMETERS ===
        # Entry price is current weekly close
        trigger_price = current_price
        
        # Stop loss: 0.5% below breakout low
        stop_loss = w_last['Low'] * 0.995
        
        # Risk protection
        risk = trigger_price - stop_loss
        if risk <= 0:
            risk = trigger_price * 0.02
            stop_loss = trigger_price - risk
            
        target_1 = trigger_price + 2 * risk
        target_2 = trigger_price + 3 * risk
        
        triggered_this_week = True

        # === SETUP GRADING ===
        if market_regime == "BULLISH" and squeeze_count >= 1 and vol_ratio >= 1.5:
            grade = "A+"
            grade_desc = "A+ (Ultra High Conviction: Volatility Squeeze + Heavy Volume + Bull Market)"
        elif market_regime in ["BULLISH", "CAUTION"] and (squeeze_count >= 1 or vol_ratio >= 1.3):
            grade = "A"
            grade_desc = "A (High Conviction: Strong Breakout & Trend Alignment)"
        else:
            grade = "B"
            grade_desc = f"B (Medium Conviction: Correcting Market/No Squeeze. Vol: {vol_ratio:.1f}x)"

        # Return successful match details
        details = {
            'status': 'MATCH',
            'grade': grade,
            'grade_desc': grade_desc,
            'trigger_price': trigger_price,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'risk_price': risk,
            'vol_ratio': vol_ratio,
            'squeeze_weeks': squeeze_count,
            'triggered': triggered_this_week,
            'current_price': current_price,
            'ema10': w_curr['EMA_10'],
            'ema40': w_curr['EMA_40'],
            'bb_upper': w_curr['BB_Upper'],
            'bb_lower': w_curr['BB_Lower'],
            'kc_upper': w_curr['KC_Upper'],
            'kc_lower': w_curr['KC_Lower']
        }
        return True, details, current_price

    except Exception as e:
        return False, {"msg": f"Error during scanning: {str(e)}"}, 0


def analyze_pullback_strategy(df, market_regime="BULLISH"):
    """
    Implements a low-risk pullback-to-support swing trading strategy:
    1. Long-Term Trend Filter: Weekly Close must be above the 40-week EMA.
    2. Proximity Check: Price is within -1.5% to +4% of the 40-week EMA or 20-week SMA (middle BB).
    3. Volume Check: Weekly Volume is below the 20-week Volume MA (low selling pressure).
    4. Bullish Reversal: Weekly close > open (green candle) OR weekly close > last week's close or Hammer.
    
    Returns:
        is_match (bool): True if pullback setup matches
        details (dict): Dict of trade parameters (grade, trigger, stop loss, targets) or error message
        price (float): Current price
    """
    try:
        # Ensure we have enough data for 40-week EMA and indicators
        if len(df) < 45:
            return False, {"msg": "Insufficient data (need >= 45 weekly bars)"}, 0

        # Calculate Indicators
        import pandas_ta as ta
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['EMA_40'] = ta.ema(df['Close'], length=40)
        
        # Bollinger Bands (needed for 20 SMA / Middle BB)
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is None or bb.empty:
            return False, {"msg": "Bollinger Bands calculation failed"}, 0
        bbm_col = next((c for c in bb.columns if c.startswith("BBM")), None)
        df['BB_Mid'] = bb[bbm_col] if bbm_col else df['Close'].rolling(20).mean()
        
        # Volume MA
        df['Vol_MA'] = ta.sma(df['Volume'], length=20)

        w_curr = df.iloc[-1]   # T
        w_last = df.iloc[-2]   # T-1
        current_price = w_curr['Close']

        # === 1. LONG-TERM TREND FILTER ===
        # Price must be above the 40-week EMA (primary uptrend definition)
        if current_price < w_curr['EMA_40']:
            return False, {"msg": "Trend failed: Close is below 40-week EMA (Not in uptrend)"}, current_price

        # === 2. SUPPORT PROXIMITY CHECK ===
        # Calculate percentage distance from 40-week EMA and 20-week SMA (BB Mid)
        dist_ema40 = (current_price - w_curr['EMA_40']) / w_curr['EMA_40']
        dist_bbmid = (current_price - w_curr['BB_Mid']) / w_curr['BB_Mid']
        
        # Check if price is within -1.5% to +4.0% of either key support line
        on_ema40 = -0.015 <= dist_ema40 <= 0.04
        on_bbmid = -0.015 <= dist_bbmid <= 0.04
        
        if not (on_ema40 or on_bbmid):
            return False, {"msg": f"Price not near support (EMA40 dist: {dist_ema40:.1%}, BBMid dist: {dist_bbmid:.1%})"}, current_price

        # === 3. DRY VOLUME CHECK (LOW SELLING PRESSURE) ===
        # Weekly volume must be below average (no institutional dump)
        vol_ratio = w_curr['Volume'] / w_curr['Vol_MA'] if w_curr['Vol_MA'] > 0 else 0
        if vol_ratio > 1.1:
            return False, {"msg": f"Pullback volume too high: {vol_ratio:.2f}x (volume must be dry, < 1.1x)"}, current_price

        # === 4. BULLISH REVERSAL SIGNALS ===
        is_green = w_curr['Close'] > w_curr['Open']
        above_prev = w_curr['Close'] > w_last['Close']
        # Hammer candle check: lower shadow is at least 1.5x body size
        body_size = abs(w_curr['Close'] - w_curr['Open'])
        lower_shadow = min(w_curr['Close'], w_curr['Open']) - w_curr['Low']
        is_hammer = lower_shadow > 1.5 * body_size if body_size > 0 else lower_shadow > current_price * 0.01
        
        reversal = is_green or above_prev or is_hammer
        if not reversal:
            return False, {"msg": "No bullish reversal signature (Not green/no hammer)"}, current_price

        # === 5. TRADE PARAMETERS CALCULATION ===
        # Support line price determines stop loss basis
        support_price = w_curr['EMA_40'] if on_ema40 else w_curr['BB_Mid']
        
        # Stop Loss: 3% below the support price or 1% below the current weekly low (whichever is lower)
        stop_loss = min(support_price * 0.97, w_curr['Low'] * 0.99)
        
        # Prevent invalid risk values
        risk = current_price - stop_loss
        if risk <= 0:
            risk = current_price * 0.02
            stop_loss = current_price - risk
            
        target_1 = current_price + 2 * risk
        target_2 = current_price + 3 * risk
        
        # === 6. SETUP GRADING ===
        # Grade A+: Market is Bullish, volume is very dry (< 0.7x MA), bounce is green
        # Grade A: Market is Bullish/Caution, volume is dry (< 0.9x MA)
        # Grade B: Normal pullback near support with volume < 1.1x MA
        if market_regime == "BULLISH" and vol_ratio < 0.7 and is_green:
            grade = "A+"
            grade_desc = "A+ (Ultra Pullback: Bull Market + Extreme Vol contraction on Support)"
        elif market_regime in ["BULLISH", "CAUTION"] and vol_ratio < 0.9:
            grade = "A"
            grade_desc = "A (High Quality Pullback: Dry volume bounce on major support)"
        else:
            grade = "B"
            grade_desc = f"B (Medium Quality Pullback: Volume dry: {vol_ratio:.1f}x)"

        # Return successful pullback match details
        details = {
            'status': 'MATCH',
            'grade': grade,
            'grade_desc': grade_desc,
            'trigger_price': current_price, # Reversal is active, enter at current close
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'risk_price': risk,
            'vol_ratio': vol_ratio,
            'squeeze_weeks': 0,
            'triggered': True,
            'current_price': current_price,
            'ema10': w_curr['EMA_10'],
            'ema40': w_curr['EMA_40'],
            'bb_mid': w_curr['BB_Mid'],
            'support_line': support_price
        }
        return True, details, current_price

    except Exception as e:
        return False, {"msg": f"Error during pullback scanning: {str(e)}"}, 0


