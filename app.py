import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os # For file checking
import analysis_engine as ae # Custom Analysis Module
import valuation_engine as ve # Deep Dive Valuation Module
import screener_engine as se # Accumulation Screener Module

# --- CONFIGURATION ---
st.set_page_config(page_title="MomentumDoctor", layout="wide", page_icon="📈")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 20px;
        color: #333 !important; /* Force dark text for readability */
    }
    .stMetric {
        background-color: #f0f2f6; 
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .stMetric label {
        color: #555 !important; /* Label color */
    }
    div[data-testid="stExpander"] {
        border: 1px solid #ddd;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA ENGINE (The Fetcher) ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to speed up the app
def fetch_weekly_data(ticker):
    """Fetches 1 year of weekly data for a given stock."""
    try:
        # Append .NS if not present (assuming NSE India)
        if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            ticker = f"{ticker}.NS"
            
        stock = yf.Ticker(ticker)
        # Fetch 2 years to ensure enough data for indicators
        df = stock.history(period="2y", interval="1wk")
        
        if df.empty:
            return None
            
        return df
    except Exception as e:
        return None

# --- 2. LOGIC ENGINE (The Brain) ---
def analyze_stock(df, buy_price=0, buy_date=None):
    """
    Applies the Momentum Doctor Logic:
    1. Bollinger Bands (20, 2)
    2. Volume Moving Average (20)
    3. Trailing Stop & Hard Stop
    """
    if df is None or len(df) < 21:
        return df, "Insufficient Data", "gray", 0, 0

    # -- A. Calculate Indicators (Robust Fix) --
    try:
        # Calculate Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        
        if bb is None or bb.empty:
            return df, "Indicator Error", "gray", 0, 0

        # DYNAMIC COLUMN FINDER: Finds columns starting with BBU (Upper) and BBL (Lower)
        # This prevents the KeyError if the library names them differently
        bbu_col = next((c for c in bb.columns if c.startswith("BBU")), None)
        bbl_col = next((c for c in bb.columns if c.startswith("BBL")), None)

        if not bbu_col or not bbl_col:
            return df, "Indicator Missing", "gray", 0, 0

        # Attach to main dataframe
        df = pd.concat([df, bb], axis=1)
        df['BB_Upper'] = df[bbu_col]
        df['BB_Lower'] = df[bbl_col]
        
        # Volume MA
        df['Vol_MA'] = ta.sma(df['Volume'], length=20)
        
    except Exception as e:
        # If calculation fails entirely
        return df, f"Math Error: {e}", "gray", 0, 0
    
    # Get current and previous week data
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal = "HOLD / NEUTRAL"
    color = "white"
    
    # -- B. Price Checks (Profit Protect) --
    current_price = curr['Close']
    
    # Calculate Highest Price SINCE Entry (if Buy Date provided)
    highest_price = current_price
    if buy_date:
        try:
            # Filter data after buy date - ROBUST TIMEZONE HANDLER
            buy_dt = pd.to_datetime(buy_date)
            
            # Check if dataframe index is timezone-aware
            if df.index.tz is not None:
                # If dataframe is aware, localize buy_date to match
                if buy_dt.tzinfo is None:
                    buy_dt = buy_dt.tz_localize(df.index.tz)
                else:
                    buy_dt = buy_dt.tz_convert(df.index.tz)
            else:
                # If dataframe is naive, make sure buy_date is naive
                if buy_dt.tzinfo is not None:
                    buy_dt = buy_dt.tz_localize(None)
            
            mask = df.index >= buy_dt
            if mask.any():
                highest_price = df.loc[mask]['High'].max()
        except Exception as e:
            pass # Fallback to current price if date math fails
    
    # -- C. The Logic Tree --
    
    # 1. HARD STOP (Maroon)
    if buy_price > 0 and current_price < (buy_price * 0.93): # 7% below buy price
        signal = "HARD STOP (Capital Preservation)"
        color = "#800000" # Maroon
        
    # 2. TRAILING STOP (Red)
    elif buy_price > 0 and current_price < (highest_price * 0.93): # 7% below Peak
        signal = "EXIT (Trailing Stop Hit)"
        color = "#FF0000" # Red
        
    # 3. MOMENTUM KILL (Light Red/Orange)
    elif curr['Close'] < prev['Low']:
        signal = "WARNING (Momentum Broken)"
        color = "#FFA07A" # Light Salmon/Red
        
    # 4. MOMENTUM IGNITION (Green) - The "Buy" Signal
    # Logic: Close > Prev High AND Close > Upper BB AND Volume > 1.5x Avg
    elif (curr['Close'] > prev['High']) and \
         (curr['Close'] > curr['BB_Upper']) and \
         (curr['Volume'] > (1.5 * curr['Vol_MA'])):
        signal = "BUY MORE / ENTRY (Momentum Ignition)"
        color = "#00FF00" # Lime Green
        
    return df, signal, color, current_price, highest_price

# --- 3. VISUALIZATION ENGINE (The Radiologist View) ---
def plot_chart(df, ticker, signal):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name=ticker))

    # Bollinger Bands
    # Bollinger Bands (Check if they exist)
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1), name='Upper BB'))
    if 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1), name='Lower BB'))

    fig.update_layout(
        title=f"{ticker} - Weekly Chart ({signal})",
        xaxis_title="Date (Weeks)",
        yaxis_title="Price",
        height=500,
        template="plotly_dark"
    )
    return fig

# --- 4. MAIN APP INTERFACE ---

st.title("👨‍⚕️ MomentumDoctor: Swing Trading System")
st.markdown("Returns are made by sitting, not trading. But sitting in the right stocks.")

# --- PORTFOLIO PERSISTENCE HELPER ---
PORTFOLIO_FILE = "portfolio.csv"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            return pd.read_csv(PORTFOLIO_FILE).to_dict('records')
        except:
            return []
    return []

def save_portfolio(data):
    if data:
        df = pd.DataFrame(data)
        df.to_csv(PORTFOLIO_FILE, index=False)
    else:
        # If empty, create empty file with headers or just delete
        if os.path.exists(PORTFOLIO_FILE):
            os.remove(PORTFOLIO_FILE)

if "portfolio_data" not in st.session_state:
    st.session_state.portfolio_data = load_portfolio()

# --- TABS FOR DASHBOARD vs SCREENER ---
# Move Management to a dedicated Expander or keep in sidebar/Main
# Using Main Page for better width with data_editor

tab1, tab2, tab5, tab3, tab6, tab7, tab4 = st.tabs(["📊 My Portfolio", "📝 Manage Holdings", "📊 Fundamental Report", "🔍 Market Screener", "🔎 Accumulation Scanner", "📖 Playbook & Risk Rules", "🧠 Detailed Analysis"])

# === TAB 4: DETAILED STOCK ANALYSIS ===
with tab4:
    st.header("🧠 Deep Dive Valuation Analysis")
    st.write("Comprehensive fair value analysis using **5 valuation models**, technical entry zones, and a composite verdict.")

    # --- DATA SOURCE TOGGLE ---
    data_source = st.radio("Select Data Source:", ["📡 Yahoo Finance (Auto-fetch)", "📋 Paste Screener.in Data Sheet"],
                           horizontal=True, key="val_source")

    val_data = None
    run_valuation = False
    val_ticker_for_tech = ""

    if data_source == "📡 Yahoo Finance (Auto-fetch)":
        col_s, col_b = st.columns([3, 1])
        with col_s:
            val_ticker = st.text_input("Enter NSE Stock Symbol:", value="RELIANCE", key="val_ticker")
        with col_b:
            st.write("")
            st.write("")
            run_valuation = st.button("🔬 Run Full Valuation")
        if run_valuation:
            with st.spinner(f"Fetching data for {val_ticker} from Yahoo Finance..."):
                val_data = ve.fetch_yahoo_data(val_ticker)
                val_ticker_for_tech = val_ticker
                if val_data is None:
                    st.error("Could not fetch data. Check the symbol and try again.")
                elif val_data['n_years'] < 3:
                    st.warning(f"⚠️ Only {val_data['n_years']} years of financial data available. Results may be less reliable. Use Screener.in paste for 10-year data.")
    else:
        st.info("📌 Open your Screener.in Excel export → Go to **'Data Sheet'** tab → Select All → Copy → Paste below.")
        val_ticker_for_tech_input = st.text_input("Stock Symbol (for price chart):", value="TCS", key="val_tech_ticker")
        val_pasted = st.text_area("Paste Data Sheet content here:", height=300, key="val_paste")
        run_valuation = st.button("🔬 Run Full Valuation", key="val_run_screener")
        if run_valuation and val_pasted:
            with st.spinner("Parsing Screener.in data..."):
                try:
                    val_data = ve.parse_screener_datasheet(val_pasted)
                    val_ticker_for_tech = val_ticker_for_tech_input
                except Exception as e:
                    st.error(f"Parsing error: {e}")

    # --- DISPLAY RESULTS ---
    if val_data is not None and run_valuation:
        with st.spinner("Running all 5 valuation models + technical analysis..."):
            results = ve.run_full_analysis(val_data, ticker_for_technical=val_ticker_for_tech)

        comp = results['composite']
        cp = val_data['meta']['current_price']

        # --- EXECUTIVE SUMMARY ---
        st.divider()
        st.subheader("🎯 Executive Summary")
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.metric("Verdict", comp['verdict'])
        with ec2:
            st.metric("Composite Score", f"{comp['score']}/10")
        with ec3:
            st.metric("Fair Value", f"₹{comp['composite_fair_price']:,.2f}",
                       f"{comp['upside']:+.1f}%")
        with ec4:
            st.metric("Buy Below (15% MoS)", f"₹{comp['buy_below_price']:,.2f}")

        # --- ALL 5 MODEL CARDS ---
        st.divider()
        st.subheader("🧮 Valuation Models")

        model_names = ['DCF', 'Graham', 'P/E Band', 'EV/EBITDA', 'DDM']
        model_keys = ['dcf', 'graham', 'pe_band', 'ev_ebitda', 'ddm']
        model_icons = ['💰', '📐', '📊', '🏢', '💎']

        m_cols = st.columns(5)
        for i, (name, key, icon) in enumerate(zip(model_names, model_keys, model_icons)):
            m = results[key]
            fp = m.get('fair_price', 0)
            conf = m.get('confidence', 0)
            err = m.get('error', '')
            with m_cols[i]:
                if fp > 0:
                    diff = ((fp - cp) / cp) * 100
                    st.metric(f"{icon} {name}", f"₹{fp:,.0f}", f"{diff:+.1f}%")
                    st.caption(f"Confidence: {conf:.0%}")
                else:
                    st.metric(f"{icon} {name}", "N/A")
                    st.caption(err[:50] if err else "—")

        # --- COMPARISON CHART ---
        st.divider()
        st.subheader("📊 Model Comparison")
        bar_names = []
        bar_prices = []
        bar_colors = []
        for name, key in zip(model_names, model_keys):
            fp = results[key].get('fair_price', 0)
            if fp > 0:
                bar_names.append(name)
                bar_prices.append(fp)
                bar_colors.append('#4CAF50' if fp > cp else '#FF5252')

        if bar_names:
            bar_names.append("Current Price")
            bar_prices.append(cp)
            bar_colors.append('#2196F3')
            bar_names.append("Buy Below")
            bar_prices.append(comp['buy_below_price'])
            bar_colors.append('#FF9800')

            fig_bar = go.Figure(go.Bar(
                x=bar_names, y=bar_prices,
                marker_color=bar_colors,
                text=[f"₹{p:,.0f}" for p in bar_prices],
                textposition='outside'
            ))
            fig_bar.update_layout(
                title="Fair Value Estimates vs Current Price",
                yaxis_title="Price (₹)", template="plotly_dark", height=450,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- TECHNICAL ENTRY ZONE ---
        tech = results['technical']
        if tech and 'error' not in tech:
            st.divider()
            st.subheader("📐 Technical Entry Zone")
            t1, t2 = st.columns([2, 1])
            with t2:
                st.markdown("**Moving Averages**")
                if tech.get('ma50'):
                    st.metric("50 DMA", f"₹{tech['ma50']:,.2f}")
                if tech.get('ma100'):
                    st.metric("100 DMA", f"₹{tech['ma100']:,.2f}")
                if tech.get('ma200'):
                    st.metric("200 DMA", f"₹{tech['ma200']:,.2f}")
                st.metric("52W High", f"₹{tech['high_52w']:,.2f}")
                st.metric("52W Low", f"₹{tech['low_52w']:,.2f}")
                st.success(f"🎯 Best Entry: ₹{tech['best_entry_low']:,.0f} – ₹{tech['best_entry_high']:,.0f}")

            with t1:
                hist = tech.get('history')
                if hist is not None and not hist.empty:
                    fig_tech = go.Figure()
                    fig_tech.add_trace(go.Candlestick(
                        x=hist.index, open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'], name='Price'))
                    if tech.get('ma50'):
                        fig_tech.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='50 DMA',
                                                      line=dict(color='yellow', width=1)))
                    if tech.get('ma100'):
                        fig_tech.add_trace(go.Scatter(x=hist.index, y=hist['MA100'], name='100 DMA',
                                                      line=dict(color='orange', width=1)))
                    if tech.get('ma200'):
                        fig_tech.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], name='200 DMA',
                                                      line=dict(color='red', width=1)))
                    # Fibonacci levels
                    for lvl_name, lvl_val in tech.get('fib_levels', {}).items():
                        fig_tech.add_hline(y=lvl_val, line_dash="dot", line_color="gray",
                                          annotation_text=f"Fib {lvl_name}")
                    # Support/Resistance
                    fig_tech.add_hline(y=tech['support'], line_color="green", line_dash="dash",
                                      annotation_text="Support")
                    fig_tech.add_hline(y=tech['resistance'], line_color="red", line_dash="dash",
                                      annotation_text="Resistance")

                    fig_tech.update_layout(title="Price with Technical Levels",
                                          template="plotly_dark", height=550, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_tech, use_container_width=True)

        # --- P/E BAND CHART ---
        pe = results['pe_band']
        if pe.get('fair_price', 0) > 0:
            st.divider()
            st.subheader("📊 P/E Band Analysis")
            pe_c1, pe_c2 = st.columns([1, 1])
            with pe_c1:
                st.metric("Current P/E", f"{pe['current_pe']:.1f}x")
                st.metric("Median P/E", f"{pe['median_pe']:.1f}x")
                st.metric("P/E Range", f"{pe['min_pe']:.1f}x – {pe['max_pe']:.1f}x")
                st.metric("Fair Price Range", f"₹{pe['fair_low']:,.0f} – ₹{pe['fair_high']:,.0f}")
            with pe_c2:
                fig_pe = go.Figure()
                fig_pe.add_trace(go.Bar(
                    x=['Min P/E', 'Median P/E', 'Current P/E', 'Max P/E'],
                    y=[pe['min_pe'], pe['median_pe'], pe['current_pe'], pe['max_pe']],
                    marker_color=['green', 'blue', 'yellow', 'red'],
                    text=[f"{v:.1f}x" for v in [pe['min_pe'], pe['median_pe'], pe['current_pe'], pe['max_pe']]],
                    textposition='outside'
                ))
                fig_pe.update_layout(title="P/E Band", template="plotly_dark", height=350)
                st.plotly_chart(fig_pe, use_container_width=True)

        # --- FULL REPORT ---
        st.divider()
        st.subheader("📝 Full Valuation Report")
        st.markdown(results['report'])

        # --- KEEP EXISTING REGRESSION & TREND (Yahoo mode only) ---
        if data_source == "📡 Yahoo Finance (Auto-fetch)":
            with st.expander("📈 Legacy Analysis (Regression + Price Trend)", expanded=False):
                history, metrics = ae.fetch_stock_details(val_ticker)
                if not metrics.empty:
                    estimates, reg_plots = ae.perform_regression_analysis(history, metrics)
                    if "Error" not in estimates:
                        c1, c2, c3 = st.columns(3)
                        curr_price = history.iloc[-1]['Close']
                        with c1:
                            est = estimates.get('Sales_Based_Price', 0)
                            r2 = estimates.get('Sales_R2', 0)
                            diff_val = ((est - curr_price) / curr_price) * 100
                            st.metric("Est. Price (Sales)", f"₹{est:.2f}", f"{diff_val:.1f}%")
                        with c2:
                            est = estimates.get('FCF_Based_Price', 0)
                            r2 = estimates.get('FCF_R2', 0)
                            diff_val = ((est - curr_price) / curr_price) * 100
                            st.metric("Est. Price (FCF)", f"₹{est:.2f}", f"{diff_val:.1f}%")
                        with c3:
                            est = estimates.get('OPM_Based_Price', 0)
                            r2 = estimates.get('OPM_R2', 0)
                            diff_val = ((est - curr_price) / curr_price) * 100
                            st.metric("Est. Price (OPM)", f"₹{est:.2f}", f"{diff_val:.1f}%")

                    trend_data = ae.predict_price_trend(history)
                    if trend_data:
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(x=trend_data['history_dates'], y=trend_data['history_actual'],
                                                       name='Actual', line=dict(color='gray', width=1)))
                        fig_trend.add_trace(go.Scatter(x=trend_data['history_dates'], y=trend_data['history_trend'],
                                                       name='Trend', line=dict(color='yellow', width=2)))
                        fig_trend.add_trace(go.Scatter(x=trend_data['future_dates'], y=trend_data['future_trend'],
                                                       name='Prediction', line=dict(color='cyan', width=2, dash='dash')))
                        fig_trend.update_layout(title="Price Trend Prediction", template="plotly_dark", height=400)
                        st.plotly_chart(fig_trend, use_container_width=True)


# === TAB 5: FUNDAMENTAL REPORT (PASTE) ===
with tab5:
    st.header("📊 Detailed Fundamental Report")
    st.caption("Paste your Excel data directly below (Select All in Excel -> Copy -> Paste).")
    
    # Text Area for Paste
    pasted_data = st.text_area("Paste Excel Data Here", height=300, help="Copy the entire sheet from Excel and paste it here.")
    
    if st.button("Generate Report"):
        if pasted_data:
            try:
                # Parse Data
                with st.spinner("Analyzing Financial Statements..."):
                    history_df, metrics_df = ae.parse_pasted_data(pasted_data)
                    
                if history_df is not None and not history_df.empty:
                    # 1. Run Regression
                    estimates, reg_plots = ae.perform_regression_analysis(history_df, metrics_df)
                    
                    # 2. Run Future Trend
                    trend_data = ae.predict_price_trend(history_df)
                    
                    # 3. Generate Text Report
                    report_text = ae.generate_fundamental_report(metrics_df, estimates, trend_data)
                    
                    # --- DISPLAY REPORT ---
                    
                    # A. Summary Metrics (Top Row)
                    col1, col2, col3 = st.columns(3)
                    latest = metrics_df.iloc[-1]
                    prev = metrics_df.iloc[0]
                    sales_growth = (latest['Sales'] - prev['Sales']) / prev['Sales'] * 100
                    
                    col1.metric("Sales Growth (10y)", f"{sales_growth:.1f}%")
                    col2.metric("Current OPM", f"{latest['OPM']:.1f}%")
                    col3.metric("Current Price", f"₹{history_df['Close'].iloc[-1]:.2f}")
                    
                    st.divider()
                    
                    # B. The REPORT
                    st.subheader("📝 Analyst Report")
                    st.markdown(report_text)
                    
                    st.divider()
                    
                    st.divider()
                    
                    # C. Visualization Suite
                    st.subheader("📊 Growth & Efficiency Trends")
                    
                    # 1. Sales vs Profit (Growth)
                    tab_g1, tab_g2, tab_g3 = st.tabs(["Sales & Net Profit", "Margins & Efficiency", "Reserves & Cash Flow"])
                    
                    with tab_g1:
                        fig_growth = go.Figure()
                        fig_growth.add_trace(go.Bar(x=metrics_df.index, y=metrics_df.get('Sales',[]), name='Sales', marker_color='blue'))
                        fig_growth.add_trace(go.Bar(x=metrics_df.index, y=metrics_df.get('Net Profit',[]), name='Net Profit', marker_color='green'))
                        fig_growth.update_layout(title="Sales vs Net Profit Growth", barmode='group', template="plotly_dark", height=400)
                        st.plotly_chart(fig_growth, use_container_width=True)
                        
                    with tab_g2:
                        fig_eff = go.Figure()
                        if 'ROE' in metrics_df:
                            fig_eff.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df['ROE'], name='ROE %', line=dict(color='cyan', width=2)))
                        if 'ROCE' in metrics_df:
                            fig_eff.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df['ROCE'], name='ROCE %', line=dict(color='magenta', width=2)))
                        if 'OPM' in metrics_df:
                            fig_eff.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df['OPM'], name='OPM %', line=dict(color='yellow', width=2, dash='dot')))
                        fig_eff.update_layout(title="Efficiency Trends (ROE, ROCE, OPM)", template="plotly_dark", height=400)
                        st.plotly_chart(fig_eff, use_container_width=True)

                    with tab_g3:
                        fig_cash = go.Figure()
                        if 'Reserves' in metrics_df:
                            fig_cash.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df['Reserves'], name='Reserves', fill='tozeroy'))
                        if 'FCF' in metrics_df:
                            fig_cash.add_trace(go.Bar(x=metrics_df.index, y=metrics_df['FCF'], name='Free Cash Flow', marker_color='lightgreen'))
                        fig_cash.update_layout(title="Reserves vs Free Cash Flow", template="plotly_dark", height=400)
                        st.plotly_chart(fig_cash, use_container_width=True)

                    st.divider()

                    # C. Valuation Correlation Charts
                    st.subheader("🧠 Valuation Correlations (Regression Models)")
                    
                    c_reg1, c_reg2 = st.columns(2)
                    
                    with c_reg1:
                        # Price vs OPM
                        if 'OPM' in reg_plots:
                             p_data = reg_plots['OPM']
                             fig_reg = go.Figure()
                             fig_reg.add_trace(go.Scatter(x=p_data['x'], y=p_data['y'], mode='markers', name='Actual'))
                             fig_reg.add_trace(go.Scatter(x=p_data['tx'], y=p_data['ty'], mode='lines', name='Cubic Fit'))
                             fig_reg.update_layout(title="Price vs OPM %", xaxis_title="OPM %", yaxis_title="Price", template="plotly_dark", height=350)
                             st.plotly_chart(fig_reg, use_container_width=True)

                    with c_reg2:
                        # Price vs EPS (New)
                        if 'EPS' in reg_plots:
                             p_data = reg_plots['EPS']
                             fig_reg = go.Figure()
                             fig_reg.add_trace(go.Scatter(x=p_data['x'], y=p_data['y'], mode='markers', name='Actual'))
                             fig_reg.add_trace(go.Scatter(x=p_data['tx'], y=p_data['ty'], mode='lines', name='Cubic Fit'))
                             fig_reg.update_layout(title="Price vs EPS", xaxis_title="EPS", yaxis_title="Price", template="plotly_dark", height=350)
                             st.plotly_chart(fig_reg, use_container_width=True)
                    
                    # Future Trend
                    if trend_data:
                        st.subheader("🔮 Price Trend Projection")
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(x=trend_data['history_dates'], y=trend_data['history_actual'], name='History'))
                        fig_trend.add_trace(go.Scatter(x=trend_data['future_dates'], y=trend_data['future_trend'], name='Projection (1y)', line=dict(dash='dash', color='cyan')))
                        fig_trend.update_layout(title="1-Year Price Prediction", template="plotly_dark")
                        st.plotly_chart(fig_trend, use_container_width=True)
                        
                else:
                    st.error("Could not parse the pasted data. Please ensure it is a standard Screener.in export format.")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please paste some data first!")


# === TAB 3: MARKET SCREENER ===
# === TAB 2: MANAGE HOLDINGS ===
with tab2:
    st.header("Manage Holdings")
    st.info("💡 Tip: You can edit values directly in the table below. Add new rows at the bottom. Select checkbox to delete.")
    
    # Convert list of dicts to DataFrame for editor
    if st.session_state.portfolio_data:
        df_portfolio = pd.DataFrame(st.session_state.portfolio_data)
        # FORCE DATE TYPE: Convert to datetime to avoid "Float" error if column has NaNs
        df_portfolio['Buy Date'] = pd.to_datetime(df_portfolio['Buy Date'])
    else:
        # Create empty DF with columns
        df_portfolio = pd.DataFrame(columns=["Symbol", "Buy Price", "Qty", "Buy Date"])
        df_portfolio['Buy Date'] = pd.to_datetime(df_portfolio['Buy Date']) # Initialize as datetime

    # DATA EDITOR
    edited_df = st.data_editor(
        df_portfolio,
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor",
        column_config={
            "Buy Date": st.column_config.DateColumn(
                "Buy Date",
                format="YYYY-MM-DD",
                step=1,
            ),
             "Buy Price": st.column_config.NumberColumn(
                "Buy Price",
                min_value=0,
                step=0.05,
                format="₹ %.2f",
            ),
             "Qty": st.column_config.NumberColumn(
                "Quantity",
                min_value=1,
                step=1,
            )
        }
    )
    
    # SAVE BUTTON
    if st.button("💾 Save Changes"):
        # Update session state
        st.session_state.portfolio_data = edited_df.to_dict('records')
        # Save to CSV
        save_portfolio(st.session_state.portfolio_data)
        st.success("Portfolio saved successfully!")

# === TAB 1: PORTFOLIO DASHBOARD ===
with tab1:
    portfolio_data = st.session_state.portfolio_data # Get latest data
    
    if not portfolio_data:
        st.info("👈 Go to 'Manage Holdings' tab to add stocks!")
    else:
        st.subheader("Portfolio Diagnosis")
        
        # Refresh Button
        if st.button("🔄 Refresh Analysis"):
            st.rerun()

        # Process each stock
        results = []
        for stock in portfolio_data:
            df = fetch_weekly_data(stock['Symbol'])
            if df is not None:
                df, sig, color, curr_price, peak_price = analyze_stock(df, stock['Buy Price'], stock.get('Buy Date'))
                
                # Visual Card for each stock
                with st.expander(f"{stock['Symbol']} - {sig}", expanded=False):
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        st.metric("Current Price", f"₹{curr_price:.2f}", f"{(curr_price - stock['Buy Price']):.2f}")
                        st.metric("Buy Price", f"₹{stock['Buy Price']}")
                    
                    with col2:
                        st.markdown(f"**Signal:**")
                        st.markdown(f"<div style='background-color: {color}; color: black; padding: 10px; border-radius: 5px; text-align: center;'><b>{sig}</b></div>", unsafe_allow_html=True)
                        st.write(f"Highest Peak: ₹{peak_price:.2f}")
                        
                    with col3:
                        st.plotly_chart(plot_chart(df, stock['Symbol'], sig), use_container_width=True)





# === TAB 3: MARKET SCREENER ===
with tab3:
    st.header("🎯 High-Probability Breakout & Pullback Radar")
    st.markdown("""
    **A professional swing trading radar featuring two institutional scanning modes:**
    - **🚀 Momentum Breakouts:** Catch explosive momentum runs out of tight weekly squeeze consolidations.
    - **🛡️ Pullbacks to Support:** Buy strong, uptrending stocks as they rest at key moving averages on dry volume (low-risk entries).
    """)
    st.info("⚡ Enhanced with High-Speed Parallel Scanning. 500 stocks scan will complete in 30-45 seconds.")

    # Strategy Mode Selector
    strategy_mode = st.radio("Select Strategy Mode:", ["🚀 Momentum Breakouts", "🛡️ Pullbacks to Support"], horizontal=True, key="screener_strategy_mode")

    # Dynamic controls
    require_trend = True
    min_vol_ratio = 1.5
    require_squeeze = False
    
    with st.expander("🛠️ Scan & Filter Configuration Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        if strategy_mode == "🚀 Momentum Breakouts":
            with c1:
                require_trend = st.checkbox("Strict Weekly Trend Filter (Close > 10w & 40w EMA)", value=True, key="cfg_require_trend")
            with c2:
                min_vol_ratio = st.slider("Minimum Breakout Volume Spike (x of 20w MA)", 1.0, 3.0, 1.5, 0.1, key="cfg_min_vol")
            with c3:
                require_squeeze = st.checkbox("Require Bollinger Band Squeeze in consolidation", value=False, key="cfg_require_squeeze")
        else:
            with c1:
                st.write("**Support Proximity:**")
                st.caption("Price must be within -1.5% to +4.0% of the 40-week EMA or 20-week SMA.")
            with c2:
                st.write("**Dry Volume Rule:**")
                st.caption("Weekly volume must be below its 20-week average volume to confirm no institutional selling.")
            with c3:
                st.write("**Reversal Trigger:**")
                st.caption("Requires a weekly green candle close, hammer pattern, or close above prior week's close.")

    # Controls
    scan_c1, scan_c2 = st.columns([2, 1])
    with scan_c1:
        grade_filter = st.multiselect("Filter by Setup Grade:", ["A+", "A", "B"], default=["A+", "A", "B"], key="screener_grade_filter")
    with scan_c2:
        st.write("")
        st.write("")
        start_scan = st.button("🚀 Start High-Speed Scan (Update)", key="btn_start_scan")

    if start_scan:
        # Fetch Nifty regime
        with st.spinner("Fetching Nifty 50 Trend to determine Market Regime..."):
            nifty_status, nifty_details = ae.get_nifty_market_regime()
            
        st.subheader("🌐 Market Environment Diagnosis")
        if nifty_status == "BULLISH":
            st.success(f"🟢 **Nifty 50 Market Status: BULLISH** | Price: ₹{nifty_details.get('price', 0):,.2f} | 10w EMA: ₹{nifty_details.get('ema10', 0):,.2f} | 40w EMA: ₹{nifty_details.get('ema40', 0):,.2f}\n\n*Excellent environment for trades. Win rate is naturally optimized.*")
        elif nifty_status == "BEARISH":
            st.error(f"🔴 **Nifty 50 Market Status: BEARISH / CORRECTION** | Price: ₹{nifty_details.get('price', 0):,.2f} | 40w EMA: ₹{nifty_details.get('ema40', 0):,.2f}\n\n*High risk of false breakouts. Breakout setups are downgraded; pullbacks are generally safer.*")
        else:
            st.warning(f"🟡 **Nifty 50 Market Status: CAUTION / SIDEWAYS** | Price: ₹{nifty_details.get('price', 0):,.2f}\n\n*Selectively trade only the highest quality (A+) setups.*")

        # Load NIFTY 500
        try:
            nifty_df = pd.read_csv("k:/PYTHON PROJECTS/SWING TRADING/nifty500.csv")
            nifty_500 = nifty_df['Symbol'].tolist()
        except FileNotFoundError:
            st.error("nifty500.csv not found! Using fallback list.")
            nifty_500 = [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", 
                "KOTAKBANK", "LT", "AXISBANK", "HUL", "TATAMOTORS", "MARUTI", "SUNPHARMA"
            ]

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        found_stocks = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def scan_single_ticker(ticker):
            try:
                df = fetch_weekly_data(ticker)
                if df is not None:
                    if strategy_mode == "🚀 Momentum Breakouts":
                        is_match, details, price = ae.analyze_screener_strategy(
                            df, market_regime=nifty_status, 
                            require_trend=require_trend, 
                            min_vol_ratio=min_vol_ratio, 
                            require_squeeze=require_squeeze
                        )
                    else:
                        is_match, details, price = ae.analyze_pullback_strategy(df, market_regime=nifty_status)
                        
                    if is_match:
                        details['Symbol'] = ticker
                        return details
            except Exception as e:
                pass
            return None

        total_stocks = len(nifty_500)
        completed = 0

        with st.spinner(f"Scanning {total_stocks} stocks in parallel for {strategy_mode}..."):
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(scan_single_ticker, ticker): ticker for ticker in nifty_500}
                for future in as_completed(futures):
                    completed += 1
                    ticker = futures[future]
                    status_text.text(f"Scanning {ticker} ({completed}/{total_stocks})...")
                    progress_bar.progress(completed / total_stocks)
                    
                    res = future.result()
                    if res:
                        found_stocks.append(res)

        st.success("Scan Complete!")
        status_text.empty()
        progress_bar.empty()

        # Grade Priority for sorting
        grade_priority = {'A+': 0, 'A': 1, 'B': 2}
        found_stocks.sort(key=lambda x: (grade_priority.get(x['grade'], 9), -x['vol_ratio']))

        # Apply Grade Filter
        filtered_stocks = [s for s in found_stocks if s['grade'] in grade_filter]

        if filtered_stocks:
            st.balloons()
            st.subheader(f"🎯 Stocks matching criteria ({len(filtered_stocks)} setups found):")

            # Create summary dataframe
            summary_data = []
            for r in filtered_stocks:
                is_pb = 'support_line' in r
                summary_data.append({
                    "Symbol": r['Symbol'],
                    "Grade": r['grade'],
                    "Current Price": f"₹{r['current_price']:,.2f}",
                    "Buy Trigger (Confirmation)": f"₹{r['trigger_price']:,.2f}" if not is_pb else "Buy Immediately (Close)",
                    "Stop Loss (SL)": f"₹{r['stop_loss']:,.2f}",
                    "Target 1 (1:2 R:R)": f"₹{r['target_1']:,.2f}",
                    "Target 2 (1:3 R:R)": f"₹{r['target_2']:,.2f}",
                    "Volume Spike": f"{r['vol_ratio']:.2f}x",
                    "Squeeze History": f"{r['squeeze_weeks']} weeks" if not is_pb else "N/A (Pullback)",
                    "Triggered?": "🟢 Yes" if r['triggered'] else "⏳ Pending"
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Save full details to CSV
            results_save_df = pd.DataFrame(filtered_stocks)
            results_save_df.to_csv("screener_results.csv", index=False)
            st.info("Scan results successfully saved to `screener_results.csv`.")

            # Show expanders with charts
            for rec in filtered_stocks:
                ticker = rec['Symbol']
                is_pb = 'support_line' in rec
                trig_icon = "🟢" if rec['triggered'] else "⏳"
                title_trigger = f"₹{rec['trigger_price']:,.2f}" if not is_pb else "Buy at Close"
                with st.expander(f"**{ticker}** — Grade: {rec['grade']} | Price: ₹{rec['current_price']:,.2f} | Trigger: {title_trigger} {trig_icon}"):
                    st.markdown(f"#### Setup Analysis: {rec['grade_desc']}")
                    col_info, col_chart = st.columns([1, 2])
                    with col_info:
                        st.metric("Setup Grade", rec['grade'])
                        st.metric("Current Price", f"₹{rec['current_price']:,.2f}")
                        st.metric("Buy Trigger Price", f"₹{rec['trigger_price']:,.2f}", 
                                  help="For Breakouts: Buy high breakout high. For Pullbacks: Buy current price.")
                        st.metric("Stop Loss (SL)", f"₹{rec['stop_loss']:,.2f}",
                                  help="Place stop loss below breakout low or support line")
                        st.metric("Target 1 (1:2 R:R)", f"₹{rec['target_1']:,.2f}")
                        st.metric("Target 2 (1:3 R:R)", f"₹{rec['target_2']:,.2f}")
                        st.metric("Volume Spike/Ratio", f"{rec['vol_ratio']:.2f}x avg")
                        st.metric("Squeeze Weeks", f"{rec['squeeze_weeks']} weeks" if not is_pb else "N/A (Pullback)")
                        st.metric("Status", "Triggered 🟢" if rec['triggered'] else "Pending Trigger ⏳")
                    with col_chart:
                        df_chart = fetch_weekly_data(ticker)
                        if df_chart is not None:
                            df_chart_slice = df_chart.tail(52).copy()
                            import pandas_ta as ta
                            df_chart_slice['EMA_10'] = ta.ema(df_chart_slice['Close'], length=10)
                            df_chart_slice['EMA_40'] = ta.ema(df_chart_slice['Close'], length=40)
                            
                            bb = ta.bbands(df_chart_slice['Close'], length=20, std=2)
                            bbu_col = next((c for c in bb.columns if c.startswith("BBU")), None)
                            bbm_col = next((c for c in bb.columns if c.startswith("BBM")), None)
                            df_chart_slice['BB_Upper'] = bb[bbu_col]
                            df_chart_slice['BB_Mid'] = bb[bbm_col] if bbm_col else df_chart_slice['Close'].rolling(20).mean()

                            fig = go.Figure()
                            fig.add_trace(go.Candlestick(
                                x=df_chart_slice.index, open=df_chart_slice['Open'], high=df_chart_slice['High'],
                                low=df_chart_slice['Low'], close=df_chart_slice['Close'], name='Price'))
                            fig.add_trace(go.Scatter(x=df_chart_slice.index, y=df_chart_slice['EMA_10'], name='10w EMA',
                                                     line=dict(color='orange', width=1)))
                            fig.add_trace(go.Scatter(x=df_chart_slice.index, y=df_chart_slice['EMA_40'], name='40w EMA',
                                                     line=dict(color='cyan', width=1.5)))
                            
                            if is_pb:
                                fig.add_trace(go.Scatter(x=df_chart_slice.index, y=df_chart_slice['BB_Mid'], name='20w SMA (BB Mid)',
                                                         line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot')))
                                fig.add_hline(y=rec['support_line'], line_dash="dash", line_color="magenta",
                                              annotation_text="Support Line")
                            else:
                                fig.add_trace(go.Scatter(x=df_chart_slice.index, y=df_chart_slice['BB_Upper'], name='BB Upper',
                                                         line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot')))
                                fig.add_hline(y=rec['trigger_price'], line_dash="dash", line_color="green",
                                              annotation_text="Buy Trigger")
                            
                            fig.add_hline(y=rec['stop_loss'], line_dash="dash", line_color="red", annotation_text="Stop Loss")
                            fig.add_hline(y=rec['target_1'], line_dash="dash", line_color="lime", annotation_text="Target 1")
                            fig.add_hline(y=rec['target_2'], line_dash="dash", line_color="gold", annotation_text="Target 2")
                            
                            fig.update_layout(
                                title=f"{ticker} — Weekly Setup Chart",
                                template="plotly_dark", height=450,
                                xaxis_rangeslider_visible=False,
                                yaxis=dict(title="Price", side="right"),
                                showlegend=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stocks found matching the criteria and grade filters this week.")

    # --- Load Previous Results if available ---
    elif os.path.exists("screener_results.csv"):
        st.subheader("📜 Last Scan Results")
        try:
            old_results = pd.read_csv("screener_results.csv")
            if 'Grade' in old_results.columns:
                # Apply grade filter to old results
                old_results = old_results[old_results['Grade'].isin(grade_filter)]
                
                # Re-display summary dataframe
                st.dataframe(old_results[[
                    "Symbol", "Grade", "Current Price", "Buy Trigger (Confirmation)",
                    "Stop Loss (SL)", "Target 1 (1:2 R:R)", "Target 2 (1:3 R:R)",
                    "Volume Spike", "Squeeze History", "Triggered?"
                ]], use_container_width=True)
                
                # Show charts for last results
                for index, row in old_results.iterrows():
                    ticker = row['Symbol']
                    grade = row['Grade']
                    price = row['Current Price']
                    trigger = row['Buy Trigger (Confirmation)']
                    stop_loss = row['Stop Loss (SL)']
                    t1 = row['Target 1 (1:2 R:R)']
                    t2 = row['Target 2 (1:3 R:R)']
                    vol_ratio = row['Volume Spike']
                    sqz_weeks = row['Squeeze History']
                    triggered_val = row['Triggered?']
                    
                    with st.expander(f"**{ticker}** — Grade: {grade} | Price: {price} | Trigger: {trigger} | {triggered_val}"):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Setup Grade", grade)
                            st.metric("Current Price", price)
                            st.metric("Buy Trigger Price", trigger)
                            st.metric("Stop Loss (SL)", stop_loss)
                            st.metric("Target 1 (1:2 R:R)", t1)
                            st.metric("Target 2 (1:3 R:R)", t2)
                            st.metric("Volume Spike", vol_ratio)
                            st.metric("Squeeze History", sqz_weeks)
                            st.metric("Status", triggered_val)
                        with col2:
                            df_chart = fetch_weekly_data(ticker)
                            if df_chart is not None:
                                df_chart_slice = df_chart.tail(52).copy()
                                import pandas_ta as ta
                                df_chart_slice['EMA_10'] = ta.ema(df_chart_slice['Close'], length=10)
                                df_chart_slice['EMA_40'] = ta.ema(df_chart_slice['Close'], length=40)
                                bb = ta.bbands(df_chart_slice['Close'], length=20, std=2)
                                bbu_col = next((c for c in bb.columns if c.startswith("BBU")), None)
                                df_chart_slice['BB_Upper'] = bb[bbu_col]
                                
                                try:
                                    trig_f = float(str(trigger).replace('₹', '').replace(',', ''))
                                    sl_f = float(str(stop_loss).replace('₹', '').replace(',', ''))
                                    t1_f = float(str(t1).replace('₹', '').replace(',', ''))
                                    t2_f = float(str(t2).replace('₹', '').replace(',', ''))
                                
                                except:
                                    trig_f = df_chart_slice['Close'].iloc[-1]
                                    sl_f = trig_f * 0.95
                                    t1_f = trig_f * 1.10
                                    t2_f = trig_f * 1.15

                                fig = go.Figure()
                                fig.add_trace(go.Candlestick(
                                    x=df_chart_slice.index, open=df_chart_slice['Open'], high=df_chart_slice['High'],
                                    low=df_chart_slice['Low'], close=df_chart_slice['Close'], name='Price'))
                                fig.add_trace(go.Scatter(x=df_chart_slice.index, y=df_chart_slice['EMA_10'], name='10w EMA',
                                                         line=dict(color='orange', width=1)))
                                fig.add_trace(go.Scatter(x=df_chart_slice.index, y=df_chart_slice['EMA_40'], name='40w EMA',
                                                         line=dict(color='cyan', width=1.5)))
                                fig.add_trace(go.Scatter(x=df_chart_slice.index, y=df_chart_slice['BB_Upper'], name='BB Upper',
                                                         line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot')))
                                
                                fig.add_hline(y=trig_f, line_dash="dash", line_color="green", annotation_text="Buy Trigger")
                                fig.add_hline(y=sl_f, line_dash="dash", line_color="red", annotation_text="Stop Loss")
                                fig.add_hline(y=t1_f, line_dash="dash", line_color="lime", annotation_text="Target 1")
                                fig.add_hline(y=t2_f, line_dash="dash", line_color="gold", annotation_text="Target 2")
                                
                                fig.update_layout(
                                    title=f"{ticker} — Weekly Setup Chart",
                                    template="plotly_dark", height=450,
                                    xaxis_rangeslider_visible=False,
                                    yaxis=dict(title="Price", side="right"),
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
            else:
                # Old CSV format fallback
                st.dataframe(old_results, use_container_width=True)
                st.info("👆 These are results from your last scan. Click 'Start Scan (Update)' to refresh.")
        except Exception as e:
            st.error(f"Could not load previous results: {e}")
    else:
        st.info("No previous scan results found. Click 'Start Scan (Update)' to begin.")


# === TAB 6: ACCUMULATION SCANNER ===
with tab6:
    st.header("🔎 Accumulation Scanner — Smart Money Detector")
    st.markdown("""
    Detects **institutional accumulation** (volume contracting + price flat in a squeeze) and **early breakouts** on weekly charts.
    
    | Filter | Purpose |
    |--------|---------|
    | **40-week EMA Trend** | Prevents buying consolidations in down-trending stocks (redistribution). |
    | **Up/Down Vol Ratio** | Verifies that volume is higher on green weeks than red weeks (buying pressure). |
    | **Volume Dry-up** | Confirms that average volume inside the squeeze is low, indicating supply is locked. |
    """)

    # Controls
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        min_squeeze = st.slider("Min Accumulation Duration (weeks)", 3, 12, 4, key="acc_squeeze")
    with ctrl2:
        st.write("")
        st.write("")
        scan_btn = st.button("🚀 Start Full Scan (Nifty 500)", key="acc_scan")
    with ctrl3:
        st.write("")
        st.write("")
        quick_btn = st.button("⚡ Quick Scan (Top 50)", key="acc_quick")
        
    with st.expander("🛠️ Advanced Accumulation Filters (Reduce False Signals)", expanded=True):
        ac1, ac2 = st.columns(2)
        with ac1:
            require_trend_acc = st.checkbox("Uptrend Filter (Close > 40w EMA)", value=True, key="acc_require_trend")
        with ac2:
            min_up_down_ratio = st.slider("Min Up/Down Volume Pressure Ratio", 1.0, 2.0, 1.2, 0.1, key="acc_min_vol_ratio")

    if scan_btn or quick_btn:
        # Load stock list
        try:
            nifty_df = pd.read_csv("k:/PYTHON PROJECTS/SWING TRADING/nifty500.csv")
            stock_list = nifty_df['Symbol'].tolist()
        except FileNotFoundError:
            st.error("nifty500.csv not found!")
            stock_list = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN",
                          "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "HUL",
                          "TATAMOTORS", "MARUTI", "SUNPHARMA"]

        if quick_btn:
            stock_list = stock_list[:50]

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_cb(ticker, idx, total):
            pct = (idx + 1) / total
            progress_bar.progress(pct)
            status_text.text(f"Scanning {idx+1}/{total}: {ticker}")

        with st.spinner(f"Scanning {len(stock_list)} stocks on weekly charts..."):
            accumulating, breakouts = se.run_full_scan(
                stock_list, min_squeeze_weeks=min_squeeze, 
                require_trend=require_trend_acc, 
                min_up_down_ratio=min_up_down_ratio, 
                progress_callback=progress_cb
            )

        progress_bar.empty()
        status_text.empty()

        st.success(f"Scan complete! Found **{len(accumulating)}** accumulating + **{len(breakouts)}** breakout stocks.")

        # === ACCUMULATING NOW ===
        st.divider()
        st.subheader(f"🟡 Accumulating Now ({len(accumulating)} stocks)")
        st.caption("Volume is contracting but price is tight — smart money loading positions quietly near support.")

        if not accumulating:
            st.info("No stocks found in accumulation phase with current filters. Try relaxing the filters in the configuration panel.")
        else:
            for rec in accumulating:
                ticker = rec['ticker'].replace('.NS', '')
                with st.expander(f"**{ticker}** — Score: {rec['score']}/10 | Price: ₹{rec['price']:,.2f} | Squeeze: {rec['squeeze_weeks']} weeks | Up/Down Vol: {rec['up_down_ratio']:.2f}x", expanded=False):
                    mc1, mc2 = st.columns([2, 1])
                    with mc2:
                        st.metric("Accumulation Score", f"{rec['score']}/10")
                        st.metric("Current Price", f"₹{rec['price']:,.2f}")
                        st.metric("Trend Status", rec['trend_status'])
                        st.metric("Squeeze Duration", f"{rec['squeeze_weeks']} weeks")
                        st.metric("RSI", f"{rec['rsi']:.1f}")
                        st.metric("Up/Down Vol Ratio", f"{rec['up_down_ratio']:.2f}x")
                        st.metric("Buy Zone (Mid to Lower BB)", f"₹{rec['buy_zone_low']:,.2f} – ₹{rec['buy_zone_high']:,.2f}")
                        st.metric("Stop Loss (Exit below)", f"₹{rec['stop_loss']:,.2f}")
                        st.metric("Target Price", f"₹{rec['target_price']:,.2f}")
                        rr = rec['risk_reward']
                        st.metric("Risk:Reward", f"1:{rr:.1f}" if rr > 0 else "N/A")
                    with mc1:
                        df_chart = rec['df'].tail(52)  # Last 1 year
                        fig = go.Figure()
                        # Candlestick
                        fig.add_trace(go.Candlestick(
                            x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                            low=df_chart['Low'], close=df_chart['Close'], name='Price'))
                        # Volume bars
                        colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_chart['Close'], df_chart['Open'])]
                        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name='Volume',
                                             marker_color=colors, opacity=0.4, yaxis='y2'))
                        # BB
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Upper'], name='BB Upper',
                                                 line=dict(color='rgba(100,200,100,0.5)', width=1)))
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Lower'], name='BB Lower',
                                                 line=dict(color='rgba(100,200,100,0.5)', width=1), fill='tonexty',
                                                 fillcolor='rgba(100,200,100,0.05)'))
                        # KC
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['KC_Upper'], name='KC Upper',
                                                 line=dict(color='rgba(255,165,0,0.5)', width=1, dash='dot')))
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['KC_Lower'], name='KC Lower',
                                                 line=dict(color='rgba(255,165,0,0.5)', width=1, dash='dot')))
                        
                        # Support / Target lines
                        fig.add_hline(y=rec['buy_zone_high'], line_dash="dot", line_color="green",
                                      annotation_text="Buy Zone High (Mid BB)")
                        fig.add_hline(y=rec['buy_zone_low'], line_dash="dot", line_color="green",
                                      annotation_text="Buy Zone Low (Lower BB)")
                        fig.add_hline(y=rec['stop_loss'], line_dash="dash", line_color="red",
                                      annotation_text="Stop Loss")
                        fig.add_hline(y=rec['target_price'], line_dash="dash", line_color="lime",
                                      annotation_text="Target")
                        fig.update_layout(
                            title=f"{ticker} — Weekly (Squeeze Active {rec['squeeze_weeks']}w)",
                            template="plotly_dark", height=450,
                            xaxis_rangeslider_visible=False,
                            yaxis=dict(title="Price", side="right"),
                            yaxis2=dict(title="Volume", overlaying='y', side='left', showgrid=False,
                                        range=[0, df_chart['Volume'].max() * 4]),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # === BREAKOUT INITIATED ===
        st.divider()
        st.subheader(f"🟢 Breakout Initiated ({len(breakouts)} stocks)")
        st.caption("Squeeze just released with volume spike — the move has begun.")

        if not breakouts:
            st.info("No breakout signals found in the current scan.")
        else:
            for rec in breakouts:
                ticker = rec['ticker'].replace('.NS', '')
                with st.expander(f"**{ticker}** — Score: {rec['score']}/10 | Price: ₹{rec['price']:,.2f} | Vol Spike: {rec['vol_ratio']:.1f}x", expanded=False):
                    mc1, mc2 = st.columns([2, 1])
                    with mc2:
                        st.metric("Breakout Score", f"{rec['score']}/10")
                        st.metric("Current Price", f"₹{rec['price']:,.2f}")
                        st.metric("Trend Status", rec['trend_status'])
                        st.metric("Prior Squeeze", f"{rec['squeeze_weeks']} weeks")
                        st.metric("RSI", f"{rec['rsi']:.1f}")
                        st.metric("Volume Spike", f"{rec['vol_ratio']:.1f}x avg")
                        st.metric("Breakout Level", f"₹{rec['breakout_level']:,.2f}")
                        st.metric("Target Price", f"₹{rec['target_price']:,.2f}")
                        st.metric("Stop Loss (Mid BB)", f"₹{rec['stop_loss']:,.2f}")
                        rr = rec['risk_reward']
                        st.metric("Risk:Reward", f"1:{rr:.1f}" if rr > 0 else "N/A")
                    with mc1:
                        df_chart = rec['df'].tail(52)
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                            low=df_chart['Low'], close=df_chart['Close'], name='Price'))
                        colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_chart['Close'], df_chart['Open'])]
                        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name='Volume',
                                             marker_color=colors, opacity=0.4, yaxis='y2'))
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Upper'], name='BB Upper',
                                                 line=dict(color='rgba(100,200,100,0.5)', width=1)))
                        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Lower'], name='BB Lower',
                                                 line=dict(color='rgba(100,200,100,0.5)', width=1), fill='tonexty',
                                                 fillcolor='rgba(100,200,100,0.05)'))
                        fig.add_hline(y=rec['breakout_level'], line_dash="dash", line_color="yellow",
                                      annotation_text="Breakout")
                        fig.add_hline(y=rec['target_price'], line_dash="dash", line_color="lime",
                                      annotation_text="Target")
                        fig.add_hline(y=rec['stop_loss'], line_dash="dash", line_color="red",
                                      annotation_text="Stop Loss")
                        fig.update_layout(
                            title=f"{ticker} — Weekly (Breakout! Vol {rec['vol_ratio']:.1f}x)",
                            template="plotly_dark", height=450,
                            xaxis_rangeslider_visible=False,
                            yaxis=dict(title="Price", side="right"),
                            yaxis2=dict(title="Volume", overlaying='y', side='left', showgrid=False,
                                        range=[0, df_chart['Volume'].max() * 4]),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)


# === TAB 7: PLAYBOOK & RISK RULES ===
with tab7:
    st.header("📖 Swing Trader's Playbook & Capital Rules")
    st.write("A professional risk management blueprint to maximize win rates, handle empty scanners, and neutralize false breakouts.")
    
    tab_p1, tab_p2 = st.tabs(["🛡️ Handling Empty Radars", "🛑 Managing False Breakouts"])
    
    with tab_p1:
        st.subheader("What to do if the Radar is Empty?")
        st.info("💡 **Key Rule:** An empty scanner is a **bullish signal for your capital**. It means the broad market is correcting or distributing, and smart money is sitting in cash. In trading, **cash is a position**.")
        
        st.markdown("""
        ### Action Plan A: Switch to Pullbacks to Support
        - **Market Squeezes** (accumulation) and **Breakouts** work best in an active bull market.
        - If Tab 6 (Accumulation Scanner) or Tab 3 (Breakouts) are empty, switch Tab 3's Strategy Mode to **🛡️ Pullbacks to Support**.
        - This scans for strong stocks resting at major moving averages (40-week EMA) on dry volume. Pullbacks have a higher success rate when the overall index (Nifty 50) is consolidating.
        
        ### Action Plan B: Relax the Scanner Filters
        If you still want to search for setups, open the collapsible config parameters inside the scanner tab and:
        1. **Lower Volume Spike:** Slide from 1.5x down to **1.0x or 1.2x** (requires less heavy institutional volume).
        2. **Disable EMA Trend Filters:** Uncheck the strict EMA trend filters to allow stocks consolidating at multi-year lows (bottom-fishing setups).
        3. **Reduce Squeeze Weeks:** Set minimum accumulation squeeze weeks to **3 weeks** instead of 4 or 5.
        """)
        
    with tab_p2:
        st.subheader("How to Neutralize False Breakouts?")
        st.warning("⚠️ **Reality Check:** No scanner has a 100% win rate. False breakouts (where a stock looks like it is accumulating but then breaks down) **will** happen. Capital preservation is the only way to survive.")
        
        st.markdown("""
        ### Rule 1: The 50/50 Scale-In Entry (Reduces Risk by 50%)
        - **50% Position:** Buy half of your planned position inside the accumulation buy zone (near the BB Lower or BB Mid support).
        - **50% Position:** Buy the remaining half **ONLY** when the stock breaks out of the squeeze (Weekly close above the Upper Bollinger Band with a clear volume spike).
        - This prevents you from buying a consolidation that ultimately breaks down to the downside.
        
        ### Rule 2: Enforce the Hard Stop Loss
        - Inside Tab 6, the scanner calculates a stop loss at **1.5% below the BB Lower boundary**. 
        - If a weekly candle closes below this level, the consolidation has failed (it was distribution, not accumulation). **Exit immediately.** Do not hope or average down.
        
        ### Rule 3: Enforce a Time Stop (Avoid Dead Money)
        - If you enter an accumulation position and the stock does not break out within **6 weeks**, exit. 
        - Smart money accumulation should lead to markup. If it remains flat, your capital is locked in 'dead money'. Reallocate to active setups.
        """)