from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


UNIVERSE = {
    "Technology": (
        "AAPL:Apple,MSFT:Microsoft,NVDA:Nvidia,AMD:AMD,AVGO:Broadcom,"
        "ORCL:Oracle,CRM:Salesforce,ADBE:Adobe,INTC:Intel,QCOM:Qualcomm,"
        "MU:Micron,AMAT:Applied Materials,LRCX:Lam Research,KLAC:KLA,"
        "NOW:ServiceNow,PLTR:Palantir,CSCO:Cisco,DELL:Dell,IBM:IBM,"
        "PANW:Palo Alto Networks,CRWD:CrowdStrike,TSM:TSMC ADR,ASML:ASML ADR"
    ),
    "Internet & Consumer": (
        "GOOGL:Alphabet,META:Meta,AMZN:Amazon,NFLX:Netflix,TSLA:Tesla,"
        "UBER:Uber,DASH:DoorDash,ABNB:Airbnb,BKNG:Booking Holdings,"
        "WMT:Walmart,COST:Costco,HD:Home Depot,LOW:Lowe’s,NKE:Nike,"
        "MCD:McDonald’s,SBUX:Starbucks,KO:Coca-Cola,PEP:PepsiCo,PG:Procter & Gamble,"
        "DIS:Disney,SPOT:Spotify,SHOP:Shopify"
    ),
    "Financials": (
        "JPM:JPMorgan,BAC:Bank of America,GS:Goldman Sachs,MS:Morgan Stanley,"
        "C:Citigroup,WFC:Wells Fargo,BLK:BlackRock,BRK-B:Berkshire Hathaway,"
        "V:Visa,MA:Mastercard,AXP:American Express,SCHW:Charles Schwab,"
        "KKR:KKR,BX:Blackstone,COIN:Coinbase,HOOD:Robinhood"
    ),
    "Health Care": (
        "LLY:Eli Lilly,UNH:UnitedHealth,JNJ:Johnson & Johnson,ABBV:AbbVie,"
        "MRK:Merck,PFE:Pfizer,TMO:Thermo Fisher,ISRG:Intuitive Surgical,"
        "AMGN:Amgen,VRTX:Vertex,GILD:Gilead,DHR:Danaher"
    ),
    "Industrials & Energy": (
        "GE:GE Aerospace,BA:Boeing,CAT:Caterpillar,DE:Deere,HON:Honeywell,"
        "RTX:RTX,LMT:Lockheed Martin,UNP:Union Pacific,UPS:UPS,FDX:FedEx,"
        "ETN:Eaton,XOM:ExxonMobil,CVX:Chevron,COP:ConocoPhillips,"
        "SLB:SLB,NEE:NextEra Energy,CEG:Constellation Energy,LIN:Linde,FCX:Freeport-McMoRan"
    ),
    "Broad market & Style ETFs": (
        "SPY:S&P 500,VOO:Vanguard S&P 500,IVV:iShares S&P 500,QQQ:Nasdaq-100,"
        "QQQM:Invesco Nasdaq-100,IWM:Russell 2000,IWV:Russell 3000,"
        "VTI:Total US Market,DIA:Dow Jones,VT:Total World,VXUS:International Equity,"
        "EFA:Developed Markets,EEM:Emerging Markets,VEA:Developed ex-US,"
        "VWO:Emerging Markets,RSP:Equal-Weight S&P 500,SCHD:Dividend Equity,"
        "VUG:US Growth,VTV:US Value,MTUM:US Momentum,QUAL:US Quality,USMV:Min Vol USA"
    ),
    "Sector & Thematic ETFs": (
        "XLK:Technology,XLF:Financials,XLE:Energy,XLV:Health Care,"
        "XLY:Consumer Discretionary,XLP:Consumer Staples,XLI:Industrials,"
        "XLB:Materials,XLU:Utilities,XLRE:Real Estate,XLC:Communication Services,"
        "SMH:Semiconductors,SOXX:Semiconductors,IGV:Software,XBI:Biotech,"
        "KRE:Regional Banks,ITA:Aerospace & Defense,JETS:Airlines,ARKK:Innovation,"
        "TAN:Solar,ICLN:Clean Energy,URA:Uranium"
    ),
    "Bonds, Commodities & Digital Asset ETFs": (
        "TLT:Long Treasuries,IEF:Intermediate Treasuries,SHY:Short Treasuries,"
        "SGOV:Treasury Bills,BND:Total US Bonds,AGG:US Aggregate Bonds,"
        "LQD:Investment Grade Credit,HYG:High Yield Credit,TIP:Inflation Protected,"
        "GLD:Gold,IAU:Gold,SLV:Silver,USO:Oil,DBC:Commodities,"
        "IBIT:Spot Bitcoin,FBTC:Spot Bitcoin,ETHA:Spot Ether"
    ),
}
MODEL_NAMES = ["Ridge", "Random Forest", "Gradient Boosting"]


def forecast_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_leaf=12,
            random_state=42, n_jobs=1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=90, max_leaf_nodes=12, learning_rate=0.05,
            min_samples_leaf=20, early_stopping=False, random_state=42,
        ),
    }


def completed_bars(frame):
    now = pd.Timestamp.now(tz="UTC")
    index = frame.index
    utc_index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return frame.loc[utc_index + pd.Timedelta(minutes=5) <= now]


def features_from_prices(asset, spy, qqq, intraday):
    panel = pd.concat(
        [asset.rename("Asset"), spy.rename("SPY"), qqq.rename("QQQ")],
        axis=1, join="inner",
    ).dropna().sort_index()
    panel = panel.where(panel > 0).dropna()
    log_price = np.log(panel)
    moves = log_price.diff()
    features = moves.rename(columns={"Asset": "Asset_1", "SPY": "SPY_1", "QQQ": "QQQ_1"})
    for lookback in [3, 5, 10, 20]:
        features[f"Asset_{lookback}"] = log_price["Asset"].diff(lookback)
        features[f"SPY_{lookback}"] = log_price["SPY"].diff(lookback)
        features[f"QQQ_{lookback}"] = log_price["QQQ"].diff(lookback)
    features["Volatility_10"] = moves["Asset"].rolling(10).std()
    features["Volatility_20"] = moves["Asset"].rolling(20).std()
    features["Relative_to_SPY_10"] = features["Asset_10"] - features["SPY_10"]
    features["Relative_to_QQQ_10"] = features["Asset_10"] - features["QQQ_10"]
    if intraday:
        timestamps = panel.index.to_series()
        continuous = timestamps.diff(20).eq(pd.Timedelta(minutes=100))
        features = features.where(continuous, axis=0)
    return panel["Asset"], features


def direction_score(actual, prediction):
    up = actual > 0
    down = actual < 0
    if up.sum() < 5 or down.sum() < 5:
        return np.nan
    return float(((prediction[up] > 0).mean() + (prediction[down] < 0).mean()) / 2)


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def estimate_forecast(asset, spy, qqq, horizon, intraday):
    close, features = features_from_prices(asset, spy, qqq, intraday)
    if features.empty or features.iloc[-1].isna().any():
        return None

    target = np.log(close.shift(-horizon) / close)
    bar_position = pd.Series(np.arange(len(close)), index=close.index)
    if intraday:
        timestamps = close.index.to_series()
        continuous = (timestamps.shift(-horizon) - timestamps).eq(
            pd.Timedelta(minutes=5 * horizon)
        )
        target = target.where(continuous)

    dataset = features.join(target.rename("Target")).join(
        bar_position.rename("Position")
    ).dropna()
    # Sample the holdout at the forecast horizon so its returns do not overlap.
    if len(dataset) < max(180, 30 * horizon):
        return None
    split = int(len(dataset) * 0.75)
    test = dataset.iloc[split::horizon]
    train = dataset.loc[dataset["Position"] + horizon < test["Position"].iloc[0]]
    if len(train) < 120 or len(test) < 12:
        return None

    columns = features.columns
    actual = test["Target"].to_numpy()
    validation_predictions, current_predictions = {}, {}
    for name, model in forecast_models().items():
        model.fit(train[columns], train["Target"])
        validation_predictions[name] = model.predict(test[columns])
        model.fit(dataset[columns], dataset["Target"])
        current_predictions[name] = float(model.predict(features.iloc[[-1]])[0])

    ensemble_test = np.mean(list(validation_predictions.values()), axis=0)
    prediction = float(np.mean(list(current_predictions.values())))
    errors = actual - ensemble_test
    lower_error, upper_error = np.quantile(errors, [0.10, 0.90])
    baseline_mae = float(np.abs(np.expm1(actual)).mean())
    ensemble_mae = float(np.abs(np.expm1(actual) - np.expm1(ensemble_test)).mean())
    rows = []
    validation_predictions["Ensemble"] = ensemble_test
    current_predictions["Ensemble"] = prediction
    for name, values in validation_predictions.items():
        rows.append({
            "Model": name,
            "Projected return %": 100 * np.expm1(current_predictions[name]),
            "Holdout MAE (pp)": 100 * np.abs(np.expm1(actual) - np.expm1(values)).mean(),
            "Balanced directional accuracy %": 100 * direction_score(actual, values),
        })

    votes = [current_predictions[name] for name in MODEL_NAMES]
    agreement = sum(np.sign(vote) == np.sign(prediction) for vote in votes)
    accuracy = direction_score(actual, ensemble_test)
    anchor = float(close.iloc[-1])
    return {
        "rows": rows, "anchor": anchor, "as_of": close.index[-1],
        "forecast": anchor * np.exp(prediction),
        "return": float(np.expm1(prediction)),
        "range": (anchor * np.exp(prediction + lower_error),
                  anchor * np.exp(prediction + upper_error)),
        "agreement": agreement, "accuracy": accuracy,
        "mae": ensemble_mae, "baseline": baseline_mae, "n": len(test),
        "test_start": test.index[0], "test_end": test.index[-1],
    }


def draw_forecast(result, intraday, min_move):
    if result is None:
        st.info("A forecast is unavailable for this horizon and data history.")
        return

    stamp = result["as_of"]
    fresh = True
    if intraday:
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        bar_end = stamp + pd.Timedelta(minutes=5)
        now = pd.Timestamp.now(tz="UTC")
        target_et = (bar_end + pd.Timedelta(minutes=30)).tz_convert(NY)
        fresh = (
            now - bar_end <= pd.Timedelta(minutes=15)
            and (16, 0) >= (target_et.hour, target_et.minute)
            and target_et.date() == now.tz_convert(NY).date()
        )

    estimated = result["return"]
    qualified = (
        fresh and result["n"] >= 30
        and result["mae"] < result["baseline"]
        and result["accuracy"] >= 0.55
        and result["agreement"] == len(MODEL_NAMES)
        and abs(estimated) >= min_move
    )
    label = ("BUY BIAS" if estimated > 0 else "SELL BIAS") if qualified else "WAIT"
    cards = st.columns(4)
    cards[0].metric("Research signal", label)
    cards[1].metric("Projected price", f"${result['forecast']:,.2f}", f"{estimated:+.2%}")
    lo, hi = result["range"]
    cards[2].metric("Historical error range", f"${lo:,.2f} – ${hi:,.2f}")
    cards[3].metric("Model agreement", f"{result['agreement']}/{len(MODEL_NAMES)}")
    if intraday:
        st.caption(f"Anchor ${result['anchor']:,.2f} · bar ending {bar_end.tz_convert(NY):%b %d, %H:%M} ET")
        if not fresh:
            st.warning("Historical intraday forecast. A fresh 30-minute regular-session signal is unavailable.")
    else:
        st.caption(f"Anchor ${result['anchor']:,.2f} · adjusted daily close on {stamp:%Y-%m-%d}")

    st.dataframe(pd.DataFrame(result["rows"]).round(3), hide_index=True)
    st.caption(
        f"Holdout {result['test_start']:%Y-%m-%d} to {result['test_end']:%Y-%m-%d} · "
        f"{result['n']} non-overlapping forecasts · "
        f"flat-price baseline MAE {100 * result['baseline']:.3f} percentage points."
    )
    with st.expander("What do these forecast numbers mean?"):
        st.markdown(
            """
- **Projected return** is the average predicted log return from the three models,
  converted to a percentage. Projected price applies that return to the displayed anchor.
- **Historical error range** uses the 10th and 90th percentiles of the holdout log-return
  errors. Future outcomes can fall outside it, especially around earnings or economic releases.
- **MAE** measures the typical absolute prediction error in percentage points. An MAE of
  2.0 means the return forecast missed by two percentage points on average.
- **Balanced directional accuracy** gives rising and falling observations equal weight.
  A model that predicts up every time scores 50% when both directions are present.
- **Buy / Sell bias** requires all three models to agree, at least 30 holdout forecasts,
  at least 55% balanced directional accuracy, lower MAE than a flat-price forecast,
  and a move above your threshold. These are screening thresholds. The holdout measures
  forecast errors and does not estimate strategy profitability after costs.
            """
        )


st.markdown(
    """
<style>
.stApp {background: #0b1422; color: #e4edf7;}
[data-testid="stSidebar"] {background: #111e30;}
[data-testid="stHeader"] {background: transparent;}
h1, h2, h3, label, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
    color: #e4edf7 !important;
}
[data-testid="stMetric"] {
    background: #142337;
    border: 1px solid #293d57;
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {color: #e4edf7;}
.block-container {max-width: 1480px; padding-top: 2.5rem;}
</style>
    """,
    unsafe_allow_html=True,
)

st.title("US Equity Research Terminal")
st.caption("Nasdaq · S&P 500 · US-listed stocks and ETFs")

with st.sidebar:
    group = st.selectbox("Asset group", ["All stocks & ETFs", *UNIVERSE])
    groups = UNIVERSE.values() if group == "All stocks & ETFs" else [UNIVERSE[group]]
    options = dict(
        entry.split(":", 1) for listing in groups for entry in listing.split(",")
    )
    default_index = list(options).index("NVDA") if "NVDA" in options else 0
    selected = st.selectbox(
        "Search company or ticker", list(options), index=default_index,
        format_func=lambda symbol: f"{symbol} · {options[symbol]}",
    )
    custom = st.text_input("Custom US ticker", max_chars=15).strip().upper()
    ticker = custom or selected
    if not re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)?", ticker):
        st.warning("Enter a ticker such as AAPL, BRK-B, or QQQ.")
        st.stop()

    page = st.radio("View", ["Overview", "Factor models", "Forecasts", "News"])
    auto_refresh = st.toggle("Refresh every five minutes", value=True)
    if page == "Overview":
        fair = st.number_input("Your fair value estimate ($)", min_value=0.0, step=1.0)
    elif page == "Factor models":
        years = st.selectbox("Regression history (years)", [1, 2, 3, 5], index=2)
        window = st.selectbox("Relative pricing window (sessions)", [5, 10, 20], index=1)
        chosen_model = st.radio("Active model", list(MODELS), index=2, horizontal=True)
    elif page == "Forecasts":
        segment = st.radio("Forecast horizon", ["30 minutes", "1 session", "5 sessions", "20 sessions"])
        intraday_forecast = segment == "30 minutes"
        threshold = st.number_input(
            "Minimum absolute predicted move (%)", min_value=0.0, max_value=20.0,
            value=0.15 if intraday_forecast else 0.5, step=0.05,
            key=f"threshold_{segment}",
        ) / 100


def draw_market():
    symbols = tuple(dict.fromkeys([ticker, *MARKETS, *SECTORS]))
    bars, checked = prices(symbols, "5d", "5m", False)
    daily, _ = prices(symbols, "1mo", "1d", False)
    market = quote_table(MARKETS, bars, daily)
    for card, (_, row) in zip(st.columns(len(MARKETS)), market.iterrows()):
        quote, change = row["Price"], row["Session move"]
        card.metric(
            row["Instrument"], "—" if pd.isna(quote) else f"{quote:,.2f}",
            None if pd.isna(change) else f"{change:+.2f}%",
        )
    st.caption(f"Yahoo Finance request · {checked.tz_convert(ZURICH):%b %d, %H:%M:%S} Zurich")
    st.subheader(f"{ticker} · Price and volume")
    if ticker in bars:
        session = session_bars(bars[ticker])
        quote = float(session["Close"].iloc[-1])
        quote_card, bar_card, value_card = st.columns(3)
        quote_card.metric("Latest bar close", f"${quote:,.2f}")
        bar_card.metric("Latest bar start (ET)", session.index[-1].strftime("%b %d · %H:%M"))
        if fair > 0:
            value_card.metric("Gap to your fair value", f"{quote / fair - 1:+.2%}")
        chart = go.Figure(go.Candlestick(
            x=session.index, open=session["Open"], high=session["High"],
            low=session["Low"], close=session["Close"],
            increasing_line_color="#39c9aa", decreasing_line_color="#ff737f",
        ))
        chart.update_layout(
            template="plotly_dark", paper_bgcolor="#0b1422", plot_bgcolor="#0b1422",
            height=420, xaxis_rangeslider_visible=False, yaxis_title="USD",
            xaxis_title="New York time", margin={"l": 0, "r": 0, "t": 12, "b": 0},
        )
        st.plotly_chart(chart, use_container_width=True)
        if "Volume" in session and session["Volume"].sum() > 0:
            st.bar_chart(session["Volume"].rename("Shares traded per bar"), color="#4f9fea")
    else:
        st.warning(f"Yahoo Finance returned no five-minute bars for {ticker}.")

    st.subheader("Sector performance")
    sectors = quote_table(SECTORS, bars, daily)
    st.dataframe(sectors.sort_values("Session move", ascending=False).round(2), hide_index=True)
    with st.expander("What do these market numbers mean?"):
        st.markdown(
            """
- **Session move** compares the latest reported bar close with the previous daily close.
  Use the displayed bar time to distinguish an active session from the last available session.
- **Candles** show the open, high, low, and close of each five-minute bar. The latest
  candle can still be forming.
- **Gap to fair value** compares market price with the value you entered. A −10% gap
  means the current price is 10% below your estimate.
- **Sector performance** compares sector ETF moves. VIX is an index of expected volatility,
  so its percentage move describes changing implied volatility rather than an equity return.
            """
        )


def draw_factor_guide():
    with st.expander("What do these factor-model numbers mean?", expanded=True):
        st.markdown(
            """
| Metric | How to read it |
|---|---|
| **Market beta** | A beta of 1.25 means a one-percentage-point rise in the market excess return was associated with about a 1.25-point increase in the asset's excess return, holding the other included factors fixed. |
| **Alpha** | Average daily excess return left unexplained by the fitted factors. The annualized figure multiplies daily alpha by 252. |
| **t-statistic** | Estimate divided by its HAC standard error. A larger absolute t-stat indicates stronger evidence against a zero coefficient under the model assumptions. |
| **p-value and 95% interval** | A p-value below 0.05 and an interval excluding zero indicate significance at the 5% level for the reported coefficient test. |
| **Adjusted R²** | How much variation in daily excess returns the model explains, adjusted for the number of predictors. |
| **Residual volatility** | Annualized variability in returns left unexplained by the model. |
| **Relative-pricing score** | Positive means the asset outperformed its factor-implied return. Negative means it underperformed. A score of +2 is about two historical residual standard deviations above the factor baseline. |
| **Rolling beta** | Shows how market sensitivity changes using the most recent 126 aligned sessions at each point. |

**CAPM** uses market exposure. **FF3** adds size (SMB) and value (HML).
**FF5** also adds profitability (RMW) and investment (CMA).
The relative-pricing score measures historical return deviation. The dashboard shows the
last matched factor date so you can see any publication lag.
            """
        )


def draw_news():
    st.subheader("Company & market news")
    frames = []
    for symbol in dict.fromkeys([ticker, "SPY", "QQQ"]):
        rows, ok = headlines(symbol)
        if not ok or not rows:
            st.caption(f"No headlines returned for {symbol}.")
        if rows:
            frames.append(pd.DataFrame(rows).assign(Feed=symbol))
    if frames:
        news = pd.concat(frames, ignore_index=True)
        news["Feed"] = news.groupby("URL")["Feed"].transform(
            lambda values: ", ".join(dict.fromkeys(values))
        )
        news["Published"] = pd.to_datetime(news["Published"], utc=True).dt.tz_convert(ZURICH)
        st.dataframe(
            news.drop_duplicates("URL").sort_values("Published", ascending=False)
            [["Published", "Feed", "Headline", "Publisher", "URL"]],
            hide_index=True,
            column_config={"URL": st.column_config.LinkColumn("Article", display_text="Open")},
        )
    st.caption("Headlines are market context. Forecasts use timestamped price data and benchmark returns.")


@st.fragment(run_every="5m" if auto_refresh else None)
def research_terminal():
    if st.button("Refresh feeds now"):
        prices.clear()
        headlines.clear()
        french_factors.clear()
        estimate_forecast.clear()

    if page == "Overview":
        draw_market()
        return
    if page == "Factor models":
        render_factors()
        draw_factor_guide()
        return
    if page == "News":
        draw_news()
        return

    st.subheader(f"{ticker} · {segment} forecast")
    symbols = tuple(dict.fromkeys([ticker, "SPY", "QQQ"]))
    if intraday_forecast:
        feed, _ = prices(symbols, "1mo", "5m", True)
        close_series = {symbol: completed_bars(frame)["Close"] for symbol, frame in feed.items()}
        horizon = 6
    else:
        feed, _ = prices(symbols, "5y", "1d", True)
        close_series = {symbol: daily_close(frame) for symbol, frame in feed.items()}
        horizon = int(segment.split()[0])

    if any(symbol not in close_series for symbol in symbols):
        st.warning("The selected ticker or one of the benchmark feeds is unavailable.")
        return

    with st.spinner("Evaluating the models against historical prices…"):
        result = estimate_forecast(
            close_series[ticker], close_series["SPY"], close_series["QQQ"],
            horizon, intraday_forecast,
        )
    draw_forecast(result, intraday_forecast, threshold)


research_terminal()
