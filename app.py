import re
from io import BytesIO, StringIO
from urllib.parse import urlparse
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.rolling import RollingOLS


st.set_page_config(
    page_title="US Equity Research Terminal",
    page_icon="📈",
    layout="wide",
)

NY = ZoneInfo("America/New_York")
ZURICH = ZoneInfo("Europe/Zurich")
BAR = pd.Timedelta(minutes=5)

MARKETS = {
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq-100",
    "^IXIC": "Nasdaq Composite",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
}

SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

UNIVERSE = {
    "Technology & Semiconductors": (
        "AAPL:Apple,MSFT:Microsoft,NVDA:Nvidia,AMD:AMD,AVGO:Broadcom,"
        "ORCL:Oracle,CRM:Salesforce,ADBE:Adobe,INTC:Intel,QCOM:Qualcomm,"
        "MU:Micron,AMAT:Applied Materials,LRCX:Lam Research,KLAC:KLA,"
        "NOW:ServiceNow,PLTR:Palantir,CSCO:Cisco,DELL:Dell,IBM:IBM,"
        "PANW:Palo Alto Networks,CRWD:CrowdStrike,TSM:TSMC ADR,ASML:ASML ADR,"
        "ARM:Arm,SNPS:Synopsys,CDNS:Cadence,ANET:Arista,NET:Cloudflare,"
        "DDOG:Datadog,SNOW:Snowflake,FTNT:Fortinet,ACN:Accenture,TXN:Texas Instruments"
    ),
    "Internet, Media & Consumer": (
        "GOOGL:Alphabet Class A,GOOG:Alphabet Class C,META:Meta,AMZN:Amazon,"
        "NFLX:Netflix,TSLA:Tesla,UBER:Uber,DASH:DoorDash,ABNB:Airbnb,"
        "BKNG:Booking Holdings,WMT:Walmart,COST:Costco,HD:Home Depot,"
        "LOW:Lowe's,NKE:Nike,MCD:McDonald's,SBUX:Starbucks,KO:Coca-Cola,"
        "PEP:PepsiCo,PG:Procter & Gamble,DIS:Disney,SPOT:Spotify,SHOP:Shopify,"
        "CMCSA:Comcast,GM:General Motors,F:Ford,TGT:Target,TJX:TJX,"
        "EL:Estée Lauder,CL:Colgate-Palmolive,PM:Philip Morris,MO:Altria"
    ),
    "Financials & Payments": (
        "JPM:JPMorgan,BAC:Bank of America,GS:Goldman Sachs,MS:Morgan Stanley,"
        "C:Citigroup,WFC:Wells Fargo,BLK:BlackRock,BRK-B:Berkshire Hathaway,"
        "V:Visa,MA:Mastercard,AXP:American Express,SCHW:Charles Schwab,"
        "KKR:KKR,BX:Blackstone,APO:Apollo,COIN:Coinbase,HOOD:Robinhood,"
        "CME:CME Group,ICE:Intercontinental Exchange,SPGI:S&P Global,"
        "MCO:Moody's,CB:Chubb,PGR:Progressive,AIG:AIG,PYPL:PayPal"
    ),
    "Health Care": (
        "LLY:Eli Lilly,UNH:UnitedHealth,JNJ:Johnson & Johnson,ABBV:AbbVie,"
        "MRK:Merck,PFE:Pfizer,TMO:Thermo Fisher,ISRG:Intuitive Surgical,"
        "AMGN:Amgen,VRTX:Vertex,GILD:Gilead,DHR:Danaher,ABT:Abbott,"
        "MDT:Medtronic,BSX:Boston Scientific,REGN:Regeneron,MRNA:Moderna,"
        "CVS:CVS Health,CI:Cigna,HUM:Humana,NVO:Novo Nordisk ADR"
    ),
    "Industrials, Energy & Materials": (
        "GE:GE Aerospace,BA:Boeing,CAT:Caterpillar,DE:Deere,HON:Honeywell,"
        "RTX:RTX,LMT:Lockheed Martin,UNP:Union Pacific,UPS:UPS,FDX:FedEx,"
        "ETN:Eaton,XOM:ExxonMobil,CVX:Chevron,COP:ConocoPhillips,"
        "SLB:SLB,NEE:NextEra Energy,CEG:Constellation Energy,LIN:Linde,"
        "FCX:Freeport-McMoRan,NEM:Newmont,GEV:GE Vernova,WM:Waste Management,"
        "EMR:Emerson,MMM:3M,OXY:Occidental,EOG:EOG Resources,"
        "MPC:Marathon Petroleum,PSX:Phillips 66,DUK:Duke Energy,SO:Southern"
    ),
    "Broad Market & Style ETFs": (
        "SPY:S&P 500,VOO:Vanguard S&P 500,IVV:iShares S&P 500,"
        "QQQ:Nasdaq-100,QQQM:Invesco Nasdaq-100,IWM:Russell 2000,"
        "IWV:Russell 3000,VTI:Total US Market,DIA:Dow Jones,VT:Total World,"
        "VXUS:International Equity,EFA:Developed Markets,EEM:Emerging Markets,"
        "VEA:Developed ex-US,VWO:Emerging Markets,RSP:Equal-Weight S&P 500,"
        "SCHD:Dividend Equity,VUG:US Growth,VTV:US Value,MTUM:US Momentum,"
        "QUAL:US Quality,USMV:Minimum Volatility,AVUV:US Small Cap Value"
    ),
    "Sector & Thematic ETFs": (
        ",".join(f"{symbol}:{name}" for symbol, name in SECTORS.items())
        + ",SMH:Semiconductors,SOXX:Semiconductors,IGV:Software,XBI:Biotech,"
        "KRE:Regional Banks,ITA:Aerospace & Defense,JETS:Airlines,"
        "ARKK:Innovation,TAN:Solar,ICLN:Clean Energy,URA:Uranium,"
        "XME:Metals & Mining,GDX:Gold Miners,VNQ:US Real Estate"
    ),
    "Bonds, Commodities & Digital Assets": (
        "TLT:Long Treasuries,IEF:Intermediate Treasuries,SHY:Short Treasuries,"
        "SGOV:Treasury Bills,BND:Total US Bonds,AGG:US Aggregate Bonds,"
        "LQD:Investment Grade Credit,HYG:High Yield Credit,TIP:Inflation Protected,"
        "GLD:Gold,IAU:Gold,SLV:Silver,USO:Oil,DBC:Commodities,"
        "IBIT:Spot Bitcoin,FBTC:Spot Bitcoin,ETHA:Spot Ether"
    ),
}

FACTOR_MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}
FACTOR_FILES = {
    "FF3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "FF5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

HORIZONS = {
    "30 minutes": (6, True),
    "1 session": (1, False),
    "5 sessions": (5, False),
    "20 sessions": (20, False),
}


# Data feeds

@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def yahoo_prices(symbols, period, interval, adjusted):
    try:
        raw = yf.download(
            tickers=list(symbols),
            period=period,
            interval=interval,
            auto_adjust=adjusted,
            group_by="ticker",
            multi_level_index=True,
            prepost=False,
            progress=False,
            threads=False,
            timeout=10,
        )
        checked = pd.Timestamp.now(tz="UTC")
        if raw is None or raw.empty:
            return {}, checked

        frames = {}
        for symbol in symbols:
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(0):
                    continue
                frame = raw[symbol]
            elif len(symbols) == 1:
                frame = raw
            else:
                continue
            if "Close" not in frame:
                continue
            frame = frame.dropna(subset=["Close"]).sort_index()
            if not frame.empty:
                frames[symbol] = frame.loc[~frame.index.duplicated(keep="last")]
        return frames, checked
    except Exception:
        return {}, pd.Timestamp.now(tz="UTC")


def to_new_york(frame):
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    return frame.set_axis(index.tz_convert(NY))


def completed_bars(frame):
    local = to_new_york(frame)
    return local.loc[local.index + BAR <= pd.Timestamp.now(tz=NY)]


def daily_close(frame):
    close = frame["Close"].copy()
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    now = pd.Timestamp.now(tz=NY)
    # Allow the US daily bar time to settle after the regular close.
    if (now.hour, now.minute) < (16, 15):
        close = close.loc[close.index < pd.Timestamp(now.date())]
    return close.loc[~close.index.duplicated(keep="last")]


@st.cache_data(ttl=21600, max_entries=2, show_spinner=False)
def french_factors(dataset):
    response = requests.get(FRENCH_URL + FACTOR_FILES[dataset], timeout=15)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        filename = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        lines = archive.read(filename).decode("latin-1").splitlines()

    header = next(line.strip() for line in lines if "Mkt-RF" in line and "RF" in line)
    fields = [field.strip() for field in header.split(",") if field.strip()]
    rows = [line for line in lines if re.match(r"^\s*\d{8}\s*,", line)]
    if not rows:
        raise ValueError("No daily factor observations.")

    frame = pd.read_csv(StringIO("\n".join(rows)), header=None, names=["Date", *fields])
    frame.index = pd.to_datetime(frame.pop("Date").astype(str), format="%Y%m%d")
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999], np.nan)
    return frame.sort_index() / 100


@st.cache_data(ttl=300, max_entries=128, show_spinner=False)
def yahoo_news(ticker):
    try:
        articles = yf.Ticker(ticker).get_news(count=12) or []
    except Exception:
        return []

    rows = []
    for item in articles:
        if not isinstance(item, dict):
            continue
        article = item.get("content") or item
        canonical = article.get("canonicalUrl") or {}
        clickthrough = article.get("clickThroughUrl") or {}
        link = canonical.get("url") or clickthrough.get("url") or article.get("link")
        title = article.get("title")
        if not isinstance(link, str) or not title:
            continue
        url = urlparse(link)
        if url.scheme not in {"http", "https"} or not url.netloc:
            continue

        published = article.get("pubDate") or article.get("providerPublishTime")
        if isinstance(published, (int, float)):
            stamp = pd.to_datetime(published, unit="s", utc=True, errors="coerce")
        else:
            stamp = pd.to_datetime(published, utc=True, errors="coerce")
        provider = article.get("provider") or {}
        rows.append({
            "Published": stamp,
            "Ticker": ticker,
            "Headline": title,
            "Publisher": provider.get("displayName") or article.get("publisher", ""),
            "Article": link,
        })
    return rows


# Market dashboard

def quote_snapshot(labels, intraday, daily):
    rows = []
    for symbol, label in labels.items():
        frame = intraday.get(symbol)
        price, change = np.nan, np.nan
        stamp = pd.NaT
        if frame is not None:
            frame = to_new_york(frame)
            price = float(frame["Close"].iloc[-1])
            stamp = frame.index[-1]
            if symbol in daily:
                previous = daily_close(daily[symbol])
                previous = previous.loc[previous.index < pd.Timestamp(stamp.date())]
                if not previous.empty and previous.iloc[-1] > 0:
                    change = 100 * (price / previous.iloc[-1] - 1)
        rows.append({
            "Instrument": label,
            "Ticker": symbol,
            "Price": price,
            "Session move %": change,
            "Latest bar (ET)": stamp,
        })
    return pd.DataFrame(rows)


def price_chart(session, ticker):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.04,
    )
    fig.add_trace(go.Candlestick(
        x=session.index,
        open=session["Open"],
        high=session["High"],
        low=session["Low"],
        close=session["Close"],
        increasing_line_color="#32c7a7",
        decreasing_line_color="#fa7683",
        name=ticker,
    ), row=1, col=1)
    if "Volume" in session and session["Volume"].fillna(0).sum() > 0:
        colors = np.where(session["Close"] >= session["Open"], "#32c7a7", "#fa7683")
        fig.add_trace(go.Bar(
            x=session.index,
            y=session["Volume"],
            marker_color=colors,
            opacity=0.65,
            name="Volume",
        ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0c1625",
        plot_bgcolor="#0c1625",
        height=510,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=16, r=20, t=16, b=20),
    )
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="New York time", row=2, col=1)
    return fig


def render_market(ticker, fair_value):
    labels = {**MARKETS, **SECTORS, ticker: ticker}
    symbols = tuple(labels)
    intraday, checked = yahoo_prices(symbols, "5d", "5m", False)
    daily, _ = yahoo_prices(symbols, "1mo", "1d", False)

    st.subheader("US market")
    market = quote_snapshot(MARKETS, intraday, daily)
    for card, row in zip(st.columns(len(MARKETS)), market.to_dict("records")):
        price = row["Price"]
        change = row["Session move %"]
        color = "off" if row["Ticker"] == "^VIX" else "normal"
        card.metric(
            row["Instrument"],
            "—" if pd.isna(price) else f"{price:,.2f}",
            None if pd.isna(change) else f"{change:+.2f}%",
            delta_color=color,
        )

    st.caption(
        f"Yahoo request completed {checked.tz_convert(ZURICH):%b %d, %H:%M:%S} Zurich time. "
        "The feed can be delayed. See the reported bar times below."
    )
    with st.expander("Index timestamps"):
        st.dataframe(market.round(2), hide_index=True)

    st.subheader(f"{ticker} · Five-minute price and volume")
    if ticker in intraday:
        frame = to_new_york(intraday[ticker])
        last_bar = frame.index[-1]
        session = frame.loc[frame.index.date == last_bar.date()]
        price = float(session["Close"].iloc[-1])
        move = quote_snapshot({ticker: ticker}, intraday, daily)["Session move %"].iloc[0]
        cards = st.columns(4)
        cards[0].metric("Latest bar close", f"${price:,.2f}")
        cards[1].metric("Session move", "—" if pd.isna(move) else f"{move:+.2f}%")
        cards[2].metric("Latest bar start (ET)", last_bar.strftime("%b %d · %H:%M"))
        if fair_value > 0:
            cards[3].metric("Gap to your fair value", f"{price / fair_value - 1:+.2%}")
        if {"Open", "High", "Low"}.issubset(session.columns):
            st.plotly_chart(price_chart(session, ticker), use_container_width=True)
    else:
        st.warning(f"Yahoo Finance returned no five-minute bars for {ticker}.")

    st.subheader("Sector performance")
    sectors = quote_snapshot(SECTORS, intraday, daily)
    st.dataframe(
        sectors.sort_values("Session move %", ascending=False).round(2),
        hide_index=True,
    )

    with st.expander("What do these market numbers mean?", expanded=True):
        st.markdown("""
- **Session move** compares the latest reported price with the previous daily close.
- **Five-minute candle** shows the open, high, low, and close for that interval. The latest candle may still be forming.
- **VIX** tracks expected S&P 500 volatility implied by options. A higher VIX indicates higher implied volatility.
- **Sector performance** shows the move of each sector ETF, with a separate timestamp for each feed.
- **Gap to fair value** uses the estimate you enter. If the price is $90 and your estimate is $100, the app displays −10%.
        """)


# Factor models

def fit_factor_model(sample, factors):
    x = sm.add_constant(sample[factors], has_constant="add")
    y = sample["Return"] - sample["RF"]
    return sm.OLS(y, x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 5, "use_correction": True},
        use_t=True,
    )


def relative_pricing(sample, factors, window):
    train, recent = sample.iloc[:-window], sample.iloc[-window:]
    block_count = len(train) // window
    if block_count < 10:
        return None

    fitted = fit_factor_model(train, factors)
    train_baseline = train["RF"] + train[factors] @ fitted.params[factors]
    recent_baseline = recent["RF"] + recent[factors] @ fitted.params[factors]
    if (
        (train_baseline <= -1).any()
        or (recent_baseline <= -1).any()
        or (sample["Return"] <= -1).any()
    ):
        return None

    residuals = np.log1p(train["Return"]) - np.log1p(train_baseline)
    blocks = residuals.to_numpy()[-block_count * window:].reshape(block_count, window)
    dispersion = blocks.sum(axis=1).std(ddof=1)
    if not np.isfinite(dispersion) or dispersion <= 1e-12:
        return None

    actual = np.log1p(recent["Return"]).sum()
    implied = np.log1p(recent_baseline).sum()
    return (actual - implied) / dispersion, np.expm1(actual), np.expm1(implied)


def render_factors(ticker, years, active_model, window):
    st.subheader(f"{ticker} · Factor exposures")
    history, _ = yahoo_prices((ticker,), f"{years}y", "1d", True)
    if ticker not in history:
        st.warning("Adjusted daily price history is unavailable.")
        return

    try:
        ff3 = french_factors("FF3")[FACTOR_MODELS["FF3"] + ["RF"]]
        ff5 = french_factors("FF5")[FACTOR_MODELS["FF5"] + ["RF"]]
    except Exception:
        st.warning("The Kenneth French daily factor datasets could not be loaded.")
        return

    close = daily_close(history[ticker])
    returns = close.pct_change(fill_method=None).rename("Return")
    common = pd.concat(
        [returns, ff3.add_suffix("_3"), ff5.add_suffix("_5")],
        axis=1,
        join="inner",
    ).dropna()
    if len(common) < 60:
        st.warning(f"Found {len(common)} matched daily observations. At least 60 are required.")
        return

    st.caption(
        f"{len(common):,} matched observations · "
        f"{common.index[0]:%Y-%m-%d} to {common.index[-1]:%Y-%m-%d} · "
        f"Latest Yahoo daily close {close.index[-1]:%Y-%m-%d}. "
        "The factor files can lag the price feed."
    )

    summaries, coefficients, samples = [], [], {}
    for name, factors in FACTOR_MODELS.items():
        suffix = "_5" if name == "FF5" else "_3"
        mapping = {f"{term}{suffix}": term for term in [*factors, "RF"]}
        sample = common[["Return", *mapping]].rename(columns=mapping)
        samples[name] = sample
        fit = fit_factor_model(sample, factors)
        ci = fit.conf_int()
        summaries.append({
            "Model": name,
            "Market beta": fit.params["Mkt-RF"],
            "Beta t-stat vs 0": fit.tvalues["Mkt-RF"],
            "Alpha × 252 (%)": 25200 * fit.params["const"],
            "Alpha t-stat": fit.tvalues["const"],
            "Adjusted R²": fit.rsquared_adj,
            "Residual volatility %": 100 * np.sqrt(252) * fit.resid.std(ddof=1),
        })
        for term in fit.params.index:
            scale = 10000 if term == "const" else 1
            coefficients.append({
                "Model": name,
                "Term": "Alpha (bp/day)" if term == "const" else term,
                "Estimate": scale * fit.params[term],
                "HAC standard error": scale * fit.bse[term],
                "t-stat vs 0": fit.tvalues[term],
                "p-value": fit.pvalues[term],
                "95% lower": scale * ci.loc[term, 0],
                "95% upper": scale * ci.loc[term, 1],
            })

    st.dataframe(pd.DataFrame(summaries).round(4), hide_index=True)
    detail = pd.DataFrame(coefficients)
    st.dataframe(detail.loc[detail["Model"].eq(active_model)].round(4), hide_index=True)
    st.download_button(
        "Download all factor estimates",
        detail.to_csv(index=False),
        file_name=f"{ticker}_factor_models.csv",
        mime="text/csv",
    )

    sample = samples[active_model]
    factors = FACTOR_MODELS[active_model]
    if len(sample) >= 126:
        rolling = RollingOLS(
            sample["Return"] - sample["RF"],
            sm.add_constant(sample[factors], has_constant="add"),
            window=126,
        ).fit(params_only=True)
        st.subheader(f"{active_model} · 126-session rolling market beta")
        st.line_chart(rolling.params["Mkt-RF"].dropna().rename("Market beta"))

    st.subheader(f"{active_model} · {window}-session relative performance")
    result = relative_pricing(sample, factors, window)
    if result is None:
        st.info("At least ten usable historical comparison blocks are required for the score.")
    else:
        score, actual, implied = result
        if score >= 2:
            label = "Above factor range"
        elif score <= -2:
            label = "Below factor range"
        else:
            label = "Within factor range"
        cards = st.columns(3)
        cards[0].metric(label, f"{score:+.2f}σ")
        cards[1].metric("Actual return", f"{actual:+.2%}")
        cards[2].metric("Factor-implied return", f"{implied:+.2%}")

    with st.expander("What do these factor numbers mean?", expanded=True):
        st.markdown("""
| Number | How to read it |
|---|---|
| **Market beta** | A beta of 1.25 means market excess returns 1 percentage point higher were associated with asset excess returns about 1.25 points higher, holding the model's other factors fixed. |
| **Alpha** | The average daily excess return left unexplained by the fitted factors. The summary multiplies daily alpha by 252 to put it on an annual scale. |
| **HAC t-stat and p-value** | The t-stat divides the estimate by its HAC standard error. HAC adjusts the uncertainty estimate for changing variance and short-run serial correlation. A smaller p-value provides stronger evidence against a zero coefficient under the model assumptions. |
| **95% confidence interval** | Shows the uncertainty around a coefficient estimate. An interval spanning zero allows both positive and negative effects at this confidence level. |
| **Adjusted R²** | How much variation in daily excess returns the model explains, accounting for the number of factors. |
| **Residual volatility** | Annualized variation in returns left unexplained by the model. |
| **Relative-performance score** | A reading of +2 means performance was about two historical residual standard deviations above the factor-implied return. It measures return deviation. The Overview page compares price with your separate fair-value estimate. |

**CAPM** uses the broad US market excess return. **FF3** adds size (SMB) and value (HML).
**FF5** also adds profitability (RMW) and investment (CMA). For bond, commodity, and crypto
ETFs, these regressions describe exposure to US equity factors.
        """)
        st.markdown("[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)")


# Forecast models

def forecast_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=5)),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=12,
            random_state=42,
            n_jobs=1,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=90,
            max_leaf_nodes=12,
            learning_rate=0.05,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=42,
        ),
    }


def build_features(asset, spy, qqq, intraday):
    prices = pd.concat(
        [asset.rename("Asset"), spy.rename("SPY"), qqq.rename("QQQ")],
        axis=1,
        join="inner",
    ).dropna().sort_index()
    prices = prices.loc[prices.gt(0).all(axis=1)]
    log_prices = np.log(prices)
    returns = log_prices.diff()
    features = pd.DataFrame(index=prices.index)

    for length in [1, 3, 5, 10, 20]:
        for symbol in ["Asset", "SPY", "QQQ"]:
            features[f"{symbol}_{length}"] = log_prices[symbol].diff(length)

    features["Volatility_10"] = returns["Asset"].rolling(10).std()
    features["Volatility_20"] = returns["Asset"].rolling(20).std()
    features["Relative_to_SPY_10"] = features["Asset_10"] - features["SPY_10"]
    features["Relative_to_QQQ_10"] = features["Asset_10"] - features["QQQ_10"]

    if intraday:
        continuous = prices.index.to_series().diff(20).eq(20 * BAR)
        features = features.where(continuous, axis=0)
    return prices["Asset"], features


def balanced_direction(actual, predicted):
    up = actual > 0
    down = actual < 0
    if up.sum() < 5 or down.sum() < 5:
        return np.nan
    return float(((predicted[up] > 0).mean() + (predicted[down] < 0).mean()) / 2)


@st.cache_data(ttl=300, max_entries=48, show_spinner=False)
def estimate_forecast(asset, spy, qqq, horizon, intraday):
    close, features = build_features(asset, spy, qqq, intraday)
    if close.empty or features.iloc[-1].isna().any():
        return None

    target = np.log(close.shift(-horizon) / close)
    position = pd.Series(np.arange(len(close)), index=close.index, name="Position")
    if intraday:
        timestamps = close.index.to_series()
        continuous = (timestamps.shift(-horizon) - timestamps).eq(horizon * BAR)
        target = target.where(continuous)

    data = features.join(target.rename("Target")).join(position).dropna()
    if len(data) < max(180, 30 * horizon):
        return None

    # Hold out the newest 40% and leave a gap before the first test target.
    split = int(len(data) * 0.60)
    test = data.iloc[split::horizon]
    train = data.loc[data["Position"] + horizon < test["Position"].iloc[0]]
    if len(train) < 100 or len(test) < 12:
        return None

    columns = features.columns
    actual = test["Target"].to_numpy()
    validation, current = {}, {}
    for name, model in forecast_models().items():
        model.fit(train[columns], train["Target"])
        validation[name] = model.predict(test[columns])

        # After measuring holdout performance, refit on all known targets.
        model.fit(data[columns], data["Target"])
        current[name] = float(model.predict(features.iloc[[-1]])[0])

    ensemble_test = np.mean(list(validation.values()), axis=0)
    prediction = float(np.mean(list(current.values())))
    agreement = sum(np.sign(value) == np.sign(prediction) for value in current.values())
    validation["Ensemble"] = ensemble_test
    current["Ensemble"] = prediction
    errors = actual - ensemble_test
    lower_error, upper_error = np.quantile(errors, [0.1, 0.9])
    anchor = float(close.iloc[-1])

    rows = []
    for name, values in validation.items():
        projected_return = float(np.expm1(current[name]))
        error = float(np.abs(np.expm1(actual) - np.expm1(values)).mean())
        rows.append({
            "Model": name,
            "Projected price": anchor * (1 + projected_return),
            "Projected return %": 100 * projected_return,
            "Holdout MAE (pp)": 100 * error,
            "Balanced direction %": 100 * balanced_direction(actual, values),
        })

    return {
        "rows": rows,
        "anchor": anchor,
        "as_of": close.index[-1],
        "price": anchor * np.exp(prediction),
        "return": float(np.expm1(prediction)),
        "range": (
            anchor * np.exp(prediction + lower_error),
            anchor * np.exp(prediction + upper_error),
        ),
        "agreement": agreement,
        "accuracy": balanced_direction(actual, ensemble_test),
        "mae": float(np.abs(np.expm1(actual) - np.expm1(ensemble_test)).mean()),
        "baseline_mae": float(np.abs(np.expm1(actual)).mean()),
        "count": len(test),
        "test_start": test.index[0],
        "test_end": test.index[-1],
    }


def render_forecasts(ticker, segment, threshold):
    horizon, intraday = HORIZONS[segment]
    st.subheader(f"{ticker} · {segment} forecast")
    symbols = tuple(dict.fromkeys([ticker, "SPY", "QQQ"]))
    if intraday:
        data, _ = yahoo_prices(symbols, "1mo", "5m", True)
        series = {symbol: completed_bars(frame)["Close"] for symbol, frame in data.items()}
    else:
        data, _ = yahoo_prices(symbols, "5y", "1d", True)
        series = {symbol: daily_close(frame) for symbol, frame in data.items()}

    if any(symbol not in series or series[symbol].empty for symbol in symbols):
        st.warning("The ticker or one of its benchmark feeds is unavailable.")
        return

    with st.spinner("Fitting models and checking historical forecast errors…"):
        result = estimate_forecast(series[ticker], series["SPY"], series["QQQ"], horizon, intraday)

    if result is None:
        st.info(
            "This horizon has too little usable history or the latest shared price "
            "window has gaps. Intraday forecasts need 20 consecutive completed "
            "five-minute intervals before the forecast anchor."
        )
        return

    stamp = result["as_of"]
    now = pd.Timestamp.now(tz=NY)
    if intraday:
        bar_end = stamp + BAR
        target = bar_end + horizon * BAR
        fresh = (
            pd.Timedelta(0) <= now - bar_end <= pd.Timedelta(minutes=15)
            and target.date() == now.date()
            and (target.hour, target.minute) <= (16, 0)
        )
        anchor_text = f"Completed bar ending {bar_end:%b %d, %H:%M} ET"
    else:
        fresh = (pd.Timestamp(now.date()) - stamp).days <= 4
        anchor_text = f"Adjusted daily close on {stamp:%Y-%m-%d}"

    checks = {
        "A recent price anchor": fresh,
        "At least 30 non-overlapping holdout forecasts": result["count"] >= 30,
        "Lower MAE than the flat-price baseline": result["mae"] < result["baseline_mae"],
        "At least 55% balanced directional accuracy": result["accuracy"] >= 0.55,
        "All three models agree on direction": result["agreement"] == 3,
        "Projected move clears your threshold": abs(result["return"]) >= threshold,
    }
    label = "WAIT"
    if all(checks.values()) and result["return"] != 0:
        label = "BUY BIAS" if result["return"] > 0 else "SELL BIAS"

    low, high = result["range"]
    cards = st.columns(4)
    cards[0].metric("Research signal", label)
    cards[1].metric("Projected price", f"${result['price']:,.2f}", f"{result['return']:+.2%}")
    cards[2].metric("Historical error range", f"${low:,.2f} – ${high:,.2f}")
    cards[3].metric("Model agreement", f"{result['agreement']}/3")
    st.caption(f"Anchor ${result['anchor']:,.2f} · {anchor_text}")

    if label == "WAIT":
        failed = [name[0].lower() + name[1:] for name, passed in checks.items() if not passed]
        if failed:
            st.info("Checks still unmet: " + ", ".join(failed) + ".")

    st.dataframe(pd.DataFrame(result["rows"]).round(3), hide_index=True)
    st.caption(
        f"Holdout {result['test_start']:%Y-%m-%d} to {result['test_end']:%Y-%m-%d} · "
        f"{result['count']} forecasts · Flat-price baseline MAE "
        f"{100 * result['baseline_mae']:.3f} percentage points."
    )

    with st.expander("What do these forecast numbers mean?", expanded=True):
        st.markdown("""
- **Projected price** applies the average predicted log return from Ridge, Random Forest, and Gradient Boosting to the displayed price anchor. Inputs are past asset, SPY, and QQQ returns, plus recent volatility.
- **Historical error range** applies the 10th and 90th percentiles of held-out forecast errors to the current projection. Future prices can fall outside the range.
- **MAE** measures the typical size of the return error. A reading of 2.0 means predictions missed by 2 percentage points on average. Compare it with the flat-price baseline shown above.
- **Balanced direction** gives up and down periods equal weight. Always predicting up scores 50% when both directions are represented.
- **Model agreement** counts how many models predict the same direction as the ensemble. Models use the same features, so agreement can reflect common errors.
- **Buy Bias / Sell Bias / Wait** summarizes the checks shown above. The holdout measures forecast performance. It does not measure trading profitability after fees, spreads, and slippage.
        """)


# News and application layout

def render_news(ticker):
    st.subheader(f"{ticker} · Company and market news")
    rows = [
        row
        for symbol in dict.fromkeys([ticker, "SPY", "QQQ"])
        for row in yahoo_news(symbol)
    ]
    if not rows:
        st.info("Yahoo Finance did not return headlines at this time.")
        return

    news = pd.DataFrame(rows)
    news["Ticker"] = news.groupby("Article")["Ticker"].transform(
        lambda values: ", ".join(dict.fromkeys(values))
    )
    news["Published"] = pd.to_datetime(news["Published"], utc=True).dt.tz_convert(ZURICH)
    news = news.drop_duplicates("Article").sort_values("Published", ascending=False)
    st.dataframe(
        news,
        hide_index=True,
        column_config={
            "Article": st.column_config.LinkColumn("Article", display_text="Open"),
        },
    )
    st.caption(
        "Publication times are shown in Zurich time. Headlines provide context "
        "for company and market moves. The forecast models use price history."
    )


def apply_style():
    st.markdown(
        """
<style>
.block-container {max-width: 1450px; padding-top: 2.5rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(126, 156, 190, 0.35);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    background: rgba(85, 139, 200, 0.08);
}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_style()
    st.title("US Equity Research Terminal")
    st.caption("S&P 500, Nasdaq, US-listed stocks and ETFs")

    with st.sidebar:
        group = st.selectbox("Asset group", ["All stocks & ETFs", *UNIVERSE])
        listings = UNIVERSE.values() if group == "All stocks & ETFs" else [UNIVERSE[group]]
        options = dict(entry.split(":", 1) for listing in listings for entry in listing.split(","))
        symbols = list(options)
        selected = st.selectbox(
            "Search company or ticker",
            symbols,
            index=symbols.index("NVDA") if "NVDA" in options else 0,
            format_func=lambda symbol: f"{symbol} · {options[symbol]}",
        )
        ticker = st.text_input("Custom US ticker", max_chars=15).strip().upper() or selected
        if not re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)?", ticker):
            st.warning("Use a US ticker such as AAPL, BRK-B, or QQQ.")
            st.stop()

        page = st.radio("View", ["Overview", "Factor Models", "Forecasts", "Public News"])
        auto_refresh = st.toggle("Refresh every five minutes", value=True)

        fair_value = 0.0
        years, active_model, window = 3, "FF5", 10
        segment, threshold = "1 session", 0.005
        if page == "Overview":
            fair_value = st.number_input(
                "Your fair value estimate (USD)",
                min_value=0.0,
                step=1.0,
                help="Leave at zero to skip the price-to-value comparison.",
            )
        elif page == "Factor Models":
            years = st.selectbox("Daily estimation history (years)", [1, 2, 3, 5], index=2)
            active_model = st.radio("Model for detailed analysis", list(FACTOR_MODELS), index=2)
            window = st.selectbox("Relative-performance window (sessions)", [5, 10, 20], index=1)
        elif page == "Forecasts":
            segment = st.radio("Forecast horizon", list(HORIZONS), index=1)
            threshold = st.number_input(
                "Minimum absolute projected move (%)",
                min_value=0.0,
                max_value=20.0,
                value=0.15 if segment == "30 minutes" else 0.50,
                step=0.05,
                key=f"threshold_{segment}",
            ) / 100

    @st.fragment(run_every="5m" if auto_refresh else None)
    def dashboard():
        if st.button("Refresh feeds now"):
            yahoo_prices.clear()
            yahoo_news.clear()
            french_factors.clear()

        if page == "Overview":
            render_market(ticker, fair_value)
        elif page == "Factor Models":
            render_factors(ticker, years, active_model, window)
        elif page == "Forecasts":
            render_forecasts(ticker, segment, threshold)
        else:
            render_news(ticker)

    dashboard()


if __name__ == "__main__":
    main()
