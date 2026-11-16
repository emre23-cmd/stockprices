import re
from datetime import datetime, time
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
from statsmodels.regression.rolling import RollingOLS

st.set_page_config(page_title="Asset Pricing Terminal", page_icon="📈", layout="wide")

NY = ZoneInfo("America/New_York")
ZURICH = ZoneInfo("Europe/Zurich")
MARKETS = {
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq-100",
    "^IXIC": "Nasdaq Composite",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
}
SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLI": "Industrials", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
}
MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}
FACTOR_FILES = {
    "FF3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "FF5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def prices(symbols, period, interval, adjusted):
    fetched = pd.Timestamp.now(tz="UTC")
    try:
        raw = yf.download(
            list(symbols), period=period, interval=interval,
            auto_adjust=adjusted, prepost=False, group_by="ticker",
            multi_level_index=True, progress=False, threads=4, timeout=10,
        )
    except Exception:
        return {}, fetched

    result = {}
    if raw is None or raw.empty:
        return result, fetched
    for symbol in symbols:
        if symbol not in raw.columns.get_level_values(0):
            continue
        frame = raw[symbol].dropna(subset=["Close"]).sort_index()
        if not frame.empty:
            result[symbol] = frame.loc[~frame.index.duplicated(keep="last")]
    return result, fetched


@st.cache_data(ttl=21600, max_entries=2, show_spinner=False)
def french_factors(dataset):
    response = requests.get(FRENCH_BASE + FACTOR_FILES[dataset], timeout=15)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        filename = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        lines = archive.read(filename).decode("latin-1").splitlines()

    header = next(line for line in lines if "Mkt-RF" in line and "RF" in line)
    rows = [line for line in lines if re.match(r"^\s*\d{8}\s*,", line)]
    if not rows:
        raise ValueError("The factor file contains no daily rows.")
    frame = pd.read_csv(StringIO("\n".join(["Date" + header.strip(), *rows])))
    frame.columns = frame.columns.str.strip()
    frame.index = pd.to_datetime(frame.pop("Date").astype(str), format="%Y%m%d")
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999], np.nan)
    return frame.sort_index() / 100


def daily_close(frame):
    close = frame["Close"].copy()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index = close.index.normalize()
    # Exclude an open US session from daily regressions and return baselines.
    now = datetime.now(NY)
    if now.time() < time(16, 15):
        close = close.loc[close.index < pd.Timestamp(now.date())]
    return close


@st.cache_data(ttl=300, max_entries=128, show_spinner=False)
def headlines(symbol):
    try:
        items = yf.Ticker(symbol).get_news(count=12)
        rows = []
        for item in items or []:
            article = item.get("content") or item
            title = str(article.get("title") or "").strip()
            link = (
                (article.get("canonicalUrl") or {}).get("url")
                or (article.get("clickThroughUrl") or {}).get("url")
                or article.get("link", "")
            )
            url = urlparse(link)
            if not title or url.scheme not in {"https", "http"} or not url.netloc:
                continue
            published = article.get("pubDate") or article.get("providerPublishTime")
            stamp = (
                pd.to_datetime(published, unit="s", utc=True, errors="coerce")
                if isinstance(published, (int, float))
                else pd.to_datetime(published, utc=True, errors="coerce")
            )
            publisher = (article.get("provider") or {}).get("displayName")
            rows.append({
                "Headline": title, "Publisher": publisher or article.get("publisher", ""),
                "Published": stamp, "URL": link,
            })
        return rows, True
    except Exception:
        return [], False


def session_bars(frame):
    index = frame.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    local = frame.set_axis(index.tz_convert(NY))
    return local.loc[local.index.date == local.index[-1].date()]


def quote_table(labels, intraday, history):
    records = []
    for symbol, name in labels.items():
        intr = intraday.get(symbol)
        daily = history.get(symbol)
        change = np.nan
        if intr is None:
            records.append({
                "Instrument": name, "Ticker": symbol, "Price": np.nan,
                "Session move": np.nan, "Latest bar (ET)": "Unavailable",
            })
            continue
        bars = session_bars(intr)
        stamp = bars.index[-1]
        quote = float(bars["Close"].iloc[-1])
        if daily is not None:
            prior = daily_close(daily)
            prior = prior.loc[prior.index < pd.Timestamp(stamp.date())]
            if not prior.empty and prior.iloc[-1] > 0:
                change = 100 * (quote / prior.iloc[-1] - 1)
        records.append({
            "Instrument": name, "Ticker": symbol, "Price": quote,
            "Session move": change, "Latest bar (ET)": stamp.strftime("%b %d %H:%M"),
        })
    return pd.DataFrame(records)


def fit_model(sample, columns):
    y = sample["Return"] - sample["RF"]
    x = sm.add_constant(sample[columns], has_constant="add")
    return sm.OLS(y, x).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": True},
        use_t=True,
    )


def relative_score(sample, columns, days):
    train = sample.iloc[:-days]
    recent = sample.iloc[-days:]
    model = fit_model(train, columns)
    x = sm.add_constant(recent[columns], has_constant="add")
    baseline = recent["RF"] + x[columns] @ model.params[columns]
    if (baseline <= -1).any() or (recent["Return"] <= -1).any():
        return None

    # Use non-overlapping, equally long training blocks to estimate dispersion.
    residual = np.log1p(train["Return"]) - np.log1p(
        train["RF"] + train[columns] @ model.params[columns]
    )
    if not np.isfinite(residual).all():
        return None
    block_count = len(residual) // days
    if block_count < 10:
        return None
    blocks = residual.to_numpy()[-block_count * days:].reshape(-1, days).sum(axis=1)
    scale = blocks.std(ddof=1)
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    observed = np.log1p(recent["Return"]).sum()
    implied = np.log1p(baseline).sum()
    return (observed - implied) / scale, np.expm1(observed), np.expm1(implied)


st.title("Asset Pricing Terminal")
st.caption("US equities and ETFs · New York market time · Five-minute feed refresh")

with st.sidebar:
    ticker = st.text_input("US ticker", "NVDA", max_chars=15).strip().upper()
    years = st.selectbox("Daily regression history", [1, 2, 3, 5], index=2)
    window = st.selectbox("Relative pricing window", [5, 10, 20], index=1)
    chosen_model = st.radio("Active model", list(MODELS), index=2, horizontal=True)
    auto_refresh = st.toggle("Refresh every five minutes", value=True)
    fair = st.number_input("Your fair value per share (USD)", min_value=0.0, step=1.0)
    st.caption("Use a US-listed USD stock or ETF. Leave fair value at zero to skip the comparison.")

if not re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)?", ticker):
    st.warning("Enter a US stock or ETF ticker such as AAPL, BRK-B, SPY, or QQQ.")
    st.stop()


@st.fragment(run_every="5m" if auto_refresh else None)
def dashboard():
    if st.button("Refresh feeds now"):
        prices.clear()
        headlines.clear()

    symbols = tuple(dict.fromkeys([ticker, *MARKETS, *SECTORS]))
    intraday, checked = prices(symbols, "5d", "5m", False)
    quote_history, _ = prices(symbols, "1mo", "1d", False)
    previous = st.session_state.get("last_bars", {})
    st.session_state["last_bars"] = {**previous, **intraday}
    missing = set(symbols) - set(intraday)
    intraday = {
        symbol: st.session_state["last_bars"][symbol]
        for symbol in symbols
        if symbol in st.session_state["last_bars"]
    }
    retained = sorted(missing & intraday.keys())
    if retained:
        st.warning("Last available bars shown for " + ", ".join(retained))

    st.caption(f"Yahoo request at {checked.tz_convert(ZURICH):%H:%M:%S} Zurich time. "
               "Check the bar time before using the displayed price.")
    live, factors_tab, news = st.tabs(["Market & price", "Factor analysis", "Public news"])

    with live:
        st.subheader("US market")
        st.dataframe(
            quote_table(MARKETS, intraday, quote_history),
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Session move": st.column_config.NumberColumn(format="%+.2f%%"),
            },
        )
        st.subheader(ticker)
        if ticker in intraday:
            bars = session_bars(intraday[ticker])
            latest = bars.index[-1]
            price = float(bars["Close"].iloc[-1])
            asset_row = quote_table({ticker: ticker}, intraday, quote_history).iloc[0]
            move = asset_row["Session move"]
            cols = st.columns(3)
            cols[0].metric("Latest bar close", f"${price:,.2f}")
            cols[1].metric("Vs. previous session close", f"{move:+.2f}%" if pd.notna(move) else "—")
            cols[2].metric("Latest bar start (ET)", latest.strftime("%b %d · %H:%M"))
            if fair > 0:
                gap = 100 * (price / fair - 1)
                label = "Above" if gap > 0 else "Below" if gap < 0 else "At"
                st.metric(f"{label} your fair value of ${fair:,.2f}", f"{gap:+.2f}%")
            chart = go.Figure(go.Candlestick(
                x=bars.index, open=bars["Open"], high=bars["High"],
                low=bars["Low"], close=bars["Close"], name=ticker,
            ))
            chart.update_layout(
                height=420, xaxis_rangeslider_visible=False,
                yaxis_title="USD", xaxis_title="Bar start (New York time)",
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
            )
            st.plotly_chart(chart, use_container_width=True)
            if "Volume" in bars and bars["Volume"].sum() > 0:
                st.bar_chart(bars["Volume"].rename("Shares per five-minute bar"))
        else:
            st.info("No five-minute bars were returned for this ticker.")

        sector_moves = quote_table(SECTORS, intraday, quote_history)
        dates = sector_moves["Latest bar (ET)"].str.slice(0, 6)
        latest_sector_dates = [
            session_bars(intraday[symbol]).index[-1].date()
            for symbol in SECTORS if symbol in intraday
        ]
        if latest_sector_dates:
            target = max(latest_sector_dates).strftime("%b %d")
            sectors = sector_moves.loc[dates.eq(target)].dropna(subset=["Session move"])
            if not sectors.empty:
                st.subheader(f"Sector rotation · {target}")
                st.bar_chart(
                    sectors.sort_values("Session move").set_index("Instrument")["Session move"],
                    horizontal=True,
                )

    with factors_tab:
        render_factors()

    with news:
        st.caption("Yahoo Finance publisher headlines · times shown in Zurich")
        all_rows = []
        for symbol in dict.fromkeys([ticker, "SPY", "QQQ"]):
            rows, ok = headlines(symbol)
            if not ok or not rows:
                st.caption(f"No headlines returned for {symbol}.")
            all_rows.extend({**row, "Feed": symbol} for row in rows)
        if all_rows:
            feed = pd.DataFrame(all_rows)
            feed["Feed"] = feed.groupby("URL")["Feed"].transform(
                lambda values: ", ".join(dict.fromkeys(values))
            )
            feed = feed.drop_duplicates("URL").sort_values("Published", ascending=False)
            feed["Published"] = pd.to_datetime(feed["Published"], utc=True).dt.tz_convert(ZURICH)
            st.dataframe(
                feed[["Published", "Feed", "Headline", "Publisher", "URL"]],
                hide_index=True,
                column_config={"URL": st.column_config.LinkColumn("Article", display_text="Open")},
            )


def render_factors():
    st.subheader(f"{ticker} factor exposures")
    daily, _ = prices((ticker,), f"{years}y", "1d", True)
    if ticker not in daily:
        st.warning("Adjusted daily Yahoo Finance history is unavailable.")
        return

    try:
        ff3, ff5 = french_factors("FF3"), french_factors("FF5")
    except (requests.RequestException, ValueError, KeyError, StopIteration, OSError) as exc:
        st.warning(f"Could not load the Kenneth French datasets ({type(exc).__name__}).")
        return

    close = daily_close(daily[ticker])
    returns = close.pct_change(fill_method=None).rename("Return")
    # Align both publications and calculate every regression on the same dates.
    common = (
        returns.to_frame()
        .join(ff3[MODELS["FF3"] + ["RF"]].add_suffix("_3"), how="inner")
        .join(ff5[MODELS["FF5"] + ["RF"]].add_suffix("_5"), how="inner")
        .dropna()
    )
    if len(common) < 60:
        st.warning(f"Only {len(common)} aligned daily observations. At least 60 are required.")
        return

    st.caption(
        f"{len(common):,} common daily observations · "
        f"{common.index.min():%Y-%m-%d} to {common.index.max():%Y-%m-%d}. "
        f"Latest adjusted price date: {close.index.max():%Y-%m-%d}. "
        "Official daily factors can lag Yahoo, so the relative pricing signal can lag the quote."
    )

    summaries, coefficients, samples = [], [], {}
    for name, columns in MODELS.items():
        suffix = "_5" if name == "FF5" else "_3"
        factor_names = columns + ["RF"]
        sample = common[["Return", *[factor + suffix for factor in factor_names]]]
        sample = sample.rename(columns={factor + suffix: factor for factor in factor_names})
        samples[name] = sample
        model = fit_model(sample, columns)
        ci = model.conf_int()

        summaries.append({
            "Model": name, "Market beta": model.params["Mkt-RF"],
            "Beta t-stat vs 0": model.tvalues["Mkt-RF"],
            "Alpha × 252 (%)": 25200 * model.params["const"],
            "Alpha t-stat": model.tvalues["const"],
            "Alpha p-value": model.pvalues["const"],
            "Adjusted R²": model.rsquared_adj,
            "Residual vol. (%)": 100 * np.sqrt(252) * model.resid.std(ddof=1),
            "Observations": int(model.nobs),
        })
        for term in model.params.index:
            multiplier = 10000 if term == "const" else 1
            coefficients.append({
                "Model": name, "Term": "Alpha (bp/day)" if term == "const" else term,
                "Estimate": multiplier * model.params[term],
                "HAC SE": multiplier * model.bse[term], "t-stat vs 0": model.tvalues[term],
                "p-value": model.pvalues[term],
                "95% lower": multiplier * ci.loc[term, 0],
                "95% upper": multiplier * ci.loc[term, 1],
            })

    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(coefficients)
    st.dataframe(summary.round(4), hide_index=True)
    st.dataframe(detail.round(4), hide_index=True)
    st.download_button(
        "Download coefficients as CSV", detail.to_csv(index=False),
        file_name=f"{ticker}_factor_coefficients.csv", mime="text/csv",
    )

    selected = samples[chosen_model]
    leverage = MODELS[chosen_model]
    if len(selected) >= 126:
        x = sm.add_constant(selected[leverage], has_constant="add")
        rolling = RollingOLS(selected["Return"] - selected["RF"], x, window=126)
        beta = rolling.fit(params_only=True).params["Mkt-RF"].dropna()
        st.subheader(f"126-session rolling market beta · {chosen_model}")
        st.line_chart(beta.rename("Market beta"))

    st.subheader(f"Relative pricing · {chosen_model}")
    score = relative_score(selected, leverage, window)
    if score is None:
        st.info("This window needs at least ten comparable historical blocks before a score is shown.")
        return

    z, actual, implied = score
    label = "Rich vs. factors" if z >= 2 else "Cheap vs. factors" if z <= -2 else "Within usual range"
    cols = st.columns(3)
    cols[0].metric(label, f"{z:+.2f}σ")
    cols[1].metric(f"Actual return · {window} sessions", f"{actual:+.2%}")
    cols[2].metric("Factor-implied return", f"{implied:+.2%}")
    st.caption(
        f"Window ends {selected.index[-1]:%Y-%m-%d}. The baseline uses realized factors "
        "and exposures fitted before this window. The ±2σ bands use historical residual "
        "dispersion. This measures relative performance and has no intrinsic-price target."
    )


dashboard()
