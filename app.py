import math
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.rolling import RollingOLS


st.set_page_config(page_title="Market Research Terminal", page_icon="📈", layout="wide")

NY = ZoneInfo("America/New_York")
ZURICH = ZoneInfo("Europe/Zurich")
BAR = pd.Timedelta(minutes=5)
BENCHMARKS = ("SPY", "QQQ")
INDEXES = {
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
GROUPS = {
    "Technology": (
        "NVDA:Nvidia|AAPL:Apple|MSFT:Microsoft|AMD:Advanced Micro Devices|"
        "AVGO:Broadcom|ORCL:Oracle|CRM:Salesforce|ADBE:Adobe|INTC:Intel|"
        "MU:Micron|QCOM:Qualcomm|AMAT:Applied Materials|LRCX:Lam Research|"
        "KLAC:KLA|ASML:ASML|TSM:Taiwan Semiconductor|ARM:Arm Holdings|"
        "NOW:ServiceNow|PLTR:Palantir|CRWD:CrowdStrike|PANW:Palo Alto Networks|"
        "ANET:Arista Networks|CSCO:Cisco|IBM:IBM|TXN:Texas Instruments|"
        "SNPS:Synopsys|CDNS:Cadence|NET:Cloudflare|DDOG:Datadog|"
        "SNOW:Snowflake|DELL:Dell Technologies"
    ),
    "Consumer & Communications": (
    "TTWO:Take-Two Interactive|"
    "AMZN:Amazon|GOOGL:Alphabet|META:Meta Platforms|NFLX:Netflix|"
    "TSLA:Tesla|UBER:Uber|ABNB:Airbnb|DASH:DoorDash|"
    "BKNG:Booking Holdings|SPOT:Spotify|SHOP:Shopify|DIS:Walt Disney|"
    "WMT:Walmart|COST:Costco|HD:Home Depot|LOW:Lowe's|"
    "MCD:McDonald's|SBUX:Starbucks|NKE:Nike|TGT:Target|"
    "PG:Procter & Gamble|KO:Coca-Cola|PEP:PepsiCo|TJX:TJX Companies"
        
    ),
    "Financials": (
        "JPM:JPMorgan|BAC:Bank of America|GS:Goldman Sachs|MS:Morgan Stanley|"
        "WFC:Wells Fargo|C:Citigroup|BRK-B:Berkshire Hathaway|BLK:BlackRock|"
        "BX:Blackstone|KKR:KKR|APO:Apollo Global Management|V:Visa|"
        "MA:Mastercard|AXP:American Express|PYPL:PayPal|"
        "COIN:Coinbase|HOOD:Robinhood|SPGI:S&P Global|MCO:Moody's"
    ),
    "Health Care": (
        "LLY:Eli Lilly|UNH:UnitedHealth|JNJ:Johnson & Johnson|"
        "ABBV:AbbVie|MRK:Merck|PFE:Pfizer|AMGN:Amgen|"
        "GILD:Gilead Sciences|VRTX:Vertex Pharmaceuticals|"
        "ISRG:Intuitive Surgical|TMO:Thermo Fisher|"
        "DHR:Danaher|NVO:Novo Nordisk|ABT:Abbott"
    ),
    "Industrials, Energy & Utilities": (
        "GE:GE Aerospace|GEV:GE Vernova|BA:Boeing|CAT:Caterpillar|DE:Deere|"
        "HON:Honeywell|RTX:RTX|LMT:Lockheed Martin|UNP:Union Pacific|"
        "UPS:United Parcel Service|ETN:Eaton|XOM:Exxon Mobil|"
        "CVX:Chevron|COP:ConocoPhillips|SLB:SLB|"
        "CEG:Constellation Energy|NEE:NextEra Energy|"
        "LIN:Linde|FCX:Freeport-McMoRan|NEM:Newmont|WM:Waste Management"
    ),
    "Equity ETFs": (
        "SPY:SPDR S&P 500 ETF|VOO:Vanguard S&P 500 ETF|IVV:iShares Core S&P 500 ETF|"
        "QQQ:Invesco QQQ Trust|QQQM:Invesco Nasdaq-100 ETF|"
        "IWM:iShares Russell 2000 ETF|VTI:Vanguard Total Stock Market ETF|"
        "DIA:SPDR Dow Jones Industrial Average ETF|"
        "RSP:Invesco S&P 500 Equal Weight ETF|VT:Vanguard Total World Stock ETF|"
        "VEA:Vanguard FTSE Developed Markets ETF|VWO:Vanguard Emerging Markets ETF|"
        "SCHD:Schwab US Dividend Equity ETF|VUG:Vanguard Growth ETF|"
        "VTV:Vanguard Value ETF|QUAL:iShares MSCI USA Quality Factor ETF|"
        "SMH:VanEck Semiconductor ETF|SOXX:iShares Semiconductor ETF|"
        "IGV:iShares Expanded Tech Software ETF|XBI:SPDR S&P Biotech ETF|"
        "VNQ:Vanguard Real Estate ETF"
    ),
    "Other ETFs": (
        "TLT:iShares 20+ Year Treasury Bond ETF|IEF:iShares 7-10 Year Treasury Bond ETF|"
        "SGOV:iShares 0-3 Month Treasury Bond ETF|BND:Vanguard Total Bond Market ETF|"
        "LQD:iShares Investment Grade Corporate Bond ETF|"
        "HYG:iShares High Yield Corporate Bond ETF|GLD:SPDR Gold Shares|"
        "SLV:iShares Silver Trust|DBC:Invesco DB Commodity Index Tracking Fund|"
        "IBIT:iShares Bitcoin Trust ETF|FBTC:Fidelity Wise Origin Bitcoin Fund"
    ),
}

CATALOG = {}
for group, entries in GROUPS.items():
    for entry in entries.split("|"):
        symbol, name = entry.split(":", 1)
        CATALOG[symbol] = (name, group)
CATALOG.update({symbol: (f"{name} Select Sector SPDR ETF", "Equity ETFs")
                for symbol, name in SECTORS.items()})

DEFAULT_SCAN = (
    "NVDA AAPL MSFT AMD AVGO GOOGL META AMZN NFLX TSLA JPM GS V "
    "LLY UNH GE CAT XOM CVX COST WMT SPY QQQ IWM RSP VTI SMH XLK XLF XLE"
).split()

MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}
FACTOR_FILES = {
    "FF3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "FF5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}
FACTOR_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

CATALYST = re.compile(
    r"\b(earnings|guidance|profit warning|investigation|antitrust|lawsuit|"
    r"recall|accounting probe|dividend cut|trading halt|bankruptcy|"
    r"credit downgrade|delisting|sec charges)\b",
    re.IGNORECASE,
)


def ticker_label(symbol):
    return f"{symbol} · {CATALOG[symbol][0]}" if symbol in CATALOG else symbol


def now_et():
    return pd.Timestamp.now(tz=NY)


def daily_price_date():
    now = now_et()
    # Recheck the close instead of retaining a pre-close response through the evening.
    return (now.date().isoformat(), (now.hour, now.minute) >= (16, 20))


# Market and factor data

@st.cache_data(ttl=300, max_entries=48, show_spinner=False)
def download_prices(symbols, period, interval, session_key=None):
    prices = yf.download(
        list(symbols),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        prepost=False,
        progress=False,
        multi_level_index=True,
        threads=4 if len(symbols) > 1 else False,
        timeout=10,
    )
    output = {}
    if prices is None or prices.empty:
        return output

    for symbol in symbols:
        if isinstance(prices.columns, pd.MultiIndex):
            if symbol not in prices.columns.get_level_values(0):
                continue
            frame = prices[symbol]
        elif len(symbols) == 1:
            frame = prices
        else:
            continue
        if "Close" in frame:
            frame = frame.dropna(subset=["Close"]).sort_index()
            if not frame.empty:
                output[symbol] = frame.loc[~frame.index.duplicated(keep="last")]
    return output


def get_prices(symbols, period, interval):
    unique = tuple(dict.fromkeys(symbols))
    try:
        key = daily_price_date() if interval == "1d" else None
        return download_prices(unique, period, interval, key)
    except Exception:
        return {}


def daily_series(frame, adjusted=True):
    field = "Adj Close" if adjusted else "Close"
    if field not in frame:
        return pd.Series(dtype=float)
    series = frame[field].dropna().copy()
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    series.index = index.normalize()
    today, after_close = daily_price_date()
    cutoff = pd.Timestamp(today)
    series = series.loc[series.index <= cutoff] if after_close else series.loc[series.index < cutoff]
    return series.loc[(series > 0) & ~series.index.duplicated(keep="last")]


def intraday_frame(frame, completed=False):
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    converted = frame.set_axis(index.tz_convert(NY))
    if completed:
        converted = converted.loc[converted.index + BAR <= now_et()]
    return converted


@st.cache_data(ttl=21600, max_entries=2, show_spinner=False)
def load_factors(dataset):
    response = requests.get(FACTOR_BASE + FACTOR_FILES[dataset], timeout=12)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        filename = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        lines = archive.read(filename).decode("latin-1").splitlines()

    header = next(line for line in lines if "Mkt-RF" in line and "RF" in line)
    fields = [field.strip() for field in header.split(",") if field.strip()]
    rows = [line for line in lines if re.match(r"^\s*\d{8}\s*,", line)]
    if not rows:
        raise ValueError("The factor file has no daily observations.")
    frame = pd.read_csv(StringIO("\n".join(rows)), header=None, names=["Date", *fields])
    frame.index = pd.to_datetime(frame.pop("Date").astype(str), format="%Y%m%d")
    return frame.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999], np.nan).sort_index() / 100


# Two independent news feeds

def news_item(title, link, stamp, publisher, feed):
    if not isinstance(title, str) or not isinstance(link, str):
        return None
    title = title.strip()
    url = urlparse(link)
    if not title or url.scheme not in {"http", "https"} or not url.netloc:
        return None
    return {
        "Headline": title, "Article": link, "Observed": stamp,
        "Publisher": publisher or url.netloc, "Feed": feed,
    }


@st.cache_data(ttl=1800, max_entries=128, show_spinner=False)
def yahoo_headlines(symbol):
    try:
        articles = yf.Ticker(symbol).get_news(count=12) or []
        rows = []
        for item in articles:
            article = item.get("content") or item
            link = (
                (article.get("canonicalUrl") or {}).get("url")
                or (article.get("clickThroughUrl") or {}).get("url")
                or article.get("link")
            )
            stamp = article.get("pubDate") or article.get("providerPublishTime")
            if isinstance(stamp, (int, float)):
                stamp = pd.to_datetime(stamp, unit="s", utc=True, errors="coerce")
            else:
                stamp = pd.to_datetime(stamp, utc=True, errors="coerce")
            row = news_item(
                article.get("title"), link, stamp,
                (article.get("provider") or {}).get("displayName") or article.get("publisher"),
                "Yahoo Finance",
            )
            if row:
                rows.append(row)
        return rows, True
    except Exception:
        return [], False


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def gdelt_headlines(query):
    try:
        response = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": query, "mode": "ArtList", "format": "json",
                "maxrecords": 15, "timespan": "7d", "sort": "DateDesc",
            },
            timeout=6,
        )
        response.raise_for_status()
        rows = []
        for article in response.json().get("articles", []):
            stamp = pd.to_datetime(
                article.get("seendate"), format="%Y%m%dT%H%M%SZ",
                utc=True, errors="coerce",
            )
            row = news_item(
                article.get("title"), article.get("url"), stamp,
                article.get("domain"), "GDELT",
            )
            if row:
                rows.append(row)
        return rows, True
    except Exception:
        return [], False


def company_match(title, symbol, name):
    # Use exact names and tickers. Single-letter tickers need a company-name match.
    if name != symbol and name.casefold() in title.casefold():
        return True
    return len(symbol) >= 2 and bool(re.search(rf"\b{re.escape(symbol)}\b", title))


def company_news(symbol):
    name = CATALOG.get(symbol, (symbol, ""))[0]
    safe_name = name.replace('"', "")
    yahoo, yahoo_ok = yahoo_headlines(symbol)
    gdelt, gdelt_ok = gdelt_headlines(f'"{safe_name}" sourcelang:english')
    rows = [
        row for row in yahoo + gdelt
        if company_match(row["Headline"], symbol, name)
    ]
    return rows, yahoo_ok, gdelt_ok


def news_table(rows):
    if not rows:
        st.caption("No matching headlines were returned.")
        return
    frame = pd.DataFrame(rows)
    frame["Observed"] = pd.to_datetime(frame["Observed"], utc=True).dt.tz_convert(ZURICH)
    frame = frame.sort_values("Observed", ascending=False)
    frame["_headline"] = frame["Headline"].str.casefold().str.replace(r"\W+", " ", regex=True)
    frame = frame.drop_duplicates("Article").drop_duplicates("_headline").drop(columns="_headline")
    st.dataframe(
        frame, hide_index=True,
        column_config={"Article": st.column_config.LinkColumn("Link", display_text="Open")},
    )


# Forecast engine

def model_suite():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=8.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=80, max_depth=7, min_samples_leaf=20,
            n_jobs=1, random_state=42,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=15, learning_rate=0.05,
            min_samples_leaf=30, early_stopping=False, random_state=42,
        ),
    }


def make_panel(asset, spy, qqq, horizon, intraday):
    prices = pd.concat(
        [asset.rename("Asset"), spy.rename("SPY"), qqq.rename("QQQ")],
        axis=1, join="inner",
    ).dropna().sort_index()
    prices = prices.loc[prices.gt(0).all(axis=1)]
    logs = np.log(prices)
    features = pd.DataFrame(index=prices.index)

    for market in prices.columns:
        for lag in (1, 3, 5, 10, 20):
            features[f"{market}_{lag}"] = logs[market].diff(lag)
    returns = logs["Asset"].diff()
    features["Volatility_10"] = returns.rolling(10).std()
    features["Volatility_20"] = returns.rolling(20).std()
    features["Relative_SPY_10"] = features["Asset_10"] - features["SPY_10"]
    features["Relative_QQQ_10"] = features["Asset_10"] - features["QQQ_10"]

    timestamps = prices.index.to_series()
    target_end = timestamps.shift(-horizon)
    target = logs["Asset"].shift(-horizon) - logs["Asset"]

    if intraday:
        features = features.where(timestamps.diff(20).eq(20 * BAR), axis=0)
        target = target.where((target_end - timestamps).eq(horizon * BAR))

    labeled = features.assign(Target=target, TargetEnd=target_end).dropna()
    return prices, features, labeled


def balanced_accuracy(actual, forecast):
    up = actual > 0
    down = actual < 0
    if min(up.sum(), down.sum()) < 5:
        return np.nan
    return float(((forecast[up] > 0).mean() + (forecast[down] < 0).mean()) / 2)


def nonoverlapping_holdout(frame):
    positions = []
    previous_end = None
    for position, (anchor, target_end) in enumerate(frame["TargetEnd"].items()):
        if previous_end is None or anchor >= previous_end:
            positions.append(position)
            previous_end = target_end
    return frame.iloc[positions]


@st.cache_data(ttl=43200, max_entries=20, show_spinner=False)
def forecast_universe(series, symbols, horizon, intraday):
    if any(symbol not in series for symbol in BENCHMARKS):
        return {}

    spy, qqq = (series[symbol] for symbol in BENCHMARKS)
    panels = {}
    latest = {}
    sample_minimum = max(180, horizon * 25)
    for symbol in symbols:
        if symbol not in series:
            continue
        prices, features, labeled = make_panel(series[symbol], spy, qqq, horizon, intraday)
        if (
            len(labeled) < sample_minimum
            or features.empty
            or features.iloc[-1].isna().any()
        ):
            continue
        panels[symbol] = labeled
        latest[symbol] = (prices, features.iloc[[-1]])

    if not panels:
        return {}

    # Use one date boundary for every ticker so the pooled model cannot
    # learn from another ticker's future within the same holdout window.
    all_dates = pd.concat(panels.values()).index.unique().sort_values()
    test_start = all_dates[int(len(all_dates) * 0.55)]
    training_parts, testing_parts, refit_parts, valid_symbols = [], [], [], []
    for symbol, frame in panels.items():
        train = frame.loc[frame["TargetEnd"] < test_start]
        test = nonoverlapping_holdout(frame.loc[frame.index >= test_start])
        if len(train) < 80 or len(test) < 10:
            continue
        training_parts.append(train)
        testing_parts.append(test.assign(Symbol=symbol))
        refit_parts.append(frame)
        valid_symbols.append(symbol)

    if not valid_symbols:
        return {}

    train = pd.concat(training_parts)
    holdout = pd.concat(testing_parts)
    refit = pd.concat(refit_parts)
    features = [column for column in train if column not in {"Target", "TargetEnd"}]
    current = pd.concat(
        [latest[symbol][1].assign(Symbol=symbol) for symbol in valid_symbols]
    ).set_index("Symbol")

    test_forecasts, current_forecasts = {}, {}
    for name, model in model_suite().items():
        model.fit(train[features], train["Target"])
        test_forecasts[name] = model.predict(holdout[features])
        model.fit(refit[features], refit["Target"])
        current_forecasts[name] = model.predict(current[features])

    for name, values in test_forecasts.items():
        holdout[name] = values
    for name, values in current_forecasts.items():
        current[name] = values

    names = list(test_forecasts)
    result = {}
    for symbol, test in holdout.groupby("Symbol", sort=False):
        actual = test["Target"].to_numpy()
        prediction = test[names].mean(axis=1).to_numpy()
        model_now = current.loc[symbol, names].astype(float)
        projected = float(model_now.mean())
        price, _ = latest[symbol]
        anchor = float(price["Asset"].iloc[-1])
        low_error, high_error = np.quantile(actual - prediction, [0.1, 0.9])
        actual_simple = np.expm1(actual)
        ma = float(price["Asset"].tail(50).mean())

        recent = price.iloc[[-11, -1]]
        relative_10 = float(
            (recent["Asset"].iloc[-1] / recent["Asset"].iloc[0])
            / (recent["SPY"].iloc[-1] / recent["SPY"].iloc[0]) - 1
        )
        result[symbol] = {
            "anchor_date": price.index[-1],
            "anchor": anchor,
            "return": float(np.expm1(projected)),
            "low": float(anchor * np.exp(projected + low_error)),
            "high": float(anchor * np.exp(projected + high_error)),
            "mae": float(np.abs(actual_simple - np.expm1(prediction)).mean()),
            "flat_mae": float(np.abs(actual_simple).mean()),
            "direction": balanced_accuracy(actual, prediction),
            "count": len(test),
            "start": test.index[0],
            "end": test.index[-1],
            "agreement": int((np.sign(model_now) == np.sign(projected)).sum()),
            "relative_10": relative_10,
            "ma50_gap": anchor / ma - 1,
            "methods": [
                {
                    "Model": name,
                    "Projected move %": 100 * np.expm1(model_now[name]),
                    "Holdout MAE (pp)": 100 * np.abs(
                        actual_simple - np.expm1(test[name].to_numpy())
                    ).mean(),
                    "Balanced direction %": 100 * balanced_accuracy(
                        actual, test[name].to_numpy()
                    ),
                }
                for name in names
            ],
        }
    return result


def signal_checks(result, latest_date, minimum_move, intraday=False):
    stamp = result["anchor_date"]
    fresh = stamp == latest_date
    if intraday:
        close_time = stamp + BAR
        target = close_time + 6 * BAR
        now = now_et()
        fresh = (
            fresh
            and pd.Timedelta(0) <= now - close_time <= pd.Timedelta(minutes=15)
            and target.date() == now.date()
            and target <= target.normalize() + pd.Timedelta(hours=16)
        )
    elif (pd.Timestamp(now_et().date()) - stamp).days > 5:
        fresh = False

    accuracy = result["direction"]
    return {
        "Recent, benchmark-aligned price": fresh,
        "At least 25 independent holdout windows": result["count"] >= 25,
        "At least 55% balanced direction accuracy": np.isfinite(accuracy) and accuracy >= 0.55,
        "At least 5% less error than flat price": (
            result["flat_mae"] > 0 and result["mae"] <= result["flat_mae"] * 0.95
        ),
        "All three models agree": result["agreement"] == 3,
        "Forecast clears your move threshold": abs(result["return"]) >= minimum_move,
    }


def choose_signal(result, checks):
    if not all(checks.values()):
        return "Wait"
    if result["return"] > 0:
        return "Buy Candidate"
    if result["return"] < 0:
        return "Sell / Avoid Bias"
    return "Wait"


def valid_price_series(data, adjusted=True):
    return {symbol: daily_series(frame, adjusted) for symbol, frame in data.items()}


def benchmark_anchor(series):
    if any(symbol not in series or series[symbol].empty for symbol in BENCHMARKS):
        return None
    common = series["SPY"].index.intersection(series["QQQ"].index)
    return common.max() if len(common) else None


# Daily shortlist

def candle_pattern(result):
    relative = result["relative_10"]
    above_average = result["ma50_gap"] >= 0
    if relative > 0 and above_average:
        return "Trend continuation"
    if relative < 0 and not above_average:
        return "Rebound setup"
    return "Mixed price trend"


def show_daily_picks(symbols, minimum_move):
    st.subheader("Daily Picks")
    st.caption("Five-session forecasts based on the most recent completed US session. Each ticker is judged on its own holdout results.")

    with st.spinner("Loading daily closes and evaluating three models…"):
        data = get_prices([*symbols, *BENCHMARKS], "5y", "1d")
        series = valid_price_series(data)
        latest_date = benchmark_anchor(series)
        results = forecast_universe(series, tuple(symbols), 5, False) if latest_date is not None else {}

    if latest_date is None:
        st.warning("SPY or QQQ history is unavailable. The scan needs both benchmarks.")
        return

    st.caption(f"Latest shared benchmark close {latest_date:%Y-%m-%d} · {len(results)} of {len(symbols)} selected instruments evaluated.")
    if not results:
        st.info("No instruments have enough usable data for the scan.")
        return

    scans = []
    for symbol, result in results.items():
        checks = signal_checks(result, latest_date, minimum_move)
        signal = choose_signal(result, checks)
        # Rank by the 10th-percentile error-adjusted return, then the point forecast.
        conservative_return = result["low"] / result["anchor"] - 1
        scans.append((symbol, result, signal, checks, conservative_return))
    scans.sort(key=lambda item: (item[4], item[1]["return"]), reverse=True)

    candidates = [item for item in scans if item[2] == "Buy Candidate"]
    news_checked = {}
    for kind in ("Stock", "ETF"):
        eligible = [
            item for item in candidates
            if ("ETF" if CATALOG[item[0]][1].endswith("ETFs") else "Stock") == kind
        ]
        # Check the top three in each category. Other candidates remain pending.
        for symbol, *_ in eligible[:3]:
            rows, yahoo_ok, gdelt_ok = company_news(symbol)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
            flagged = [
                row for row in rows
                if pd.notna(row["Observed"]) and row["Observed"] >= cutoff
                and CATALYST.search(row["Headline"])
            ]
            news_checked[symbol] = (flagged, yahoo_ok, gdelt_ok, len(rows))

    full_rows, details = [], {}
    for symbol, result, signal, checks, conservative_return in scans:
        if signal == "Buy Candidate":
            if symbol not in news_checked:
                signal = "News Check Pending"
            else:
                flagged, yahoo_ok, gdelt_ok, coverage = news_checked[symbol]
                if flagged:
                    signal = "Review Catalyst"
                elif not (yahoo_ok or gdelt_ok):
                    signal = "News Unavailable"
                elif coverage == 0:
                    signal = "Limited News Coverage"
        improvement = (
            1 - result["mae"] / result["flat_mae"] if result["flat_mae"] > 0 else np.nan
        )
        full_rows.append({
            "Ticker": symbol,
            "Name": CATALOG[symbol][0],
            "Type": "ETF" if CATALOG[symbol][1].endswith("ETFs") else "Stock",
            "Group": CATALOG[symbol][1],
            "Signal": signal,
            "Setup": candle_pattern(result),
            "Close": result["anchor"],
            "5-session forecast %": 100 * result["return"],
            "Error-adjusted lower move %": 100 * conservative_return,
            "vs SPY over 10 sessions %": 100 * result["relative_10"],
            "Distance from 50-day average %": 100 * result["ma50_gap"],
            "Balanced direction %": 100 * result["direction"],
            "MAE improvement %": 100 * improvement,
            "Holdout windows": result["count"],
            "Why": (
                "Passed all price-model checks"
                if all(checks.values())
                else "Unmet: " + "; ".join(name for name, passed in checks.items() if not passed)
            ),
        })
        details[symbol] = (result, signal, improvement)

    table = pd.DataFrame(full_rows)
    for kind, heading in (("Stock", "Stocks to research for a buy"), ("ETF", "ETFs to research for a buy")):
        st.markdown(f"### {heading}")
        shortlist = table.loc[table["Type"].eq(kind) & table["Signal"].eq("Buy Candidate")]
        if shortlist.empty:
            st.info(f"No {kind.lower()} passed the model and headline checks today.")
        for row in shortlist.to_dict("records"):
            symbol = row["Ticker"]
            result, _, improvement = details[symbol]
            with st.container(border=True):
                st.markdown(f"**{ticker_label(symbol)} · {row['Setup']}**")
                a, b, c = st.columns(3)
                a.metric("Five-session projection", f"{result['return']:+.2%}")
                b.metric("Historical error range", f"${result['low']:,.2f}–${result['high']:,.2f}")
                c.metric("10-session move vs SPY", f"{result['relative_10']:+.2%}")
                st.write(
                    f"All three models predict upside. For this ticker, the ensemble "
                    f"reduced historical forecast error by {improvement:.0%} versus a flat price "
                    f"and scored {result['direction']:.0%} on balanced direction accuracy. "
                    f"The close is {result['ma50_gap']:+.1%} from its 50-session average."
                )
                if result["low"] < result["anchor"]:
                    st.caption("The lower end of the historical error range still allows a price decline.")

    flagged_rows = [row for flagged, *_ in news_checked.values() for row in flagged]
    if flagged_rows:
        with st.expander("Headlines that moved candidates to Review Catalyst", expanded=True):
            news_table(flagged_rows)

    with st.expander("Full scan and reasons", expanded=True):
        st.dataframe(table.round(2), hide_index=True)
        st.download_button(
            "Download daily scan", table.to_csv(index=False),
            file_name=f"daily_picks_{latest_date:%Y-%m-%d}.csv", mime="text/csv",
        )

    with st.expander("How are picks classified?"):
        st.markdown("""
- **Buy Candidate** means the forecast points up, all three models agree, and the ticker passes the historical error and direction checks. The headline check found coverage and no matching catalyst keyword in the recent results.
- **Trend continuation** means the asset is above its 50-session average and beat SPY over the last ten sessions. **Rebound setup** means it is below its average and lagged SPY. **Mixed price trend** covers the other combinations.
- **Review Catalyst** means a recent matching headline mentions an event that could change the price quickly. The rule reads headlines and can miss relevant events.
- **Sell / Avoid Bias** means the downside forecast passed the same model checks. **Wait** means one or more checks failed.
- The shortlist ranks eligible assets by the lower end of their historical error range, expressed as a return. A higher rank can still include downside. The setup describes price behavior and relative performance. Intrinsic value depends on separate assumptions about cash flows or fund holdings.
        """)


# Intraday market overview

def market_table(labels, live, daily):
    rows = []
    for symbol, name in labels.items():
        price, move, stamp = np.nan, np.nan, "—"
        if symbol in live:
            frame = intraday_frame(live[symbol])
            last = frame.index[-1]
            stamp = last.strftime("%b %d, %H:%M")
            price = float(frame["Close"].iloc[-1])
            if symbol in daily:
                previous = daily_series(daily[symbol], adjusted=False)
                previous = previous.loc[previous.index < pd.Timestamp(last.date())]
                if not previous.empty:
                    move = 100 * (price / previous.iloc[-1] - 1)
        rows.append({"Instrument": name, "Ticker": symbol, "Price": price, "Move %": move, "Bar start (ET)": stamp})
    return pd.DataFrame(rows)


def show_live(ticker, fair_value):
    symbols = [ticker, *INDEXES, *SECTORS]
    live = get_prices(symbols, "5d", "5m")
    daily = get_prices(symbols, "1mo", "1d")
    market = market_table(INDEXES, live, daily)

    st.subheader("US Market")
    for card, row in zip(st.columns(len(INDEXES)), market.to_dict("records")):
        price, move = row["Price"], row["Move %"]
        card.metric(
            row["Instrument"], "—" if pd.isna(price) else f"{price:,.2f}",
            None if pd.isna(move) else f"{move:+.2f}%",
            delta_color="off" if row["Ticker"] == "^VIX" else "normal",
        )

    with st.expander("Market prices and timestamps"):
        st.dataframe(market.round(2), hide_index=True)

    st.subheader(f"{ticker} · Five-minute price")
    if ticker not in live:
        st.info("Yahoo Finance did not return intraday bars for this ticker.")
    else:
        frame = intraday_frame(live[ticker])
        last = frame.index[-1]
        frame = frame.loc[frame.index.date == last.date()]
        close = float(frame["Close"].iloc[-1])
        a, b, c = st.columns(3)
        a.metric("Latest bar price", f"${close:,.2f}")
        b.metric("Latest bar start (ET)", last.strftime("%b %d · %H:%M"))
        if fair_value > 0:
            c.metric("Gap to your fair value", f"{close / fair_value - 1:+.2%}")
        if {"Open", "High", "Low"}.issubset(frame.columns):
            fig = go.Figure(go.Candlestick(
                x=frame.index, open=frame["Open"], high=frame["High"],
                low=frame["Low"], close=frame["Close"],
                increasing_line_color="#22c5a0", decreasing_line_color="#f07584",
            ))
            fig.update_layout(
                template="plotly_dark", height=440, xaxis_rangeslider_visible=False,
                yaxis_title="USD", margin=dict(l=10, r=20, t=15, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        if "Volume" in frame and frame["Volume"].fillna(0).sum() > 0:
            st.bar_chart(frame["Volume"].rename("Shares per bar"))

    st.subheader("Sector performance")
    sectors = market_table(SECTORS, live, daily)
    st.dataframe(sectors.sort_values("Move %", ascending=False).round(2), hide_index=True)
    st.caption("Move % compares the latest bar with the previous daily close. A bar can still be forming. Check its timestamp.")


# CAPM and Fama–French

def fit_factors(sample, terms):
    return sm.OLS(
        sample["Return"] - sample["RF"],
        sm.add_constant(sample[terms], has_constant="add"),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": True}, use_t=True)


def factor_gap(sample, terms, days=10):
    training, recent = sample.iloc[:-days], sample.iloc[-days:]
    blocks = len(training) // days
    if blocks < 10:
        return None
    fit = fit_factors(training, terms)
    baseline = sample["RF"] + sample[terms] @ fit.params[terms]
    if baseline.le(-1).any() or sample["Return"].le(-1).any():
        return None
    residual = np.log1p(sample["Return"]) - np.log1p(baseline)
    history = residual.iloc[:-days].to_numpy()[-blocks * days:]
    scale = history.reshape(blocks, days).sum(axis=1).std(ddof=1)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return (
        float(residual.iloc[-days:].sum() / scale),
        float(np.prod(1 + recent["Return"]) - 1),
        float(np.prod(1 + baseline.iloc[-days:]) - 1),
    )


def show_factors(ticker, selected_model):
    st.subheader(f"{ticker} · Factor Models")
    data = get_prices([ticker], "3y", "1d")
    if ticker not in data:
        st.warning("Daily price history is unavailable.")
        return

    price = daily_series(data[ticker])
    returns = price.pct_change(fill_method=None).rename("Return")
    try:
        ff3 = load_factors("FF3")
        ff5 = load_factors("FF5")
        common = pd.concat(
            [returns, ff3.add_suffix("_3"), ff5.add_suffix("_5")],
            axis=1, join="inner",
        ).dropna()
    except Exception:
        st.warning("The Kenneth French factor files could not be loaded.")
        return

    if len(common) < 60:
        st.info(f"Found {len(common)} matched returns. At least 60 are required.")
        return

    st.caption(
        f"{len(common)} daily returns · Data through {common.index[-1]:%Y-%m-%d} · "
        f"Latest Yahoo adjusted close {price.index[-1]:%Y-%m-%d}. "
        "The official factor data can arrive later than Yahoo prices."
    )

    summary, coefficients = [], []
    selected_sample = None
    for name, terms in MODELS.items():
        suffix = "_5" if name == "FF5" else "_3"
        fields = {f"{term}{suffix}": term for term in [*terms, "RF"]}
        sample = common[["Return", *fields]].rename(columns=fields)
        model = fit_factors(sample, terms)

        summary.append({
            "Model": name,
            "Market beta": model.params["Mkt-RF"],
            "Annualized alpha %": 25200 * model.params["const"],
            "Alpha t-stat": model.tvalues["const"],
            "Alpha p-value": model.pvalues["const"],
            "Adjusted R²": model.rsquared_adj,
        })
        if name == selected_model:
            selected_sample = sample
            intervals = model.conf_int()
            for term in model.params.index:
                scale = 10000 if term == "const" else 1
                coefficients.append({
                    "Factor": "Alpha (bp/day)" if term == "const" else term,
                    "Estimate": scale * model.params[term],
                    "HAC t-stat": model.tvalues[term],
                    "p-value": model.pvalues[term],
                    "95% lower": scale * intervals.loc[term, 0],
                    "95% upper": scale * intervals.loc[term, 1],
                })

    st.dataframe(pd.DataFrame(summary).round(4), hide_index=True)
    st.markdown(f"**{selected_model} coefficients**")
    st.dataframe(pd.DataFrame(coefficients).round(4), hide_index=True)

    terms = MODELS[selected_model]
    if len(selected_sample) >= 126:
        rolling = RollingOLS(
            selected_sample["Return"] - selected_sample["RF"],
            sm.add_constant(selected_sample[terms], has_constant="add"),
            window=126,
        ).fit(params_only=True)
        st.markdown("**Rolling market beta · 126 sessions**")
        st.line_chart(rolling.params["Mkt-RF"].dropna().rename("Market beta"))

    gap = factor_gap(selected_sample, terms)
    if gap is not None:
        z, actual, implied = gap
        a, b, c = st.columns(3)
        a.metric("10-session factor score", f"{z:+.2f}σ")
        b.metric("Actual return", f"{actual:+.2%}")
        c.metric("Factor-implied return", f"{implied:+.2%}")

    with st.expander("What do these numbers mean?", expanded=True):
        st.markdown("""
| Number | In plain English |
|---|---|
| **Beta** | How strongly the asset moved with the US stock market. A beta of 1.3 means a one-point market move was associated with about a 1.3-point move in the asset, with the other factors held fixed. |
| **Alpha** | The average return the model's factors did not explain. Positive alpha is evidence of historical outperformance only to the extent that the estimate is precise. |
| **t-stat** | How large an estimate is compared with its uncertainty. An absolute t-stat around 2 is a common rule of thumb for a signal worth looking at more closely. |
| **p-value** | How surprising the estimate would be if the true exposure were zero. Smaller means stronger evidence against zero under this model. |
| **Adjusted R²** | How much of the asset's day-to-day movement the model explains, after accounting for the number of factors. A reading of 0.60 is roughly 60%. |
| **Factor score** | Whether recent returns were unusually strong or weak after accounting for the factors. +2 means about two historical standard deviations above the model's return. It measures relative performance. |

**CAPM** uses the broad market. **FF3** adds company size and value. **FF5** also adds profitability and investment. For bond, gold, and crypto ETFs, these figures describe exposure to equity factors.
        """)
        st.markdown("[Official daily factor data](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)")


# Individual forecasts and news

def show_forecast(ticker, horizon_name, minimum_move):
    st.subheader(f"{ticker} · {horizon_name} Forecast")
    settings = {"30 minutes": (6, True), "1 session": (1, False),
                "5 sessions": (5, False), "20 sessions": (20, False)}
    horizon, intraday = settings[horizon_name]
    symbols = [ticker, *BENCHMARKS]
    if intraday:
        raw = get_prices(symbols, "1mo", "5m")
        series = {symbol: intraday_frame(frame, completed=True)["Close"] for symbol, frame in raw.items()}
    else:
        raw = get_prices(symbols, "5y", "1d")
        series = valid_price_series(raw)

    latest_date = benchmark_anchor(series)
    if latest_date is None:
        st.info("The benchmark feeds are unavailable.")
        return

    with st.spinner("Evaluating the models on unseen historical prices…"):
        result = forecast_universe(series, (ticker,), horizon, intraday).get(ticker)
    if result is None:
        st.info(
            "There is insufficient aligned history for this forecast. The 30-minute model "
            "also needs 20 consecutive completed five-minute bars before its anchor."
        )
        return

    checks = signal_checks(result, latest_date, minimum_move, intraday)
    signal = choose_signal(result, checks)
    if signal == "Buy Candidate":
        rows, yahoo_ok, gdelt_ok = company_news(ticker)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
        events = [
            row for row in rows
            if pd.notna(row["Observed"]) and row["Observed"] >= cutoff
            and CATALYST.search(row["Headline"])
        ]
        if events:
            signal = "Review Catalyst"
        elif not (yahoo_ok or gdelt_ok):
            signal = "News Unavailable"
        elif not rows:
            signal = "Limited News Coverage"
    else:
        events = []

    projected = result["anchor"] * (1 + result["return"])
    a, b, c, d = st.columns(4)
    a.metric("Research signal", signal)
    b.metric("Projected price", f"${projected:,.2f}", f"{result['return']:+.2%}")
    c.metric("Historical error range", f"${result['low']:,.2f}–${result['high']:,.2f}")
    d.metric("Model agreement", f"{result['agreement']}/3")
    stamp = result["anchor_date"]
    anchor_time = (stamp + BAR).strftime("%Y-%m-%d %H:%M ET") if intraday else stamp.strftime("%Y-%m-%d")
    st.caption(f"Price anchor ${result['anchor']:,.2f} · {anchor_time}")

    st.dataframe(pd.DataFrame([
        {"Check": name, "Result": "Passed" if passed else "Unmet"}
        for name, passed in checks.items()
    ]), hide_index=True)
    if events:
        news_table(events)

    st.dataframe(pd.DataFrame(result["methods"]).round(3), hide_index=True)
    st.caption(
        f"Ensemble MAE {100 * result['mae']:.2f} pp · Flat-price MAE {100 * result['flat_mae']:.2f} pp · "
        f"{result['count']} holdout forecasts from {result['start']:%Y-%m-%d} to {result['end']:%Y-%m-%d}."
    )

    with st.expander("What do the forecasts mean?", expanded=True):
        st.markdown("""
- **Projected price** is the average model forecast applied to the price at the anchor. Ridge, Random Forest, and Gradient Boosting use past price changes, benchmark moves, and volatility.
- **Historical error range** applies the middle 80% of past holdout errors to the current estimate. The next price can fall outside that range.
- **MAE** is the average size of a forecast miss. A 2 pp MAE means the model missed by two percentage points on average. A flat-price forecast predicts a zero return every time.
- **Balanced direction accuracy** gives rising and falling periods equal weight. Always forecasting up scores 50% when both directions are represented.
- The test uses later dates than the training data and skips overlapping return windows for each ticker. It measures prediction quality before trading costs.
        """)


def show_news(ticker):
    st.subheader(f"{ticker} · Company News")
    rows, yahoo_ok, gdelt_ok = company_news(ticker)
    source_status = (
        f"Yahoo Finance {'available' if yahoo_ok else 'unavailable'} · "
        f"GDELT {'available' if gdelt_ok else 'unavailable'}"
    )
    st.caption(source_status)
    news_table(rows)

    st.subheader("S&P 500 and Nasdaq News")
    yahoo_spy, _ = yahoo_headlines("SPY")
    yahoo_qqq, _ = yahoo_headlines("QQQ")
    articles, _ = gdelt_headlines('("S&P 500" OR "Nasdaq 100") sourcelang:english')
    news_table(yahoo_spy + yahoo_qqq + articles)
    st.caption("Times are shown in Zurich. Yahoo supplies publication time. GDELT supplies the time it first recorded the story.")


def apply_style():
    st.markdown(
        """
<style>
.block-container {max-width: 1440px; padding-top: 2.4rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(110, 143, 170, 0.32);
    border-radius: 12px;
    background: rgba(74, 123, 174, 0.08);
    padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_style()
    st.title("Market Research Terminal")
    st.caption("US stocks and ETFs · S&P 500 and Nasdaq")

    with st.sidebar:
        page = st.radio("Page", ["Daily Picks", "Live Market", "Factor Models", "Forecasts", "News"])
        ticker = "NVDA"
        scan_symbols = ()
        minimum_move = 0.005
        fair_value = 0.0
        factor_model = "FF5"
        horizon = "5 sessions"

        if page == "Daily Picks":
            scan_symbols = tuple(st.multiselect(
                "Daily scan universe",
                list(CATALOG),
                default=DEFAULT_SCAN,
                format_func=ticker_label,
                max_selections=45,
            ))
            minimum_move = st.number_input(
                "Minimum five-session forecast (%)", min_value=0.0, max_value=20.0,
                value=0.50, step=0.10,
            ) / 100
        else:
            group = st.selectbox("Asset group", ["All", *GROUPS])
            options = [symbol for symbol, (_, category) in CATALOG.items()
                       if group == "All" or category == group]
            selected = st.selectbox("Search ticker or name", options, format_func=ticker_label)
            ticker = st.text_input("Custom Yahoo ticker", max_chars=16).strip().upper() or selected
            if not re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)?", ticker):
                st.warning("Enter a US stock or ETF ticker such as AAPL, BRK-B, or QQQ.")
                st.stop()

            if page == "Live Market":
                fair_value = st.number_input(
                    "Your fair value estimate ($)", min_value=0.0, step=1.0,
                    help="Set to zero to hide this comparison.",
                )
            elif page == "Factor Models":
                factor_model = st.selectbox("Factor model", list(MODELS), index=2)
            elif page == "Forecasts":
                horizon = st.selectbox("Forecast horizon", ["30 minutes", "1 session", "5 sessions", "20 sessions"], index=2)
                minimum_move = st.number_input(
                    "Minimum forecast move (%)", min_value=0.0, max_value=20.0,
                    value=0.15 if horizon == "30 minutes" else 0.50,
                    step=0.05, key=f"threshold_{horizon}",
                ) / 100

        refresh_interval = None
        if page in {"Live Market", "Daily Picks"} or (page == "Forecasts" and horizon == "30 minutes"):
            auto = st.toggle("Auto-refresh while this page is open", value=True)
            refresh_interval = ("15m" if page == "Daily Picks" else "5m") if auto else None

    @st.fragment(run_every=refresh_interval)
    def dashboard():
        if st.button("Refresh now"):
            download_prices.clear()
            yahoo_headlines.clear()
            gdelt_headlines.clear()
            load_factors.clear()
            forecast_universe.clear()

        if page == "Daily Picks":
            if scan_symbols:
                show_daily_picks(scan_symbols, minimum_move)
            else:
                st.info("Select at least one stock or ETF to scan.")
        elif page == "Live Market":
            show_live(ticker, fair_value)
        elif page == "Factor Models":
            show_factors(ticker, factor_model)
        elif page == "Forecasts":
            show_forecast(ticker, horizon, minimum_move)
        else:
            show_news(ticker)

    dashboard()


if __name__ == "__main__":
    main()
