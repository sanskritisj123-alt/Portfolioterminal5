import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Portfolio Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

DB_FILE = "portfolio.db"

REQUIRED_COLUMNS = [
    "Ticker",
    "Quantity",
    "Purchase Price",
    "Purchase Date"
]

BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC"
}


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    conn = sqlite3.connect(DB_FILE)

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    return conn


def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_name TEXT NOT NULL UNIQUE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            quantity REAL NOT NULL,
            purchase_price REAL NOT NULL,
            purchase_date TEXT NOT NULL,
            FOREIGN KEY (portfolio_id)
            REFERENCES portfolios(id)
            ON DELETE CASCADE
        )
        """
    )

    conn.commit()

    conn.close()


def save_portfolio_to_db(name, df):

    conn = get_connection()

    cursor = conn.cursor()

    try:

        name = str(name).strip()

        if not name:

            raise ValueError(
                "Portfolio name cannot be empty."
            )

        # ----------------------------------------------------
        # Check if portfolio already exists
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM portfolios
            WHERE portfolio_name = ?
            """,
            (name,)
        )

        result = cursor.fetchone()

        if result:

            portfolio_id = result[0]

            # Delete old holdings
            cursor.execute(
                """
                DELETE FROM holdings
                WHERE portfolio_id = ?
                """,
                (portfolio_id,)
            )

        else:

            cursor.execute(
                """
                INSERT INTO portfolios
                (portfolio_name)
                VALUES (?)
                """,
                (name,)
            )

            portfolio_id = cursor.lastrowid

        # ----------------------------------------------------
        # Insert holdings
        # ----------------------------------------------------

        for _, row in df.iterrows():

            ticker = str(
                row["Ticker"]
            ).upper().strip()

            quantity = float(
                row["Quantity"]
            )

            purchase_price = float(
                row["Purchase Price"]
            )

            purchase_date = pd.to_datetime(
                row["Purchase Date"]
            ).strftime(
                "%Y-%m-%d"
            )

            cursor.execute(
                """
                INSERT INTO holdings
                (
                    portfolio_id,
                    ticker,
                    quantity,
                    purchase_price,
                    purchase_date
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    portfolio_id,
                    ticker,
                    quantity,
                    purchase_price,
                    purchase_date
                )
            )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


def get_saved_portfolios():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT portfolio_name
        FROM portfolios
        ORDER BY portfolio_name
        """
    )

    portfolios = [
        row[0]
        for row in cursor.fetchall()
    ]

    conn.close()

    return portfolios


def load_portfolio_from_db(name):

    conn = get_connection()

    query = """
        SELECT
            h.ticker AS Ticker,
            h.quantity AS Quantity,
            h.purchase_price AS "Purchase Price",
            h.purchase_date AS "Purchase Date"
        FROM holdings h
        JOIN portfolios p
        ON h.portfolio_id = p.id
        WHERE p.portfolio_name = ?
        ORDER BY h.id
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(name,)
    )

    conn.close()

    return df


def delete_portfolio_from_db(name):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM portfolios
        WHERE portfolio_name = ?
        """,
        (name,)
    )

    conn.commit()

    conn.close()


# ============================================================
# NORMALIZE PORTFOLIO
# ============================================================

def normalize_portfolio(df):

    df = df.copy()

    # Clean column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # --------------------------------------------------------
    # Recognize alternative column names
    # --------------------------------------------------------

    column_mapping = {}

    for col in df.columns:

        lower_col = (
            str(col)
            .lower()
            .strip()
        )

        if lower_col in [
            "ticker",
            "symbol",
            "stock",
            "stock ticker"
        ]:

            column_mapping[col] = "Ticker"

        elif lower_col in [
            "quantity",
            "qty",
            "shares",
            "units"
        ]:

            column_mapping[col] = "Quantity"

        elif lower_col in [
            "purchase price",
            "buy price",
            "price",
            "purchase_price"
        ]:

            column_mapping[col] = "Purchase Price"

        elif lower_col in [
            "purchase date",
            "buy date",
            "date",
            "purchase_date"
        ]:

            column_mapping[col] = "Purchase Date"

    df = df.rename(
        columns=column_mapping
    )

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    df = df[
        REQUIRED_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Clean ticker
    # --------------------------------------------------------

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Clean quantity
    # --------------------------------------------------------

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Clean purchase price
    # --------------------------------------------------------

    df["Purchase Price"] = pd.to_numeric(
        df["Purchase Price"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Clean date
    # --------------------------------------------------------

    df["Purchase Date"] = pd.to_datetime(
        df["Purchase Date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=REQUIRED_COLUMNS
    )

    if df.empty:

        raise ValueError(
            "No valid portfolio data found."
        )

    # --------------------------------------------------------
    # Convert date to string for storage
    # --------------------------------------------------------

    df["Purchase Date"] = (
        df["Purchase Date"]
        .dt.strftime("%Y-%m-%d")
    )

    return df


# ============================================================
# MARKET DATA
# ============================================================

@st.cache_data(ttl=300)
def get_market_data(tickers):

    data = {}

    for ticker in tickers:

        try:

            stock = yf.Ticker(ticker)

            history = stock.history(
                period="1y",
                auto_adjust=True
            )

            if not history.empty:

                data[ticker] = history

        except Exception:

            continue

    return data


# ============================================================
# HOLDINGS CALCULATION
# ============================================================

def calculate_holdings(
    portfolio_df
):

    if portfolio_df.empty:

        return pd.DataFrame()

    df = portfolio_df.copy()

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["Purchase Price"] = pd.to_numeric(
        df["Purchase Price"],
        errors="coerce"
    )

    df["Invested Value"] = (
        df["Quantity"]
        * df["Purchase Price"]
    )

    tickers = (
        df["Ticker"]
        .dropna()
        .unique()
        .tolist()
    )

    market_data = get_market_data(
        tuple(tickers)
    )

    results = []

    for ticker in tickers:

        rows = df[
            df["Ticker"] == ticker
        ]

        quantity = rows[
            "Quantity"
        ].sum()

        invested_value = rows[
            "Invested Value"
        ].sum()

        if quantity != 0:

            avg_purchase_price = (
                invested_value
                / quantity
            )

        else:

            avg_purchase_price = 0

        current_price = np.nan

        if ticker in market_data:

            history = market_data[ticker]

            if not history.empty:

                current_price = float(
                    history["Close"].iloc[-1]
                )

        if pd.isna(current_price):

            current_value = np.nan
            pnl = np.nan
            return_pct = np.nan

        else:

            current_value = (
                quantity
                * current_price
            )

            pnl = (
                current_value
                - invested_value
            )

            if invested_value != 0:

                return_pct = (
                    pnl
                    / invested_value
                    * 100
                )

            else:

                return_pct = 0

        results.append(
            {
                "Ticker": ticker,
                "Quantity": quantity,
                "Avg Purchase Price":
                    avg_purchase_price,
                "Current Price":
                    current_price,
                "Invested Value":
                    invested_value,
                "Current Value":
                    current_value,
                "P&L":
                    pnl,
                "Return %":
                    return_pct
            }
        )

    holdings = pd.DataFrame(
        results
    )

    total_current_value = (
        holdings[
            "Current Value"
        ].sum()
    )

    if total_current_value != 0:

        holdings[
            "Portfolio Weight %"
        ] = (
            holdings["Current Value"]
            / total_current_value
            * 100
        )

    else:

        holdings[
            "Portfolio Weight %"
        ] = 0

    return holdings


# ============================================================
# PORTFOLIO RETURNS
# ============================================================

def build_portfolio_returns(
    holdings
):

    if holdings.empty:

        return pd.Series(
            dtype=float
        )

    tickers = holdings[
        "Ticker"
    ].tolist()

    weights = (
        holdings[
            "Portfolio Weight %"
        ].values
        / 100
    )

    prices = yf.download(
        tickers,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    if prices.empty:

        return pd.Series(
            dtype=float
        )

    close_prices = prices["Close"]

    # --------------------------------------------------------
    # Single stock
    # --------------------------------------------------------

    if len(tickers) == 1:

        if isinstance(
            close_prices,
            pd.DataFrame
        ):

            close_prices = (
                close_prices.iloc[:, 0]
            )

        return (
            close_prices
            .pct_change()
            .dropna()
        )

    # --------------------------------------------------------
    # Multiple stocks
    # --------------------------------------------------------

    returns = (
        close_prices
        .pct_change()
        .dropna()
    )

    available_tickers = [
        ticker
        for ticker in tickers
        if ticker in returns.columns
    ]

    if not available_tickers:

        return pd.Series(
            dtype=float
        )

    weight_map = {
        ticker: weight
        for ticker, weight
        in zip(tickers, weights)
    }

    portfolio_returns = pd.Series(
        0.0,
        index=returns.index
    )

    for ticker in available_tickers:

        portfolio_returns += (
            returns[ticker]
            * weight_map[ticker]
        )

    return portfolio_returns.dropna()


# ============================================================
# PERFORMANCE FUNCTIONS
# ============================================================

def annualized_return(
    returns
):

    if len(returns) == 0:

        return np.nan

    total_return = (
        1 + returns
    ).prod()

    years = (
        len(returns)
        / 252
    )

    if years <= 0:

        return np.nan

    return (
        total_return
        ** (1 / years)
        - 1
    )


def annualized_volatility(
    returns
):

    if len(returns) < 2:

        return np.nan

    return (
        returns.std()
        * np.sqrt(252)
    )


def sharpe_ratio(
    returns,
    risk_free_rate=0.06
):

    ann_return = (
        annualized_return(
            returns
        )
    )

    ann_vol = (
        annualized_volatility(
            returns
        )
    )

    if (
        pd.isna(ann_vol)
        or ann_vol == 0
    ):

        return np.nan

    return (
        ann_return
        - risk_free_rate
    ) / ann_vol


def sortino_ratio(
    returns,
    risk_free_rate=0.06
):

    ann_return = (
        annualized_return(
            returns
        )
    )

    downside = returns[
        returns < 0
    ]

    if downside.empty:

        return np.nan

    downside_deviation = (
        downside.std()
        * np.sqrt(252)
    )

    if downside_deviation == 0:

        return np.nan

    return (
        ann_return
        - risk_free_rate
    ) / downside_deviation


def max_drawdown(
    returns
):

    if len(returns) == 0:

        return np.nan

    cumulative = (
        1 + returns
    ).cumprod()

    peak = cumulative.cummax()

    drawdown = (
        cumulative / peak
    ) - 1

    return drawdown.min()


def value_at_risk(
    returns,
    confidence=0.95
):

    if len(returns) == 0:

        return np.nan

    return np.percentile(
        returns,
        (1 - confidence) * 100
    )


def conditional_var(
    returns,
    confidence=0.95
):

    if len(returns) == 0:

        return np.nan

    var = value_at_risk(
        returns,
        confidence
    )

    losses = returns[
        returns <= var
    ]

    if losses.empty:

        return var

    return losses.mean()


def beta(
    portfolio_returns,
    benchmark_returns
):

    df = pd.concat(
        [
            portfolio_returns,
            benchmark_returns
        ],
        axis=1
    ).dropna()

    if len(df) < 2:

        return np.nan

    portfolio = df.iloc[:, 0]

    benchmark = df.iloc[:, 1]

    covariance = np.cov(
        portfolio,
        benchmark
    )[0][1]

    variance = np.var(
        benchmark
    )

    if variance == 0:

        return np.nan

    return (
        covariance / variance
    )


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_weights(
    returns,
    method="max_sharpe"
):

    if returns.empty:

        return None

    n = len(
        returns.columns
    )

    mean_returns = (
        returns.mean()
        * 252
    )

    covariance = (
        returns.cov()
        * 252
    )

    def portfolio_return(
        weights
    ):

        return np.dot(
            weights,
            mean_returns
        )

    def portfolio_volatility(
        weights
    ):

        return np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    covariance,
                    weights
                )
            )
        )

    def negative_sharpe(
        weights
    ):

        ret = portfolio_return(
            weights
        )

        vol = portfolio_volatility(
            weights
        )

        if vol == 0:

            return 999

        return -(
            (ret - 0.06)
            / vol
        )

    def volatility_objective(
        weights
    ):

        return portfolio_volatility(
            weights
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1
        }
    ]

    bounds = [
        (0, 1)
        for _ in range(n)
    ]

    initial_weights = (
        np.ones(n) / n
    )

    if method == "max_sharpe":

        objective = (
            negative_sharpe
        )

    else:

        objective = (
            volatility_objective
        )

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if result.success:

        return pd.Series(
            result.x,
            index=returns.columns
        )

    return None


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# SESSION STATE
# ============================================================

if "portfolio_df" not in st.session_state:

    st.session_state.portfolio_df = (
        pd.DataFrame(
            columns=REQUIRED_COLUMNS
        )
    )


if "portfolio_name" not in st.session_state:

    st.session_state.portfolio_name = (
        "My Portfolio"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📊 Portfolio Intelligence"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Portfolio Input",
        "Dashboard",
        "Holdings",
        "Performance",
        "Risk Analysis",
        "Benchmark",
        "Optimization",
        "Efficient Frontier",
        "Correlation",
        "Stock Analysis"
    ]
)


# ============================================================
# PORTFOLIO INPUT
# ============================================================

if page == "Portfolio Input":

    st.title(
        "📂 Portfolio Input"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📂 Upload Portfolio",
            "✏️ Manual Entry",
            "💾 Saved Portfolios"
        ]
    )


    # ========================================================
    # TAB 1 - UPLOAD CSV / EXCEL
    # ========================================================

    with tab1:

        st.subheader(
            "📂 Upload Portfolio"
        )

        portfolio_name_upload = st.text_input(
            "Portfolio Name",
            value=st.session_state.portfolio_name,
            key="upload_portfolio_name"
        )

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=[
                "csv",
                "xlsx",
                "xls"
            ],
            key="portfolio_uploader"
        )

        if uploaded_file:

            try:

                if uploaded_file.name.lower().endswith(
                    ".csv"
                ):

                    df = pd.read_csv(
                        uploaded_file
                    )

                else:

                    df = pd.read_excel(
                        uploaded_file
                    )

                df = normalize_portfolio(
                    df
                )

                # Store in current session
                st.session_state.portfolio_df = (
                    df
                )

                st.session_state.portfolio_name = (
                    portfolio_name_upload
                )

                st.success(
                    "✅ Portfolio uploaded successfully!"
                )

                st.subheader(
                    "Uploaded Portfolio"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                st.divider()

                # ------------------------------------------------
                # SAVE UPLOADED PORTFOLIO
                # ------------------------------------------------

                if st.button(
                    "💾 Save Uploaded Portfolio Permanently",
                    type="primary",
                    use_container_width=True,
                    key="save_uploaded_portfolio"
                ):

                    try:

                        save_portfolio_to_db(
                            portfolio_name_upload,
                            df
                        )

                        st.success(
                            f"✅ Portfolio "
                            f"'{portfolio_name_upload}' "
                            "saved permanently!"
                        )

                    except Exception as e:

                        st.error(
                            f"❌ Could not save portfolio: {e}"
                        )

            except Exception as e:

                st.error(
                    f"❌ Error reading file: {e}"
                )


    # ========================================================
    # TAB 2 - MANUAL ENTRY
    # ========================================================

    with tab2:

        st.subheader(
            "✏️ Manual Portfolio Entry"
        )

        manual_name = st.text_input(
            "Portfolio Name",
            value=st.session_state.portfolio_name,
            key="manual_portfolio_name"
        )

        # ----------------------------------------------------
        # Prepare data for editor
        # ----------------------------------------------------

        if not st.session_state.portfolio_df.empty:

            default_data = (
                st.session_state.portfolio_df.copy()
            )

        else:

            default_data = pd.DataFrame(
                [
                    {
                        "Ticker": "",
                        "Quantity": 0,
                        "Purchase Price": 0.0,
                        "Purchase Date":
                            pd.Timestamp.today()
                    }
                ]
            )

        # ----------------------------------------------------
        # IMPORTANT FIX
        # Convert Purchase Date to datetime
        # before st.data_editor
        # ----------------------------------------------------

        default_data[
            "Purchase Date"
        ] = pd.to_datetime(
            default_data[
                "Purchase Date"
            ],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Data Editor
        # ----------------------------------------------------

        edited_df = st.data_editor(
            default_data,
            num_rows="dynamic",
            use_container_width=True,

            column_config={

                "Ticker":
                    st.column_config.TextColumn(
                        "Ticker",
                        help=(
                            "Example: RELIANCE.NS"
                        )
                    ),

                "Quantity":
                    st.column_config.NumberColumn(
                        "Quantity",
                        min_value=0,
                        step=1
                    ),

                "Purchase Price":
                    st.column_config.NumberColumn(
                        "Purchase Price",
                        min_value=0,
                        step=0.01
                    ),

                "Purchase Date":
                    st.column_config.DateColumn(
                        "Purchase Date",
                        format="YYYY-MM-DD"
                    )
            },

            key="manual_editor"
        )

        # ----------------------------------------------------
        # UPDATE CURRENT PORTFOLIO
        # ----------------------------------------------------

        if st.button(
            "🔄 Update Current Portfolio",
            use_container_width=True
        ):

            try:

                cleaned_df = (
                    normalize_portfolio(
                        edited_df
                    )
                )

                st.session_state.portfolio_df = (
                    cleaned_df
                )

                st.session_state.portfolio_name = (
                    manual_name
                )

                st.success(
                    "✅ Current portfolio updated!"
                )

            except Exception as e:

                st.error(
                    f"❌ {e}"
                )

        st.divider()

        # ----------------------------------------------------
        # SAVE MANUAL PORTFOLIO
        # ----------------------------------------------------

        if st.button(
            "💾 Save Manual Portfolio Permanently",
            type="primary",
            use_container_width=True
        ):

            try:

                cleaned_df = (
                    normalize_portfolio(
                        edited_df
                    )
                )

                save_portfolio_to_db(
                    manual_name,
                    cleaned_df
                )

                st.session_state.portfolio_df = (
                    cleaned_df
                )

                st.session_state.portfolio_name = (
                    manual_name
                )

                st.success(
                    f"✅ Portfolio "
                    f"'{manual_name}' "
                    "saved permanently!"
                )

            except Exception as e:

                st.error(
                    f"❌ Could not save portfolio: {e}"
                )


    # ========================================================
    # TAB 3 - SAVED PORTFOLIOS
    # ========================================================

    with tab3:

        st.subheader(
            "💾 Saved Portfolios"
        )

        saved_portfolios = (
            get_saved_portfolios()
        )

        if saved_portfolios:

            selected_portfolio = st.selectbox(
                "Select Portfolio",
                saved_portfolios
            )

            col1, col2 = st.columns(2)

            # ------------------------------------------------
            # LOAD
            # ------------------------------------------------

            with col1:

                if st.button(
                    "📂 Load Portfolio",
                    use_container_width=True
                ):

                    try:

                        loaded_df = (
                            load_portfolio_from_db(
                                selected_portfolio
                            )
                        )

                        st.session_state.portfolio_df = (
                            loaded_df
                        )

                        st.session_state.portfolio_name = (
                            selected_portfolio
                        )

                        st.success(
                            f"✅ "
                            f"'{selected_portfolio}' "
                            "loaded successfully!"
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"❌ Could not load portfolio: {e}"
                        )

            # ------------------------------------------------
            # DELETE
            # ------------------------------------------------

            with col2:

                if st.button(
                    "🗑️ Delete Portfolio",
                    use_container_width=True
                ):

                    try:

                        delete_portfolio_from_db(
                            selected_portfolio
                        )

                        st.success(
                            f"✅ "
                            f"'{selected_portfolio}' "
                            "deleted."
                        )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"❌ Could not delete portfolio: {e}"
                        )

        else:

            st.info(
                "No saved portfolios found."
            )


    # ========================================================
    # CURRENT PORTFOLIO
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Current Portfolio"
    )

    if not st.session_state.portfolio_df.empty:

        st.write(
            f"**Portfolio:** "
            f"{st.session_state.portfolio_name}"
        )

        st.dataframe(
            st.session_state.portfolio_df,
            use_container_width=True
        )

        csv_data = (
            st.session_state.portfolio_df
            .to_csv(
                index=False
            )
        )

        st.download_button(
            "⬇️ Download Current Portfolio CSV",
            csv_data,
            file_name="portfolio.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            "🗑️ Clear Current Portfolio",
            use_container_width=True
        ):

            st.session_state.portfolio_df = (
                pd.DataFrame(
                    columns=REQUIRED_COLUMNS
                )
            )

            st.success(
                "Current portfolio cleared."
            )

            st.rerun()

    else:

        st.info(
            "No portfolio loaded."
        )

    st.stop()


# ============================================================
# CHECK PORTFOLIO
# ============================================================

if st.session_state.portfolio_df.empty:

    st.warning(
        "⚠️ Please upload or create a portfolio first."
    )

    st.stop()


# ============================================================
# CALCULATE HOLDINGS
# ============================================================

with st.spinner(
    "Loading market data..."
):

    holdings = calculate_holdings(
        st.session_state.portfolio_df
    )


if holdings.empty:

    st.error(
        "Unable to calculate portfolio holdings."
    )

    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "📊 Portfolio Dashboard"
    )

    st.caption(
        f"Portfolio: "
        f"{st.session_state.portfolio_name}"
    )

    total_invested = holdings[
        "Invested Value"
    ].sum()

    total_current = holdings[
        "Current Value"
    ].sum()

    total_pnl = holdings[
        "P&L"
    ].sum()

    if total_invested != 0:

        total_return = (
            total_pnl
            / total_invested
            * 100
        )

    else:

        total_return = 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Invested",
        f"₹{total_invested:,.2f}"
    )

    c2.metric(
        "Current Value",
        f"₹{total_current:,.2f}"
    )

    c3.metric(
        "Total P&L",
        f"₹{total_pnl:,.2f}"
    )

    c4.metric(
        "Overall Return",
        f"{total_return:.2f}%"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            holdings,
            names="Ticker",
            values="Current Value",
            title="Portfolio Allocation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            holdings,
            x="Ticker",
            y="P&L",
            title="Profit / Loss by Stock"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "📋 Portfolio Summary"
    )

    st.dataframe(
        holdings,
        use_container_width=True
    )


# ============================================================
# HOLDINGS
# ============================================================

elif page == "Holdings":

    st.title(
        "📦 Holdings"
    )

    st.dataframe(
        holdings,
        use_container_width=True
    )

    st.subheader(
        "Portfolio Allocation"
    )

    fig = px.bar(
        holdings,
        x="Ticker",
        y="Portfolio Weight %",
        title="Portfolio Weight by Stock"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "Performance":

    st.title(
        "📈 Performance Analysis"
    )

    with st.spinner(
        "Calculating portfolio returns..."
    ):

        portfolio_returns = (
            build_portfolio_returns(
                holdings
            )
        )

    if portfolio_returns.empty:

        st.warning(
            "Not enough market data available."
        )

        st.stop()

    annual_return = (
        annualized_return(
            portfolio_returns
        )
    )

    annual_vol = (
        annualized_volatility(
            portfolio_returns
        )
    )

    sharpe = (
        sharpe_ratio(
            portfolio_returns
        )
    )

    sortino = (
        sortino_ratio(
            portfolio_returns
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Annualized Return",
        f"{annual_return * 100:.2f}%"
    )

    c2.metric(
        "Annualized Volatility",
        f"{annual_vol * 100:.2f}%"
    )

    c3.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}"
    )

    c4.metric(
        "Sortino Ratio",
        f"{sortino:.2f}"
    )

    cumulative = (
        1 + portfolio_returns
    ).cumprod()

    fig = px.line(
        x=cumulative.index,
        y=cumulative.values,
        title="Portfolio Growth"
    )

    fig.update_yaxes(
        title="Growth of ₹1"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Daily Return Distribution"
    )

    ret_df = pd.DataFrame(
        {
            "Daily Return":
                portfolio_returns
        }
    )

    fig = px.histogram(
        ret_df,
        x="Daily Return",
        nbins=50,
        title="Distribution of Daily Returns"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.title(
        "⚠️ Risk Analysis"
    )

    portfolio_returns = (
        build_portfolio_returns(
            holdings
        )
    )

    if portfolio_returns.empty:

        st.warning(
            "Not enough data available."
        )

        st.stop()

    mdd = max_drawdown(
        portfolio_returns
    )

    var = value_at_risk(
        portfolio_returns
    )

    cvar = conditional_var(
        portfolio_returns
    )

    volatility = (
        annualized_volatility(
            portfolio_returns
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Annual Volatility",
        f"{volatility * 100:.2f}%"
    )

    c2.metric(
        "Maximum Drawdown",
        f"{mdd * 100:.2f}%"
    )

    c3.metric(
        "VaR 95%",
        f"{var * 100:.2f}%"
    )

    c4.metric(
        "CVaR 95%",
        f"{cvar * 100:.2f}%"
    )

    cumulative = (
        1 + portfolio_returns
    ).cumprod()

    peak = cumulative.cummax()

    drawdown = (
        cumulative / peak
    ) - 1

    fig = px.area(
        x=drawdown.index,
        y=drawdown.values,
        title="Portfolio Drawdown"
    )

    fig.update_yaxes(
        tickformat=".2%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# BENCHMARK
# ============================================================

elif page == "Benchmark":

    st.title(
        "📊 Benchmark Comparison"
    )

    benchmark_name = st.selectbox(
        "Select Benchmark",
        list(
            BENCHMARKS.keys()
        )
    )

    benchmark_ticker = (
        BENCHMARKS[
            benchmark_name
        ]
    )

    portfolio_returns = (
        build_portfolio_returns(
            holdings
        )
    )

    benchmark_data = yf.download(
        benchmark_ticker,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    if benchmark_data.empty:

        st.warning(
            "Benchmark data unavailable."
        )

        st.stop()

    benchmark_close = (
        benchmark_data["Close"]
    )

    if isinstance(
        benchmark_close,
        pd.DataFrame
    ):

        benchmark_close = (
            benchmark_close.iloc[:, 0]
        )

    benchmark_returns = (
        benchmark_close
        .pct_change()
        .dropna()
    )

    comparison = pd.concat(
        [
            portfolio_returns.rename(
                "Portfolio"
            ),
            benchmark_returns.rename(
                benchmark_name
            )
        ],
        axis=1
    ).dropna()

    if comparison.empty:

        st.warning(
            "Not enough overlapping data."
        )

        st.stop()

    cumulative = (
        1 + comparison
    ).cumprod()

    fig = px.line(
        cumulative,
        title="Portfolio vs Benchmark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    portfolio_return = (
        annualized_return(
            comparison[
                "Portfolio"
            ]
        )
    )

    benchmark_return = (
        annualized_return(
            comparison[
                benchmark_name
            ]
        )
    )

    portfolio_beta = beta(
        comparison["Portfolio"],
        comparison[
            benchmark_name
        ]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Portfolio Annual Return",
        f"{portfolio_return * 100:.2f}%"
    )

    c2.metric(
        "Benchmark Annual Return",
        f"{benchmark_return * 100:.2f}%"
    )

    c3.metric(
        "Beta",
        f"{portfolio_beta:.2f}"
    )


# ============================================================
# OPTIMIZATION
# ============================================================

elif page == "Optimization":

    st.title(
        "⚙️ Portfolio Optimization"
    )

    tickers = holdings[
        "Ticker"
    ].tolist()

    if len(tickers) < 2:

        st.warning(
            "Optimization requires at least 2 stocks."
        )

        st.stop()

    prices = yf.download(
        tickers,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    if prices.empty:

        st.warning(
            "Not enough market data."
        )

        st.stop()

    close_prices = prices[
        "Close"
    ]

    returns = (
        close_prices
        .pct_change()
        .dropna()
    )

    method = st.selectbox(
        "Optimization Method",
        [
            "Maximum Sharpe Ratio",
            "Minimum Volatility"
        ]
    )

    optimization_method = (
        "max_sharpe"
        if method == "Maximum Sharpe Ratio"
        else "min_volatility"
    )

    optimal_weights = (
        optimize_weights(
            returns,
            optimization_method
        )
    )

    if optimal_weights is None:

        st.error(
            "Optimization failed."
        )

        st.stop()

    result_df = pd.DataFrame(
        {
            "Ticker":
                optimal_weights.index,
            "Optimal Weight %":
                optimal_weights.values
                * 100
        }
    )

    st.subheader(
        "Optimal Portfolio Allocation"
    )

    st.dataframe(
        result_df,
        use_container_width=True
    )

    fig = px.pie(
        result_df,
        names="Ticker",
        values="Optimal Weight %",
        title="Optimized Allocation"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# EFFICIENT FRONTIER
# ============================================================

elif page == "Efficient Frontier":

    st.title(
        "📈 Efficient Frontier"
    )

    tickers = holdings[
        "Ticker"
    ].tolist()

    if len(tickers) < 2:

        st.warning(
            "Efficient Frontier requires at least 2 stocks."
        )

        st.stop()

    prices = yf.download(
        tickers,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    if prices.empty:

        st.warning(
            "Not enough market data."
        )

        st.stop()

    close_prices = prices[
        "Close"
    ]

    returns = (
        close_prices
        .pct_change()
        .dropna()
    )

    num_portfolios = 100

    results = []

    for _ in range(
        num_portfolios
    ):

        weights = np.random.random(
            len(tickers)
        )

        weights = (
            weights
            / weights.sum()
        )

        portfolio_return = (
            np.dot(
                weights,
                returns.mean()
                * 252
            )
        )

        portfolio_volatility = np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    returns.cov()
                    * 252,
                    weights
                )
            )
        )

        results.append(
            [
                portfolio_volatility,
                portfolio_return
            ]
        )

    frontier_df = pd.DataFrame(
        results,
        columns=[
            "Volatility",
            "Return"
        ]
    )

    fig = px.scatter(
        frontier_df,
        x="Volatility",
        y="Return",
        title="Efficient Frontier",
        labels={
            "Volatility":
                "Annualized Volatility",
            "Return":
                "Annualized Return"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CORRELATION
# ============================================================

elif page == "Correlation":

    st.title(
        "🔗 Stock Correlation"
    )

    tickers = holdings[
        "Ticker"
    ].tolist()

    if len(tickers) < 2:

        st.warning(
            "Correlation requires at least 2 stocks."
        )

        st.stop()

    prices = yf.download(
        tickers,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    if prices.empty:

        st.warning(
            "Not enough data."
        )

        st.stop()

    close_prices = prices[
        "Close"
    ]

    returns = (
        close_prices
        .pct_change()
        .dropna()
    )

    correlation_matrix = (
        returns.corr()
    )

    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        title="Correlation Matrix",
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# STOCK ANALYSIS
# ============================================================

elif page == "Stock Analysis":

    st.title(
        "🔎 Stock Analysis"
    )

    ticker = st.selectbox(
        "Select Stock",
        holdings[
            "Ticker"
        ].tolist()
    )

    stock = yf.Ticker(
        ticker
    )

    history = stock.history(
        period="1y",
        auto_adjust=True
    )

    if history.empty:

        st.warning(
            "No data available for this stock."
        )

        st.stop()

    current_price = float(
        history[
            "Close"
        ].iloc[-1]
    )

    first_price = float(
        history[
            "Close"
        ].iloc[0]
    )

    one_year_return = (
        current_price
        / first_price
        - 1
    )

    high = float(
        history[
            "High"
        ].max()
    )

    low = float(
        history[
            "Low"
        ].min()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Price",
        f"₹{current_price:,.2f}"
    )

    c2.metric(
        "1Y Return",
        f"{one_year_return * 100:.2f}%"
    )

    c3.metric(
        "1Y High",
        f"₹{high:,.2f}"
    )

    c4.metric(
        "1Y Low",
        f"₹{low:,.2f}"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name=ticker
        )
    )

    fig.update_layout(
        title=(
            f"{ticker} - "
            "1 Year Price Chart"
        ),
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Recent Price Data"
    )

    st.dataframe(
        history.tail(20),
        use_container_width=True
    )