import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import streamlit as st
import base64


#1.Page setting and financial data cleaning

st.set_page_config(page_title="Black-Litterman Optimization", layout="wide")
df_p = pd.read_pickle("/Users/new/Desktop/QARM II/Project/sentratio_and_price_closetoclose_adj2.pkl")
tickers = ['AAPL', 'AMD', 'AMRN', 'AMZN', 'BABA', 'BAC', 'BB','GLD', 'IWM',
           'JNUG', 'MNKD', 'NFLX', 'PLUG', 'QQQ', 'SPY', 'TSLA', 'UVXY']
data = yf.download(
    tickers=tickers,
    start="2014-09-22",
    end="2020-03-23",
    interval="1mo",
    auto_adjust=True
)

df = data[['Close']]
df.columns = df.columns.droplevel()
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.date
df_melted = pd.melt(
    df,
    id_vars=['Date'],
    var_name='ticker',
    value_name='price'
)
df_p = df_p[['date', 'ticker', 'Nbullish', 'Nbearish', 'Polarity']]
df_p['Date'] = pd.to_datetime(df_p['date'])
df_p.set_index('Date', inplace=True)

# Monthly polarity computation
monthly_polarity = (
    df_p.groupby([df_p.index.to_period('M'), 'ticker'])
    .apply(
        lambda x: (x['Nbullish'].sum() - x['Nbearish'].sum()) /
                  (x['Nbullish'].sum() + x['Nbearish'].sum() + 10)
    )
    .reset_index(name='monthly_polarity')
)
monthly_polarity.set_index('Date', inplace=True)
monthly_polarity = monthly_polarity[monthly_polarity.index >= '2014-10-01']
df_melted['Date'] = pd.to_datetime(df_melted['Date'])
df_melted['Date'] = df_melted['Date'].dt.strftime('%Y-%m')
df_melted.set_index('Date', inplace=True)
df_melted_reset = df_melted.reset_index()
monthly_polarity_reset = monthly_polarity.reset_index()
df_melted_reset['Date'] = pd.to_datetime(df_melted_reset['Date']).dt.to_period('M').dt.to_timestamp()
monthly_polarity.index = monthly_polarity.index.to_timestamp()
monthly_polarity_reset = monthly_polarity.reset_index()

merged_df = pd.merge(
    df_melted_reset,
    monthly_polarity_reset,
    how='inner',
    on=['Date', 'ticker']
)

merged_df.set_index('Date', inplace=True)
merged_df.reset_index(inplace=True)
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m')

# 2. ESG data
esg_scores_df = pd.read_excel('/Users/new/Desktop/esg_score_qarm.xlsx')
esg_scores_df = esg_scores_df.rename(columns={esg_scores_df.columns[0]: 'Name'})
esg_scores_df.columns = [
    col.strftime('%Y-%m') if isinstance(col, pd.Timestamp) else col
    for col in esg_scores_df.columns
]
name_to_ticker = {
    'APPLE - ESG Score': 'AAPL',
    'SPDR GOLD SHARES - ESG Score': 'GLD',
    'ADVANCED MICRO DEVICES - ESG Score': 'AMD',
    'ISHARES RUSSELL 2000 ETF - ESG Score': 'IWM',
    'AMAZON.COM - ESG Score': 'AMZN',
    'ALIBABA GROUP HOLDING - ESG Score': 'BABA',
    'BLACKBERRY (NYS) - ESG Score': 'BB',
    'BANK OF AMERICA - ESG Score': 'BAC',
    'JNUG - ESG Scor': 'JNUG',
    'INVESCO QQQ TRUST - ESG Score': 'QQQ',
    'MANNKIND - ESG Score': 'MNKD',
    'NETFLIX - ESG Score': 'NFLX',
    'PLUG POWER - ESG Score': 'PLUG',
    'SPDR S&P 500 ETF TRUST - ESG Score': 'SPY',
    'TESLA - ESG Score': 'TSLA',
    'PROSHARES ULTRAVIX SHORT-TERM FUTURES ETF - ESG Score': 'UVXY',
    'AMARIN - ESG Score': 'AMRN'
}

esg_scores_df['Name'] = esg_scores_df['Name'].map(name_to_ticker)
esg_scores_df = esg_scores_df.dropna(subset=['Name'])
esg_scores_df.reset_index(drop=True, inplace=True)

#Define monthly returns
def calculate_monthly_returns(data):
    data['return'] = data.groupby('ticker')['price'].pct_change()
    return data.dropna(subset=['return'])

#Define implied returns
def calculate_implied_returns(gamma, cov_matrix, market_weights):
    return 0.02 + (1 / gamma) * cov_matrix @ market_weights

# Define ERC constraint
def erc_constraint(weights, cov_matrix):
    total_risk = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_risk = cov_matrix @ weights / total_risk
    risk_contributions = weights * marginal_risk
    return np.std(risk_contributions)

# Define ESG constraint
def esg_constraint(weights, esg_scores, esg_threshold):
    return np.dot(weights, esg_scores) - esg_threshold

# Define Budget Constraint (always included)
def budget_constraint(weights):
    return np.sum(weights) - 1

# Define black litterman optimisation
def optimize_black_litterman(
    cov_matrix, implied_returns, tau, P, Q, omega, bounds =None,
    esg_scores_df=None, current_date=None, tickers=None, esg_threshold=None,
    use_erc=False, use_esg=False
):
    n_assets = len(implied_returns)

    # Calculate mu_bar (posterior expected returns)
    Gamma = tau * cov_matrix
    mu_bar = implied_returns + Gamma @ P.T @ np.linalg.inv(P @ Gamma @ P.T + omega) @ (Q - P @ implied_returns)


    # Objective function
    def objective(weights):
        portfolio_return = weights @ mu_bar
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        return -portfolio_return  / portfolio_volatility

    constraints = [{'type': 'eq', 'fun': budget_constraint}]


    if use_erc:
        constraints.append({'type': 'eq', 'fun': lambda weights: erc_constraint(weights, cov_matrix)})

    # Add ESG constraint if enabled
    if use_esg and esg_scores_df is not None and current_date is not None and len(tickers) > 0:
        previous_date = pd.to_datetime(current_date) - pd.DateOffset(months=1)
        previous_month = previous_date.strftime('%Y-%m')
        esg_scores = esg_scores_df.set_index('Name')[previous_month].reindex(tickers).fillna(0).values
        constraints.append({'type': 'ineq', 'fun': lambda weights: esg_constraint(weights, esg_scores, esg_threshold)})

    # Initial guess
    x0 = np.ones(n_assets) / n_assets

    # Perform optimization
    result = minimize(objective, x0, bounds=bounds, constraints=constraints)
    return result.x


def calculate_sentiment_adjusted_forecast(training_data, tickers, num_simulations=300000, seed = 42):
    np.random.seed(seed)
    Q = []
    for ticker in tickers:
        ticker_data = training_data[training_data['ticker'] == ticker]

        if len(ticker_data) < 2:
            Q.append(0)
            continue

        mean_return = ticker_data['return'].mean()
        std_return = ticker_data['return'].std()
        initial_price = ticker_data['price'].iloc[-1]

        random_shocks = np.random.normal(mean_return, std_return, num_simulations)
        simulated_prices = initial_price * np.exp(random_shocks)

        S_MC_plus = simulated_prices.max()
        S_MC_minus = simulated_prices.min()
        sentiment_score = ticker_data['monthly_polarity'].iloc[-1]

        if sentiment_score > 0:
            S_T_adjusted = initial_price + (S_MC_plus - initial_price) * sentiment_score
        else:
            S_T_adjusted = initial_price - (initial_price - S_MC_minus) * abs(sentiment_score)

        Q.append(S_T_adjusted / initial_price - 1)
    return np.array(Q)


# Function to calculate realized ESG scores for each month
def calculate_realized_esg(optimal_weights, esg_scores_df):
    realized_esg = {}
    for date, weights in optimal_weights.items():
        # Ensure weights and ESG data align
        weights_df = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
        esg_scores_for_month = esg_scores_df[['Name', date]].rename(columns={date: 'ESG Score'})
        merged = pd.merge(weights_df, esg_scores_for_month, left_on='Ticker', right_on='Name', how='inner')
        realized_esg[date] = (merged['Weight'] * merged['ESG Score']).sum()
    return realized_esg


# ---- Création des onglets ----
st.markdown(
    """
    <style>
    /* Style de base pour les tabs */
    div[role="tablist"] > div {
        font-size: 18px; /* Taille de police par défaut */
        font-weight: bold;
        color: #000; /* Couleur du texte */
        transition: transform 0.3s ease, font-size 0.3s ease; /* Animation douce */
    }

    /* Effet hover (agrandissement lors du survol) */
    div[role="tablist"] > div:hover {
        transform: scale(1.1); /* Agrandissement léger */
        font-size: 20px; /* Augmente légèrement la taille de la police */
        color: #007bff; /* Change la couleur lors du survol (facultatif) */
    }

    /* Onglet actif (par exemple, pour le distinguer) */
    div[role="tablist"] > div[aria-selected="true"] {
        font-size: 22px; /* Taille légèrement plus grande pour l'onglet actif */
        color: #007bff; /* Couleur différente pour l'onglet actif */
    }
    </style>
    """,
    unsafe_allow_html=True
)


tabs = st.tabs(["Home", "Presentation", "KYC", "Optimization", "Cumulative returns and statistics"])

with tabs[0]:
    st.markdown("""
        <style>
        body {
            background-color: #f5f5f5;
        }
        h1 {
            color: #1a1a1a;
            font-family: 'Arial', sans-serif;
            text-align: center;
            font-weight: 700;
        }
        .stButton button {
            background-color: #5fa8d3; /* Bleu clair */
            color: white; /* Police blanche */
            font-size: 16px; /* Taille de la police */
            border-radius: 10px; /* Coins arrondis */
            padding: 10px 20px; /* Ajustement de l'espace intérieur */
            border: none; /* Retire les bordures */
            transition: all 0.3s ease; /* Animation fluide */
        }
        .stButton button:hover {
            background-color: #2d82c7; /* Bleu légèrement plus foncé au survol */
            color: white !important; /* Force la police à rester blanche */
            transform: scale(1.1); /* Agrandissement du bouton */
        }
        .stButton button:focus, .stButton button:active {
            background-color: #2d82c7; /* Couleur identique à l'état "hover" */
            color: white !important; /* Police blanche forcée */
            transform: scale(1.1); /* Conserver l'effet d'agrandissement */
            outline: none; /* Retire le contour de focus par défaut */
        }
        .stSelectbox label {
            font-weight: bold;
            color: #1a1a1a;
            font-size: 16px;
        }
        .logo-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .logo-container img {
            max-width: 200px; /* Taille maximale de l'image */
            border-radius: 10px; /* Coins légèrement arrondis */
        }
        .title-container {
            text-align: center;
            margin-top: 20px;
        }
        .title-container h1 {
            font-size: 36px;
            font-family: 'Arial', sans-serif;
            color: #333;
        }
        </style>
        """, unsafe_allow_html=True)


    st.markdown(
        """
        <div style="text-align: center; margin: 50px 0;">
            <img src="data:image/jpeg;base64,{}" alt="BENGL Alpha Logo" style="max-width: 500px; height: auto;">
        </div>
        """.format(
            base64.b64encode(open("/Users/new/Desktop/PHOTO-2024-11-30-17-57-29.jpg", "rb").read()).decode("utf-8")
        ),
        unsafe_allow_html=True
    )



    st.markdown(
        """
        <div style="background-color:#f5f5f5; padding:20px; border-radius:10px; max-width:800px; margin:auto; text-align:center;">
            <p style="font-size:18px; line-height:1.8; color:#333;">
                Welcome to <strong>BENGL Alpha</strong>, a pioneering pension fund that combines cutting-edge technology 
                with financial expertise to deliver optimized and sustainable investment solutions. 
                Our innovative approach integrates traditional asset management principles with 
                advanced <strong>sentiment analysis</strong>, harnessing insights from global markets, 
                news sources, and social media.
            </p>
            <p style="font-size:18px; line-height:1.8; color:#333;">
                At BENGL Alpha, we are committed to empowering our investors with portfolios 
                designed to maximize returns while adhering to <strong>Environmental, Social, and Governance (ESG)</strong> principles. 
                Join us as we redefine the future of pension fund management with integrity, innovation, and performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("Meet the team")

    team_members = [
        {
            "name": "Guillaume Granger",
            "role": "Data Scientist",
            "bio": "Data scientist passionate about machine learning and predictive analytics.",
            "linkedin": "https://www.linkedin.com/in/guillaumelgranger/",
            "image": "https://media.licdn.com/dms/image/v2/D4E03AQHNVx8d9k_51Q/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1726925269499?e=1738195200&v=beta&t=hREIPKI68B-kXdXfjSDouZy4cz3kOHlddnXDWZmN1q4"
        },
        {
            "name": "Bruno Oliveira da Rocha",
            "role": "Data Scientist",
            "bio": "Data scientist with experience in sentiment analysis and NLP.",
            "linkedin": "https://www.linkedin.com/in/bruno-oliveira-da-rocha-979a43326/",
            "image": "https://media.licdn.com/dms/image/v2/D4E12AQEud3Ll5MI7cQ/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1660833954461?e=1738195200&v=beta&t=3hCEiuzjv5xFnDp5z0ADmOKJe2vrq6mTRqCyD8InkZE"
        },
        {
            "name": "Enzo Rua",
            "role": "Portfolio Manager",
            "bio": "Portfolio manager with a focus on risk management and financial optimization.",
            "linkedin": "https://www.linkedin.com/in/enzo-rua-9225611a0/",
            "image": "https://media.licdn.com/dms/image/v2/D4D03AQGktvVlmxNkjg/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1731951116256?e=1738195200&v=beta&t=ou0wyGGbKMlXAviW0N17y-7mbT_uHhuCdcNEy_2TV-Y"
        },
        {
            "name": "Niels Scharwatt",
            "role": "Software Engineer",
            "bio": "Software engineer specializing in backend systems and data processing pipelines.",
            "linkedin": "https://www.linkedin.com/in/niels-scharwatt/",
            "image": "https://media.licdn.com/dms/image/v2/D4E03AQEpOac05YJLeg/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1698895473994?e=1738195200&v=beta&t=Ke2FvktbTCTYtgS_RCpuFc5wMp8ga1DwJcLNb3IYOTo"
        },
        {
            "name": "Ludovik Thorens",
            "role": "Community Manager",
            "bio": "Community manager with expertise in audience engagement and communication strategies.",
            "linkedin": "https://www.linkedin.com/in/ludovik-thorens-584b81221/",
            "image": "https://media.licdn.com/dms/image/v2/D4E03AQGDqmSows69GQ/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1731600461003?e=1738195200&v=beta&t=zNKwoR29GG8S3gQVsdjYJ5Bv1VIPDXRCGXlQWA_HTS4",
        },
    ]

    for member in team_members:
        with st.expander(member["name"]):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(member["image"], width=150)
            with col2:
                st.markdown(f"*Role:* {member['role']}")
                st.markdown(f"*Bio:* {member['bio']}")
                st.markdown(f"[LinkedIn Profile]({member['linkedin']})")

with tabs[1]:
    st.title(" Strategy Presentation")


    st.header("🌐 Stock universe")
    st.markdown(
        "In this first section, you can learn and explore a little more about the **17** TICKERS we focus on and use:")


    stocks = {
        "AAPL": "Apple Inc. - A leading technology company known for its innovative products like the iPhone, iPad and MacBook.",
        "AMD": "Advanced Micro Devices Inc. - A semiconductor company specializing in CPUs and GPUs for personal computers, gaming consoles and data centers.",
        "AMRN": "Amarin Corporation - A biopharmaceutical company focused on cardiovascular health and lipid science.",
        "AMZN": "Amazon.com Inc. - A global leader in e-commerce, cloud computing and digital streaming services.",
        "BABA": "Alibaba Group Holding Ltd. - A major Chinese e-commerce and technology conglomerate with global operations.",
        "BAC": "Bank of America Corporation - A leading multinational financial services company offering banking and wealth management solutions.",
        "BB": "BlackBerry Ltd. - A software company focusing on cybersecurity and Internet of Things (IoT) solutions.",
        "MNKD": "MannKind Corporation - A biotechnology company focused on developing inhalable therapeutic treatments.",
        "NFLX": "Netflix Inc. - A global leader in streaming entertainment and original content production.",
        "PLUG": "Plug Power Inc. - A provider of hydrogen fuel cell systems for industrial applications.",
        "TSLA": "Tesla Inc. - An electric vehicle and clean energy company known for its innovative technologies.",
    }

    etfs = {
        "GLD": "SPDR Gold Shares - Tracks the performance of the price of gold by holding physical gold. [Fact Sheet](https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-gld.pdf)",
        "IWM": "iShares Russell 2000 ETF - Tracks the performance of the Russell 2000 Index, representing small-cap U.S. companies. [Fact Sheet](https://www.ishares.com/us/literature/fact-sheet/iwm-ishares-russell-2000-etf-fund-fact-sheet-en-us.pdf)",
        "JNUG": "Direxion Daily Junior Gold Miners Index Bull 2X Shares - Seeks to deliver 200% of the daily performance of the Junior Gold Miners Index. [Fact Sheet](https://www.direxion.com/uploads/JNUG-JDST-Fact-Sheet.pdf)",
        "QQQ": "Invesco QQQ Trust Series - Tracks the performance of the 100 largest non-financial companies listed on the NASDAQ. [Fact Sheet](https://www.invesco.com/us-rest/contentdetail?contentId=3a48e01e98630410VgnVCM10000046f1bf0aRCRD&dnsName=us)",
        "SPY": "SPDR S&P 500 ETF Trust - Replicates the performance of the S&P 500 Index, which includes 500 large-cap U.S. companies. [Fact Sheet](https://www.ssga.com/library-content/products/factsheets/etfs/us/factsheet-us-en-spy.pdf)",
        "UVXY": "ProShares Ultra VIX Short-Term Futures ETF - Seeks to deliver 1.5 times the daily performance of the S&P 500 VIX Short-Term Futures Index. [Fact Sheet](https://www.proshares.com/globalassets/proshares/fact-sheet/prosharesfactsheetuvxy.pdf)"
    }


    category = st.radio("Select a category to explore:", ["11 Stocks", "6 ETFs"])

    if category == "11 Stocks":
        st.subheader("11 Stocks")
        for ticker, description in stocks.items():
            with st.expander(f"{ticker}: {description.split(' - ')[0]}"):
                st.write(description)
    else:
        st.subheader("6 ETFs")
        for ticker, description in etfs.items():
            with st.expander(f"{ticker}: {description.split(' - ')[0]}"):
                st.markdown(description, unsafe_allow_html=True)

    # Section: Black-Litterman Approach: Calculating Implied Returns
    st.header("🛠️ Methodology: Black-Litterman approach")

    st.markdown(
        """
        In this second section, we explain the complete Black-Litterman approach, which combines two sources of information: 
        - The initial allocation to calculate the implied returns. 
        - The manager views (sentiment analysis) to incorporate in the portfolio optimization.
        """
    )

    st.subheader("Step 1: Calculating implied returns")
    st.markdown("The formula use for the computation of the implied returns, is the following one:")
    st.latex(
        r"""
        \tilde{\mu} = r 1_n + \frac{1}{\gamma} \Sigma x_0
        """
    )
    st.markdown("To compute, we need the following inputs:")
    # Inputs as expandable sections
    with st.expander("1. Risk-free rate: $$r$$"):
        st.markdown(
            """
            The risk-free rate represents the return on a risk-free investment, typically short-term government securities.
            We used the 3-Month Treasury Bill rate (IRX) from yfinance as a proxy for the risk-free rate.
            """
        )
    with st.expander("2. Risk-aversion parameter: $$\gamma$$"):
        st.markdown(
            r"""
            The risk-aversion parameter reflects the investor's tolerance for risk. It plays a crucial role in balancing the portfolio's expected returns and risk. 
            This parameter is estimated based on our Know Your Customer (KYC) data.
            """
        )
    with st.expander("3. Covariance matrix: $$\Sigma$$"):
        st.markdown("""
        To calculate the covariance matrix, we use the following formula based on the asset returns:
        """)

        st.latex(
            r"""
            \text{Cov}(R_i, R_j) = \frac{1}{T-1} \sum_{t=1}^T \left( R_{i,t} - \bar{R}i \right) \left( R{j,t} - \bar{R}_j \right)
            """
        )
    with st.expander("4. Initial allocation: $$x_0$$"):
        st.markdown(
            r"""
            For the initial allocation, we assume an equally weighted portfolio $$x_0$$ where each asset is assigned 
            the same weight.

            $$x_0 = \frac{1}{N}$$

            where $$N$$ is the number of assets in the portfolio. Since $$N$$ is equal to 19, the value we obtain for $$x_0 = 0,05263$$
            """
        )

    st.subheader("Step 2:  Define our matrices P, $$\mu$$ and Q (portfolio manager's views)")
    st.markdown("The formula is this one:")
    st.latex(
        r"""
        P\mu = Q + \epsilon
        """
    )

    with st.expander("The matrix $$P$$"):
        st.markdown("""
        **P** is a matrix that relates the assets $$(n)$$ to the views $$(k)$$ expressed by the portfolio manager. 
        This matrix is created statically or calculated according to the relationships expressed in the data.
        """)

    with st.expander("$$\mu$$"):
        st.markdown(
            """
            Black and Litterman assume that $\\mu$ is a Gaussian vector with tne implied vector $\\tilde{\\mu}$ and covariance matrix $\\Gamma$:
            """
        )
        st.latex(
            r"""
            \mu \sim \mathcal{N}(\tilde{\mu}, \Gamma)
            """
        )

    with st.expander("The matrix $$Q$$"):
        st.markdown("""
        $$Q$$ is a vector $$k \\times 1$$, where $$k$$ corresponds to the number of views expressed by the portfolio manager. These scores are integrated into forward-looking views, inspired by this academic paper 
        ["BERT’s Sentiment Score for Portfolio Optimization"](https://doi.org/10.1007/s00521-022-07403-1).

        ##### **a. Extracting historical data:**
        To compute $$Q$$, we need all this 3 compute from historical data:
        - The mean return: $$\mu_i = \\frac{1}{n} \\sum_{t=1}^n r_{i,t}$$,
        - The volatility: $$\\sigma_i = \\sqrt{\\frac{1}{n-1} \\sum_{t=1}^n (r_{i,t} - \\mu_i)^2}$$,
        - The initial price: $$S_0$$

        ##### **b. Monte Carlo Simulation (300'000)**
        Future price trajectories $$(S_{\\text{sim}})$$ are simulated using a log-normal distribution:

        $$
        S_{\\text{sim}} = S_0 \\cdot \\exp(\\text{random shocks})
        $$

        Where the random shocks follow a normal distribution:

        $$
        \\text{random shocks} \\sim \\mathcal{N}(\\mu_i, \\sigma_i)
        $$

        ##### **c. Extracting simulated extremes**
        For each asset, we calculate:
        - Maximum simulated price:
          $$
          S_{\\text{MC+}} = \\max(S_{\\text{sim}})
          $$
        - Minimum simulated price:
          $$
          S_{\\text{MC-}} = \\min(S_{\\text{sim}})
          $$

        ##### **d. Adjusting with Sentiment**
        The sentiment score $$(\\text{Sentiment}_i)$$ adjusts the forecast:

        - If the sentiment score is **positive** ($$\\text{Sentiment}_i > 0$$):
          $$
          S_T^{\\text{adjusted}} = S_0 + (S_{\\text{MC+}} - S_0) \\cdot \\text{Sentiment}_i
          $$
        - If the sentiment score is **negative** ($$\\text{Sentiment}_i < 0$$):
          $$
          S_T^{\\text{adjusted}} = S_0 - (S_0 - S_{\\text{MC-}}) \\cdot |\\text{Sentiment}_i|
          $$

        ##### **e. Calculating adjusted returns**
        Once the adjusted price $$(S_T^{\\text{adjusted}})$$ is computed, the expected return is defined as:

        $$
        Q_i = \\frac{S_T^{\\text{adjusted}}}{S_0} - 1
        $$

        This is done for each asset or view, resulting in a vector $$Q$$ containing the sentiment-adjusted forecasts.
        """)

    with st.expander("$$\\epsilon$$"):
        st.markdown("""
        $$\\epsilon \\sim \\mathcal{N}(0, \\Omega)$$ represents a Gaussian error term modeling the uncertainty of the views.

        - **Mean**: The error term has a mean of 0.
        - **Covariance**: $$\\Omega$$ represents the uncertainty associated with the views. 
        """)

    st.subheader("Step 3: Computing conditional expectations")

    st.markdown("""
    To compute the conditional expectations, we use:
    """)

    st.latex(
        r"""
        \bar{\mu} = \tilde{\mu} + \tau \Sigma P^T \left( P \tau \Sigma P^T + \Omega \right)^{-1} \left( Q - P \tilde{\mu} \right)
        """
    )

    with st.expander("Inputs"):
        st.markdown("""
        - **$$\\bar{\\mu}$$**: The adjusted returns after incorporating the views.
        - **$$\\tilde{\\mu}$$**: The implied returns from market equilibrium.
        - **$$\\tau$$**: A scaling factor that reflects the uncertainty in the prior estimates. We chose 0,05.
        - **$$\\Sigma$$**: The covariance matrix of asset returns.
        - **$$P$$**: The matrix linking views to assets.
        - **$$\\Omega$$**: The covariance matrix of errors in the views.
        - **$$Q$$**: The vector of views expressed by the portfolio manager.
        """)

    st.subheader("Step 4: Portfolio optimization")
    st.markdown("""
        Final step, we perform portfolio optimization using the following inputs:
        - **$$\\Sigma$$**: The covariance matrix of asset returns.
        - **$$\\bar{\\mu}$$**: The vector of conditional returns.
        - **$$\\gamma$$**: The risk-aversion parameter.

        The goal is to find the optimal weights that maximize the portfolio's expected return while minimizing its risk.
        """)

    # Section: Sentiment Analysis
    st.header("🔍 Sentiment analysis")

    st.markdown("""
    In this section, we explain how we compute **monthly sentiment scores** for each stock using data from the social media platform **Twitter**. 
    The sentiment scores are derived from the number of **bullish** and **bearish** mentions for each stock.
    """)

    # Step 1: Sentiment Data Inputs
    st.subheader("Step 1: Load sentiment data")

    st.markdown("""
    The sentiment data contains the following columns:
    - **Nbullish**: The number of positive (bullish) mentions for a stock.
    - **Nbearish**: The number of negative (bearish) mentions for a stock.
    - **Polarity**: A score representing the sentiment direction for each mention.
    """)

    # Step 2: Compute Monthly Polarity
    st.subheader("Step 2: Compute monthly sentiment scores")

    st.markdown("""
    For each stock, we aggregate the **bullish** and **bearish** mentions on a monthly basis and calculate the **monthly polarity score**:

    $$
    \\text{Polarity} = \\frac{\\text{Nbullish} - \\text{Nbearish}}{\\text{Nbullish} + \\text{Nbearish} + 10}
    $$

    Here, the denominator includes a smoothing factor of \(10\) to prevent division by zero and ensure stable scores.
    """)

    # Key Results
    st.header("📌 Key results")

    st.markdown("""
    Our methodology delivers the following outcomes:

    - **Sentiment-adjusted portfolio weights**: Incorporating behavioral signals from sentiment data to influence asset allocation.
    - **Customizable outputs**: Results can be tailored to individual risk preferences through the risk-aversion parameter ($$\\gamma$$).
    """)



with tabs[2]:
    # Titre de l'onglet
    st.title(" KYC: Know Your Customer")

    with st.expander("Note"):
        st.markdown("""
Our methodology is based on a weighted evaluation of "gamma," an indicator used to measure risk tolerance. 
Each question presents a specific financial scenario with responses ranked by increasing risk levels (A to D). 
These responses are assigned gamma **scores** (reflecting the degree of risk) and **weights** (indicating the relative importance 
of each choice). 

The gamma scores provide a numerical value (7 for A, 5 for B, 3 for C, 1 for D) to quantify the level of risk associated with each response. 
Simultaneously, the weights (matching the scores in this methodology) determine the relative influence of each response. 

The final gamma is calculated using a weighted average: each gamma score is multiplied by its weight, and the sum is divided 
by the total weights. This method ensures that individual preferences are integrated while balancing their impact, 
offering a precise and tailored diagnosis of risk profiles.

This methodology is widely used and highly popular, adopted by major financial institutions to 
assess customer risk tolerance effectively and systematically. 
Its robustness and adaptability make it a trusted tool for professional risk profiling.
        """)

    st.markdown("Please answer the following questions to determine your risk tolerance level.")

    def calculate_gamma(choices):
        gamma_values = {'A': 7, 'B': 5, 'C': 3, 'D': 1}
        weight_values = {'A': 7, 'B': 5, 'C': 3, 'D': 1}


        total_gamma = sum(gamma_values[response] * weight_values[response] for response in choices)
        total_weight = sum(weight_values[response] for response in choices)

        return total_gamma / total_weight if total_weight != 0 else 0



    questions = [
        "If you have invested a moderate amount in a stock and its value drops by 20%, what do you do?",
        "You have the opportunity to participate in a lottery with multiple options. Which one do you choose?",
        "You are offered to invest in a new volatile financial product. What do you do?",
        "You have two job offers. What is your decision?",
        "You are offered a risky real estate investment in a developing region. What do you do?",
        "You have the opportunity to invest in a start-up. How do you respond?",
        "You expect market volatility and want to protect your portfolio, what do you do?",
        "When managing your portfolio, what approach do you prioritize?"
    ]


    options = [
        ["Please select", "A) I sell immediately to avoid further losses.",
         "B) I sell part to limit losses but keep some to see how it evolves.",
         "C) I hold all my shares and wait for the market to recover.",
         "D) I buy more shares to take advantage of the low price opportunity."],
        ["Please select", "A) Win CHF500 with certainty.",
         "B) 75% chance to win CHF1,000 and 25% chance to win nothing.",
         "C) 50% chance to win CHF2,500 and 50% chance to win nothing.",
         "D) 25% chance to win CHF5,000 and 75% chance to win nothing."],
        ["Please select", "A) I refuse the investment, I prefer safer investments.",
         "B) I invest a small portion to test with minimal risk.",
         "C) I invest a moderate amount to maximize profit while limiting losses.",
         "D) I invest a large amount to maximize potential gains."],
        ["Please select", "A) The fixed salary of CHF50,000.",
         "B) A base salary of CHF40,000 with a potential bonus up to CHF60,000.",
         "C) A base salary of CHF30,000 with a potential bonus up to CHF70,000.",
         "D) A base salary of CHF20,000 with a potential bonus up to CHF100,000."],
        ["Please select", "A) I refuse the investment, too risky.",
         "B) I invest a small amount to limit the risk while being exposed to potential growth.",
         "C) I invest a moderate amount to benefit from potential growth while accepting some risk.",
         "D) I invest a large amount to maximize gains if the market grows."],
        ["Please select", "A) I do not invest, I would prefer to invest in Treasury Bonds.",
         "B) I invest a small amount in the start-up to test.",
         "C) I invest a moderate amount while taking calculated risks.",
         "D) The potential of growth is huge, I can afford to lose money."],
        ["Please select", "A) I fully hedge my portfolio using options or other derivatives to avoid losses.",
         "B) I partially hedge key positions, keeping some exposure to potential gains.",
         "C) I only hedge if market indicators suggest a downturn.",
         "D) I do not hedge and rely on diversification to manage risks."],
        ["Please select", "A) I prefer to invest entirely in bonds, prioritizing capital preservation.",
         "B) I prefer to allocate most of my portfolio in bonds, accepting lower returns in exchange for stability.",
         "C) I choose bonds with medium returns and calculated risk.",
         "D) I focus on maximizing the Sharpe ratio, aiming to optimize returns relative to risk."]
    ]


    if "final_gamma" not in st.session_state:
        st.session_state.final_gamma = None


    responses = []


    for i, question in enumerate(questions):
        response = st.selectbox(question, options[i], index=0, key=f'Q{i + 1}')
        if response != "Please select":
            response_clean = response.split(')')[0]
            responses.append(response_clean)


    col1, col2 = st.columns(2)
    with col1:
        if st.button("Calculate my final gamma"):
            if len(responses) == len(questions):
                st.session_state.final_gamma = calculate_gamma(responses)  # Stocker dans session_state
                st.success(f"Your final gamma is: {st.session_state.final_gamma:.2f}")
            else:
                st.warning("Please answer all the questions.")

with tabs[3]:

    st.title(" Dynamic Portfolio Optimization Dashboard")


    if "final_gamma" not in st.session_state or st.session_state.final_gamma is None:
        st.session_state.final_gamma = 3


    st.markdown(f"Your risk aversion parameter (gamma) is: **{st.session_state.final_gamma:.2f}**")

    st.subheader("Constraints")
    st.markdown("Which constraints do you want to apply? (Default: None)")
    constraints = st.multiselect(
        "Select Constraints",
        ["ERC", "ESG"],
        default=[]
    )


    esg_threshold = None
    if "ESG" in constraints:
        esg_threshold = st.slider("Set ESG Threshold", min_value=0, max_value=100, value=60)


    st.subheader("Short-Selling Option")
    optimization_type = st.selectbox(
        "Do you want to enable short-selling?",
        ["Enable Short-Selling", "No Short-Selling"]
    )


    if st.button("Run Optimization"):
        st.subheader("Optimization Results")
        st.write("Optimization is running for all dates...")


        tau = 0.05
        gamma = st.session_state.final_gamma
        risk_free_rate = 0.02
        window_size = 36


        available_dates = pd.date_range(start="2018-01", end="2019-12", freq="M").strftime('%Y-%m').tolist()


        optimal_weights = {}


        for current_date in pd.date_range(start="2018-01", end="2019-12", freq="M"):

            date_str = current_date.strftime('%Y-%m')

            training_end_date = current_date - pd.DateOffset(months=1)
            training_start_date = training_end_date - pd.DateOffset(months=window_size)


            training_data = merged_df[
                (merged_df["Date"] >= training_start_date.strftime('%Y-%m')) &
                (merged_df["Date"] <= training_end_date.strftime('%Y-%m'))
            ]

            training_data["return"] = training_data.groupby("ticker")["price"].pct_change()
            training_data = training_data.dropna(subset=["return"])
            returns_pivot = training_data.pivot(index="Date", columns="ticker", values="return")
            cov_matrix = returns_pivot.cov().values

            tickers_in_data = returns_pivot.columns.tolist()
            if len(tickers_in_data) != len(tickers):
                tickers = tickers_in_data


            market_weights = np.ones(len(tickers)) / len(tickers)
            implied_returns = calculate_implied_returns(gamma, cov_matrix, market_weights)


            Q = calculate_sentiment_adjusted_forecast(training_data, tickers)
            P = np.eye(len(tickers))
            omega = np.diag((P @ (tau * cov_matrix) @ P.T).diagonal())


            use_erc = "ERC" in constraints
            use_esg = "ESG" in constraints


            if optimization_type == "Enable Short-Selling":
                bounds = [(-1, 1) for _ in range(len(tickers))]
            else:
                bounds = [(0, 1) for _ in range(len(tickers))]


            weights = optimize_black_litterman(
                cov_matrix, implied_returns, tau, P, Q, omega,
                esg_scores_df=esg_scores_df, current_date=current_date, tickers=tickers,
                esg_threshold=esg_threshold if use_esg else None,
                use_erc=use_erc, use_esg=use_esg, bounds=bounds
            )


            optimal_weights[date_str] = weights

        realized_esg = calculate_realized_esg(optimal_weights, esg_scores_df)


        st.session_state.optimal_weights = optimal_weights


        weights_df = pd.DataFrame.from_dict(optimal_weights, orient="index", columns=tickers)
        weights_df.index.name = "Date"
        st.session_state.weights_df = weights_df


    if "weights_df" in st.session_state:
        weights_df = st.session_state.weights_df


        selected_date = st.selectbox("Select a date to view results:", weights_df.index)


        st.subheader(f"Results for {selected_date}")
        st.write("### Portfolio Weights")
        selected_weights = weights_df.loc[selected_date]
        st.table(selected_weights)


        st.subheader("Weight Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            selected_weights.index,
            selected_weights.values,
            color=plt.cm.Paired(np.arange(len(selected_weights)) / len(selected_weights)),
            edgecolor="black"
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Line")
        ax.set_title(f"Optimal Weights for {selected_date}", fontsize=14)
        ax.set_xlabel("Tickers")
        ax.set_ylabel("Weight")
        ax.legend()

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha="center")

        st.pyplot(fig)


        if "ESG" in constraints:
            realized_esg = calculate_realized_esg(st.session_state.optimal_weights, esg_scores_df)
            st.write("### Realized ESG Score")
            st.write(f"The realized ESG score for {selected_date} is: **{realized_esg[selected_date]:.2f}**")



with tabs[4]:

    st.title("Portfolio ESG and Performance Dashboard")
    st.markdown("This tab provides a comparison of cumulative returns and performance metrics for various portfolios.")


    if "optimal_weights" not in locals():
        st.warning("Please run the optimization first to generate the portfolio weights.")
    else:
        # List of tickers
        symbols = ['AAPL', 'AMD', 'AMRN', 'AMZN', 'BABA', 'BAC', 'BB', 'GLD', 'IWM',
                   'JNUG', 'MNKD', 'NFLX', 'PLUG', 'QQQ', 'SPY', 'TSLA', 'UVXY']

        # Download daily data
        data_daily = yf.download(
            tickers=symbols,
            start='2018-01-01',
            end='2019-12-31',
            interval="1d",
            auto_adjust=True
        )


        data_daily_close = data_daily['Close'].reset_index()


        df_daily_melted = pd.melt(
            data_daily_close,
            id_vars=['Date'],
            var_name='ticker',
            value_name='price'
        )

        # Calculate daily returns for each ticker
        df_daily_melted['return'] = df_daily_melted.groupby('ticker')['price'].pct_change()

        daily_returns = df_daily_melted.pivot(index='Date', columns='ticker', values='return')
        daily_returns.index = pd.to_datetime(daily_returns.index)
        daily_returns = daily_returns.dropna(how='all')

        def assign_monthly_weights_to_daily(weights_dict, daily_dates, tickers):
            weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=tickers)
            weights_df.index = pd.to_datetime(weights_df.index).to_period('M')  # Convert to monthly periods
            daily_weights = pd.DataFrame(index=daily_dates, columns=tickers)

            for date in daily_dates:
                month_period = pd.Period(date, freq='M')
                if month_period in weights_df.index:
                    daily_weights.loc[date] = weights_df.loc[month_period].values

            return daily_weights.fillna(method='ffill')  # Forward-fill weights for any gaps

        daily_weights = assign_monthly_weights_to_daily(optimal_weights, daily_returns.index, symbols)

        # Function to calculate portfolio daily returns
        def calculate_portfolio_daily_returns(weights_df, daily_returns):
            return (weights_df * daily_returns).sum(axis=1)

        # Calculate optimized portfolio daily returns
        portfolio_daily_returns = calculate_portfolio_daily_returns(daily_weights, daily_returns)

        # Calculate SPY benchmark daily returns
        spy_daily_returns = daily_returns['SPY']

        # Calculate equally weighted portfolio daily returns
        equal_weights = np.ones(len(daily_returns.columns)) / len(daily_returns.columns)
        equally_weighted_daily_returns = daily_returns.dot(equal_weights)

        # Calculate minimum-variance portfolio daily returns
        def calculate_min_variance_weights(cov_matrix):
            n_assets = cov_matrix.shape[0]
            ones = np.ones(n_assets)
            inv_cov = np.linalg.inv(cov_matrix)
            min_var_weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
            return min_var_weights

        cov_matrix = daily_returns.cov()
        min_var_weights = calculate_min_variance_weights(cov_matrix)
        min_variance_daily_returns = daily_returns.dot(min_var_weights)

        # Calculate daily cumulative returns
        portfolio_cum_returns = (1 + portfolio_daily_returns).cumprod()
        spy_cum_returns = (1 + spy_daily_returns).cumprod()
        equally_weighted_cum_returns = (1 + equally_weighted_daily_returns).cumprod()
        min_variance_cum_returns = (1 + min_variance_daily_returns).cumprod()

        # Normalize cumulative returns to start at 0% for comparability
        portfolio_cum_returns = (portfolio_cum_returns / portfolio_cum_returns.iloc[0]) - 1
        spy_cum_returns = (spy_cum_returns / spy_cum_returns.iloc[0]) - 1
        equally_weighted_cum_returns = (equally_weighted_cum_returns / equally_weighted_cum_returns.iloc[0]) - 1
        min_variance_cum_returns = (min_variance_cum_returns / min_variance_cum_returns.iloc[0]) - 1

        # Metrics calculation
        def calculate_metrics(returns):
            ann_return = np.mean(returns) * 252
            ann_volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility
            cum_returns = (1 + returns).cumprod()
            max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
            return {
                'Annualized Return': ann_return,
                'Annualized Volatility': ann_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Min Daily Return': returns.min(),
                'Max Daily Return': returns.max()
            }

        # Portfolio metrics
        metrics_portfolio = calculate_metrics(portfolio_daily_returns)
        metrics_spy = calculate_metrics(spy_daily_returns)
        metrics_equal = calculate_metrics(equally_weighted_daily_returns)
        metrics_min_var = calculate_metrics(min_variance_daily_returns)

        metrics_df = pd.DataFrame({
            'Optimized Portfolio': metrics_portfolio,
            'SPY (Benchmark)': metrics_spy,
            'Equally Weighted Portfolio': metrics_equal,
            'Minimum-Variance Portfolio': metrics_min_var
        })

        # Plot cumulative returns
        st.subheader("Portfolio Cumulative Returns Comparison")
        fig_cum_returns, ax_cum_returns = plt.subplots(figsize=(14, 8))
        ax_cum_returns.plot(portfolio_cum_returns, label='Optimized Portfolio', linewidth=2, color='blue')
        ax_cum_returns.plot(spy_cum_returns, label='SPY (Benchmark)', linewidth=2, linestyle='--', color='orange')
        ax_cum_returns.plot(equally_weighted_cum_returns, label='Equally Weighted Portfolio', linewidth=2, linestyle='-.', color='green')
        ax_cum_returns.plot(min_variance_cum_returns, label='Minimum-Variance Portfolio', linewidth=2, linestyle=':', color='red')
        ax_cum_returns.set_title("Cumulative Returns Comparison", fontsize=16)
        ax_cum_returns.set_xlabel('Date', fontsize=12)
        ax_cum_returns.set_ylabel('Cumulative Return', fontsize=12)
        ax_cum_returns.legend(fontsize=12, loc='upper left', frameon=True, shadow=True, fancybox=True)
        ax_cum_returns.grid(alpha=0.3)
        st.pyplot(fig_cum_returns)

        # Display performance metrics
        st.subheader("Portfolio Performance Metrics")
        st.table(metrics_df)

        # Optionally download metrics as a CSV file
        csv_metrics = metrics_df.to_csv(index=True)
        st.download_button(
            label="Download Metrics as CSV",
            data=csv_metrics,
            file_name="portfolio_metrics.csv",
            mime="text/csv"
        )

        # Realized ESG Scores Plot
        st.subheader("Realized ESG Scores Over Time")
        fig_esg, ax_esg = plt.subplots(figsize=(14, 7))
        dates = list(realized_esg.keys())
        scores = list(realized_esg.values())

        # Enhanced line plot
        ax_esg.plot(
            dates,
            scores,
            marker='o',
            linestyle='-',
            color='royalblue',
            linewidth=2,
            markersize=8,
            label='Realized ESG Score'
        )

        ax_esg.set_title(
            "Realized ESG Score of Portfolio Over Time",
            fontsize=18,
            fontweight='bold',
            pad=20
        )
        ax_esg.set_xlabel("Date", fontsize=14, labelpad=10)
        ax_esg.set_ylabel("Realized ESG Score", fontsize=14, labelpad=10)
        ax_esg.tick_params(axis='x', rotation=45, labelsize=12)
        ax_esg.tick_params(axis='y', labelsize=12)
        ax_esg.grid(axis='y', linestyle='--', alpha=0.5)
        ax_esg.legend(fontsize=12, loc='upper left', frameon=True, shadow=True, fancybox=True)

        for i, score in enumerate(scores):
            ax_esg.annotate(
                f"{score:.2f}",
                (dates[i], score),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=10,
                color='darkblue'
            )

        plt.tight_layout()
        st.pyplot(fig_esg)



