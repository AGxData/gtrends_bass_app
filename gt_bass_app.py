import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import arviz as az

from pytrends.request import TrendReq
from datetime import date, datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

import pymc as pm
from pymc_marketing.bass.model import create_bass_model
from pymc_marketing.plot import plot_curve
from pymc_marketing.prior import Prior, Scaled
import nutpie
import numba

import streamlit as st

az.style.use("arviz-whitegrid")

# Setting up Streamlit app
st.title("Bayesian Bass Diffusion Model (PyMC-Marketing + Google Trends)")

# User input
keyword = st.text_input("Keyword:", "iphone 16")
geo = st.text_input("Geo (country code, Ex: US):", "US")
start_date = st.date_input("Release Date", datetime(2024, 9, 24))
use_periods = st.checkbox("Use periods instead of end date?")

if use_periods:
    periods = st.number_input("Number of periods", min_value = 1, value = 12)
    period_unit = st.selectbox("Period unit", ["days", "weeks", "months"], index = 1)
    end_date = None
else:
    end_date = st.date_input("End Date", datetime(2024, 11, 19))
    periods = None
    period_unit = None 


def get_pytrends_data(keywords, start_date, end_date = None, periods = None, period_unit = "weeks", geo = "US", cat = 0, gprop = "", hl = "en-US"):
    def to_dt(d):
        """Convert various date-like objects into a datetime."""
        if isinstance(d, (datetime, pd.Timestamp)):
            return d
        if isinstance(d, date):  # handles st.date_input outputs
            return datetime(d.year, d.month, d.day)
        if isinstance(d, str):
            return pd.to_datetime(d)
        raise ValueError(f"Unsupported date type: {type(d)}")
    
    start_dt = to_dt(start_date)
    
    if periods is not None:
        if isinstance(periods, timedelta):
            end_dt = start_dt + periods
        else:
            units = {
                "days": "days",
                "day": "days",
                "weeks": "weeks",
                "week": "weeks",
                "months": "days",
                "month": "days",
            }
            unit = units.get(period_unit.lower())
            if not unit:
                raise ValueError(f"Unsupported period_unit: {period_unit}")
            days_to_add = periods * (30 if "month" in period_unit else 1)
            delta = timedelta(**{unit: days_to_add})
            end_dt = start_dt + delta
    elif end_date:
        end_dt = to_dt(end_date)
    else:
        raise ValueError("Either end_date or periods must be provided.")
    
    timeframe = f"{start_dt.strftime(r"%Y-%m-%d")} {end_dt.strftime(r"%Y-%m-%d")}"
    
    kw_list = [keywords] if isinstance(keywords, str) else keywords
    
    pytrends = TrendReq(hl = hl, tz = 360)
    pytrends.build_payload(kw_list = kw_list, timeframe = timeframe, geo = geo, cat = cat, gprop = gprop)
    df = pytrends.interest_over_time()
    
    if "isPartial" in df.columns:
        df = df.drop(columns = "isPartial")
    
    df = df.rename(columns = {kw_list[0]: "interest"})
    
    return df.drop(columns = "isPartial") if "isPartial" in df.columns else df


def resample_minmax_days_to_weeks(df, column, plot = False):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Scale daily values (MinMax from original min to 100)
    original_vals = df[[column]].values
    original_min = original_vals.min()
    scaler = MinMaxScaler(feature_range = (1, 100))
    df[f"scaled_{column}"] = scaler.fit_transform(original_vals)
    
    # Resample to weekly
    df_resampled = df[f"scaled_{column}"].resample("7D", label = "left", closed = "left").mean()
    df_resampled = df_resampled.dropna().round().astype(int)
    
    # Scale weekly values (MinMax from weekly min to 100)
    weekly_vals = df_resampled.values.reshape(-1, 1)
    weekly_min = weekly_vals.min()
    weekly_scaler = MinMaxScaler(feature_range = (1, 100))
    scaled_weekly = weekly_scaler.fit_transform(weekly_vals).round().astype(int)
    
    df_weekly_scaled = pd.DataFrame(
        data = scaled_weekly,
        index = df_resampled.index,
        columns = [f"scaled_weekly_{column}"]
    )
    
    if plot:
        fig, ax = plt.subplots(figsize = (8, 6), nrows = 2, ncols = 1)
        df[column].plot(ax = ax[0])
        ax[0].set_title(f"Daily Original Values: {column}")
        
        df[f"scaled_{column}"].plot(ax = ax[1])
        ax[1].set_title(f"Daily Scaled Values: {column}")
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(figsize = (8, 6), nrows = 2, ncols = 1)
        df_resampled.plot(ax = ax[0])
        ax[0].set_title(f"Weekly Resampled: {column}")
        df_weekly_scaled.plot(ax = ax[1])
        ax[1].set_title(f"Weekly Scaled: {column}")
        plt.tight_layout()
        plt.show()
    return df, df_weekly_scaled


def plot_daily_and_weekly(keyword, start_date, end_date = None, periods = None, period_unit = "weeks", geo = "US"):
    # Grabbing data from Google Trends
    df = get_pytrends_data(
        keywords = keyword,
        start_date = start_date,
        end_date = end_date,
        periods = periods,
        period_unit = period_unit,
        geo = geo
    )
    df.index = pd.to_datetime(df.index)
    
    # Daily lineplot
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=df.index,
        y=df["interest"],
        mode="lines+markers",
        name="Daily",
        hovertemplate="%{x|%Y-%m-%d}: %{y}"
    ))
    fig_daily.update_layout(
        title = f"Daily Search Interest for '{keyword}'",
        xaxis_title = "Date",
        yaxis_title = "Search Interest",
        hovermode = "x unified",
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        xaxis = dict(showgrid = True, gridcolor = "lightgray"),
        yaxis = dict(showgrid = True, gridcolor = "lightgray")
    )
    st.plotly_chart(fig_daily, use_container_width = True)
    
    # Preprocessing into weekly data
    _, df_weekly = resample_minmax_days_to_weeks(df, column = "interest")
    
    # Weekly lineplot
    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Scatter(
        x = df_weekly.index,
        y = df_weekly["scaled_weekly_interest"],
        mode = "lines+markers",
        name = "Weekly (scaled)",
        hovertemplate = "%{x|%Y-%m-%d}: %{y}"
    ))
    fig_weekly.update_layout(
        title = f"Weekly Search Interest for '{keyword}' (scaled)",
        xaxis_title = "Date",
        yaxis_title = "Search Interest",
        hovermode = "x unified",
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        xaxis = dict(showgrid = True, gridcolor = "lightgray"),
        yaxis = dict(showgrid = True, gridcolor = "lightgray")
    )
    st.plotly_chart(fig_weekly, use_container_width = True)


def bdm_plots(idata, T, observed):
    var_names = ["p", "q", "m"]
    
    # Viewing model summary
    az.summary(data = idata, var_names = ["p", "q", "m"])
    
    # Viewing model"s trace plot
    _ = az.plot_trace(
        data = idata,
        var_names = var_names,
        compact = True,
        backend_kwargs = {"figsize": (12, 9), "layout": "constrained"}
    )
    plt.gcf().suptitle("Model Trace", fontsize = 16)
    # Plotting Posterior Predictive vs Observed Data
    fig, ax = plt.subplots(figsize = (8, 4), layout = "constrained")
    idata["posterior_predictive"]["y"].pipe(plot_curve, {"T"}, axes = ax)
    ax.plot(T, observed, color = "black")
    plt.gcf().suptitle("Posterior Predictive vs Observed Data", fontsize = 16)
    
    # Plotting Cumulative Posterior Predictive vs Observed Data
    fig, ax = plt.subplots(figsize = (8, 4), layout = "constrained")
    idata["posterior_predictive"]["y"].cumsum(dim = "T").pipe(plot_curve, {"T"}, axes = ax)
    ax.plot(T, observed.cumsum(), color = "black")
    plt.gcf().suptitle("Cumulative Posterior Predictive vs Observed Data", fontsize = 16)
    plt.tight_layout()

# Defining prior distributions
def create_bdm_priors(scale = 100000, m_sigma = 0.1, lower_p = 0.01, lower_q = 0.01, upper_p = 0.99, upper_q = 0.99,
                    like_sigma = 1_000_000):
    return {
        "m": Scaled(Prior("Gamma", mu = 1, sigma = m_sigma, dims = "search_interest"), factor = scale),
        "p": Prior("Beta", dims = "search_interest").constrain(lower = lower_p, upper = upper_p),
        "q": Prior("Beta", dims = "search_interest").constrain(lower = lower_q, upper = upper_q),
        "likelihood": Prior("InverseGamma", sigma = like_sigma, dims = "search_interest"),
    }

if st.button("Run Bass Model"):
    try:
        plot_daily_and_weekly(keyword, start_date, end_date, periods, period_unit, geo)
        # Fetch data
        df = get_pytrends_data(
            keywords = keyword,
            start_date = start_date,
            end_date = end_date if not use_periods else None,
            periods = periods if use_periods else None,
            period_unit = period_unit if use_periods else None,
            geo = geo
        )
        
        # Preprocess to weekly scaled data
        _, df_weekly = resample_minmax_days_to_weeks(df, column = "interest")
        df_weekly["scaled_weekly_interest"] = df_weekly["scaled_weekly_interest"].astype(int)
        
        # Time variable
        T = np.arange(len(df_weekly))
        
        # Set priors
        priors = create_bdm_priors(
            scale = df_weekly["scaled_weekly_interest"].sum(),
            like_sigma = 0.025 * df_weekly["scaled_weekly_interest"].sum()
        )
        
        # Create Bass model
        bdm = create_bass_model(
            t = T,
            observed = df_weekly["scaled_weekly_interest"].values.reshape(-1, 1),
            coords = {"T": T, "search_interest": np.array(["scaled_weekly_interest"])},
            priors = priors
        )
        
        # Sample posterior and posterior predictive
        with bdm:
            idata = pm.sample(nuts_sampler = "nutpie", compile_kwargs = {"mode": "numba"}, random_seed = 2025)
            idata.extend(pm.sample_posterior_predictive(idata, model = bdm, extend_inferencedata = True, random_seed = 2025))
            
        # Summary
        summary_df = az.summary(idata, var_names = ["m", "p", "q", "peak"])
        summary_df.index = ["Market Size (m)", "Coefficient of Innovation (p)", "Coefficient of Imitation (q)", "Time of Peak"]
        st.write("Posterior summary:")
        st.dataframe(summary_df.iloc[:, :4])
        
        # Plots
        bdm_plots(idata, T, df_weekly["scaled_weekly_interest"].values.reshape(-1, 1))
        # Show all figures in Streamlit
        for fig_num in plt.get_fignums():
            st.pyplot(plt.figure(fig_num))
            
    except Exception as e:
        st.error(f"Error: {e}")
