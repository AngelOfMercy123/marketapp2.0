from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import requests

from streamlit_autorefresh import st_autorefresh

from market_ai.config import Paths
from market_ai.logging_utils import setup_logging
from market_ai.storage import ensure_dirs
from market_ai.universe import (
    DEFAULT_UNIVERSE_COLUMNS,
    load_universe,
    save_universe,
    add_or_update_company,
    remove_company,
)
from market_ai.watchlist import load_watchlist, save_watchlist, add_to_watchlist, remove_from_watchlist
from market_ai.data_sources import update_prices_for_ticker
from market_ai.features import build_ohlcv_features
from market_ai.model import train_classifier, predict_proba_up, load_model, save_model, ModelBundle
from market_ai.scoring import score_universe
from market_ai.backtest import walk_forward_backtest
from market_ai.news import fetch_google_news_rss
from market_ai.tracking import load_predictions, log_prediction, score_matured_predictions
from market_ai.ticker_sources import fetch_sp500_constituents
from market_ai.backfill import run_backfill


st.set_page_config(page_title='MarketApp (Modular AI)', layout='wide')

paths = Paths()
ensure_dirs(paths.data_dir, paths.intraday_dir, paths.logs_dir, paths.cache_dir)
logger = setup_logging(paths.logs_dir)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header('Controls')

refresh_seconds = st.sidebar.number_input('Auto-refresh (seconds, 0=off)', min_value=0, max_value=3600, value=0, step=5)
if refresh_seconds and refresh_seconds > 0:
    st_autorefresh(interval=int(refresh_seconds * 1000), key='autorefresh')

horizon_days = st.sidebar.number_input('Prediction horizon (days)', min_value=1, max_value=30, value=5, step=1)
min_history_days = st.sidebar.number_input('Min history (days) to model', min_value=60, max_value=3000, value=500, step=50)

structural_weight = st.sidebar.slider('Structural weight', 0.0, 1.0, 0.60, 0.05)
demographic_weight = 1.0 - structural_weight

max_watch = st.sidebar.number_input('Max tickers to watch (auto-pick)', min_value=1, max_value=200, value=20, step=1)

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) MarketApp/1.0'})

# -------------------------
# Load data
# -------------------------
universe = load_universe(paths.universe_csv)
watchlist = load_watchlist(paths.watchlist_csv)

st.title('MarketApp — modular scoring + ML + tracking')


def _status_line(msg: str, ok: bool = True):
    (st.success if ok else st.error)(msg)


tabs = st.tabs(['Dashboard', 'Universe', 'Watchlist', 'Model & Backtest', 'News', 'Predictions Log'])

# -------------------------
# Dashboard
# -------------------------
with tabs[0]:
    st.subheader('Top scored companies')

    scored = score_universe(
        universe,
        structural_weight=float(structural_weight),
        demographic_weight=float(demographic_weight),
    )

    # If we have a trained model, attach latest probability for each ticker
    bundle = load_model(paths.model_path)
    if bundle is not None:
        probs = []
        for t in scored['ticker'].tolist()[:min(60, len(scored))]:
            res = update_prices_for_ticker(t, paths.data_dir, session=session, max_age_hours=12)
            df = res[0]
            if df is None or len(df) < min_history_days:
                probs.append(np.nan)
                continue
            X, y, feat_cols = build_ohlcv_features(df, horizon_days=int(horizon_days))
            if X.empty:
                probs.append(np.nan)
                continue
            proba = predict_proba_up(bundle, X.tail(1))
            probs.append(float(proba) if proba is not None else np.nan)
        scored = scored.head(len(probs)).copy()
        scored['ml_prob_up_latest'] = probs

    st.dataframe(scored.head(50), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Auto-pick watchlist')
        if st.button('Pick top scores → watchlist'):
            pick = scored.dropna(subset=['final_score']).head(int(max_watch))['ticker'].tolist()
            wl = watchlist.copy()
            for t in pick:
                wl = add_to_watchlist(wl, t, notes='auto-pick')
            save_watchlist(wl, paths.watchlist_csv)
            _status_line(f'Added {len(pick)} tickers to watchlist.', True)

    with col2:
        st.subheader('Model status')
        if bundle is None:
            st.warning('No saved model yet. Train one in the Model tab.')
        else:
            st.write({'model_tag': bundle.meta.get('model_tag'), 'trained_utc': bundle.meta.get('trained_utc')})

# -------------------------
# Universe
# -------------------------
with tabs[1]:
    st.subheader('Universe')
    st.caption('This CSV is your "thesis" inputs: utility, policy, leadership, workforce exposure + demographics risk fields.')

    st.dataframe(universe, use_container_width=True)

    with st.expander('Add / update company'):
        cols = st.columns(3)
        ticker = cols[0].text_input('Ticker (e.g., AAPL)').strip().upper()
        country_state = cols[1].text_input('Country/State (optional)').strip()
        notes = cols[2].text_input('Notes (optional)').strip()

        # structural
        c1, c2, c3 = st.columns(3)
        utility = c1.number_input('utility_score (0-10)', 0.0, 10.0, 5.0, 0.5)
        policy = c2.number_input('policy_score (0-10)', 0.0, 10.0, 5.0, 0.5)
        leadership = c3.number_input('leadership_score (0-10)', 0.0, 10.0, 5.0, 0.5)

        c4, c5, c6 = st.columns(3)
        labor_exposure = c4.number_input('labor_exposure (0-10)', 0.0, 10.0, 5.0, 0.5)
        jan_score = c5.number_input('jan_score (0-10)', 0.0, 10.0, 5.0, 0.5)
        workforce_stability = c6.number_input('workforce_stability (0-10)', 0.0, 10.0, 5.0, 0.5)

        # demographics
        d1, d2, d3 = st.columns(3)
        avg_age = d1.number_input('avg_workforce_age', 16.0, 80.0, 40.0, 1.0)
        retirement_age = d2.number_input('retirement_age', 40.0, 80.0, 65.0, 1.0)
        pct_physical = d3.number_input('pct_physical_labor (0-100)', 0.0, 100.0, 20.0, 1.0)

        d4, d5, d6 = st.columns(3)
        pct_female = d4.number_input('pct_female (0-100)', 0.0, 100.0, 45.0, 1.0)
        pct_immigrant = d5.number_input('pct_immigrant (0-100)', 0.0, 100.0, 10.0, 1.0)

        if st.button('Save company'):
            if not ticker:
                _status_line('Ticker is required.', False)
            else:
                universe2 = add_or_update_company(
                    universe,
                    {
                        'ticker': ticker,
                        'country_state': country_state,
                        'notes': notes,
                        'utility_score': utility,
                        'policy_score': policy,
                        'leadership_score': leadership,
                        'labor_exposure': labor_exposure,
                        'jan_score': jan_score,
                        'avg_workforce_age': avg_age,
                        'pct_physical_labor': pct_physical,
                        'pct_female': pct_female,
                        'pct_immigrant': pct_immigrant,
                        'retirement_age': retirement_age,
                        'workforce_stability': workforce_stability,
                    },
                )
                save_universe(universe2, paths.universe_csv)
                _status_line(f'Saved {ticker}.')

    with st.expander('Remove company'):
        t = st.selectbox('Ticker', options=universe['ticker'].tolist() if not universe.empty else [])
        if st.button('Remove') and t:
            universe2 = remove_company(universe, t)
            save_universe(universe2, paths.universe_csv)
            _status_line(f'Removed {t}.')

    with st.expander('Auto-add S&P 500 tickers (free source)'):
        if st.button('Fetch S&P 500 and merge (adds missing tickers only)'):
            try:
                tickers = fetch_sp500_constituents(session)
                added = 0
                uni = universe.copy()
                for t in tickers:
                    if t not in set(uni['ticker'].astype(str).str.upper()):
                        uni = add_or_update_company(uni, {'ticker': t})
                        added += 1
                save_universe(uni, paths.universe_csv)
                _status_line(f'Fetched {len(tickers)} tickers, added {added} new ones.')
            except Exception as e:
                _status_line(f'Failed to fetch tickers: {e}', False)

# -------------------------
# Watchlist
# -------------------------
with tabs[2]:
    st.subheader('Watchlist')
    st.dataframe(watchlist, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        add_t = st.text_input('Add ticker to watchlist').strip().upper()
        if st.button('Add to watchlist') and add_t:
            wl = add_to_watchlist(watchlist, add_t, notes='manual')
            save_watchlist(wl, paths.watchlist_csv)
            _status_line(f'Added {add_t}.')

    with colB:
        if not watchlist.empty:
            rem_t = st.selectbox('Remove ticker', watchlist['ticker'].tolist())
            if st.button('Remove from watchlist') and rem_t:
                wl = remove_from_watchlist(watchlist, rem_t)
                save_watchlist(wl, paths.watchlist_csv)
                _status_line(f'Removed {rem_t}.')

    st.divider()
    st.subheader('Update prices (daily)')
    tickers = watchlist['ticker'].tolist() if not watchlist.empty else []
    if st.button('Update all tickers now'):
        ok = 0
        for t in tickers:
            df, meta = update_prices_for_ticker(t, paths.data_dir, session=session, max_age_hours=0)
            if df is not None:
                ok += 1
        _status_line(f'Updated {ok}/{len(tickers)} tickers.')

    st.divider()
    st.subheader('Latest signals')
    bundle = load_model(paths.model_path)
    if bundle is None:
        st.info('Train a model first.')
    else:
        rows = []
        for t in tickers[:60]:
            df, _ = update_prices_for_ticker(t, paths.data_dir, session=session, max_age_hours=12)
            if df is None or len(df) < min_history_days:
                continue
            X, y, feat_cols = build_ohlcv_features(df, horizon_days=int(horizon_days))
            if X.empty:
                continue
            proba = predict_proba_up(bundle, X.tail(1))
            pred = int(proba >= 0.5) if proba is not None else None
            close = float(df['Close'].iloc[-1])
            rows.append({'ticker': t, 'close': close, 'prob_up': proba, 'pred_up': pred})

            # log prediction
            if proba is not None:
                log_prediction(paths.pred_log_csv, ticker=t, horizon_days=int(horizon_days), pred_prob_up=float(proba),
                               close_at_pred=close, model_tag=bundle.meta.get('model_tag', 'model'))

        if rows:
            sig = pd.DataFrame(rows).sort_values('prob_up', ascending=False)
            st.dataframe(sig, use_container_width=True)
        else:
            st.write('No signals yet (need price history).')

# -------------------------
# Model & Backtest
# -------------------------
with tabs[3]:
    st.subheader('Train model (directional classifier)')
    st.caption('Uses OHLCV technical features. Your thesis scores remain separate and can be blended in later.')

    train_tickers = st.multiselect('Tickers to train on (from universe)', options=universe['ticker'].tolist(), default=watchlist['ticker'].tolist()[:10] if not watchlist.empty else [])

    if st.button('Train / retrain'):
        allX = []
        ally = []
        feat_cols = None
        used = []
        for t in train_tickers:
            df, _ = update_prices_for_ticker(t, paths.data_dir, session=session, max_age_hours=12)
            if df is None or len(df) < min_history_days:
                continue
            X, y, cols = build_ohlcv_features(df, horizon_days=int(horizon_days))
            if X.empty:
                continue
            if feat_cols is None:
                feat_cols = cols
            else:
                X = X[feat_cols]
            allX.append(X)
            ally.append(y)
            used.append(t)

        if not allX:
            _status_line('No training data. Update prices + ensure enough history.', False)
        else:
            X = pd.concat(allX).sort_index()
            y = pd.concat(ally).loc[X.index]
            bundle = train_classifier(X, y)
            save_model(bundle, paths.model_path)
            _status_line(f'Trained model on {len(used)} tickers, rows={len(X)}. Saved to {paths.model_path}.')

    st.divider()
    st.subheader('Quick walk-forward backtest (single ticker)')
    t = st.selectbox('Ticker', options=watchlist['ticker'].tolist() if not watchlist.empty else universe['ticker'].tolist())
    if st.button('Run backtest') and t:
        df, _ = update_prices_for_ticker(t, paths.data_dir, session=session, max_age_hours=12)
        if df is None or len(df) < min_history_days:
            _status_line('Not enough data.', False)
        else:
            X, y, cols = build_ohlcv_features(df, horizon_days=int(horizon_days))
            res = walk_forward_backtest(X, y)
            st.write(res.__dict__)
            st.dataframe(res.sample.tail(20), use_container_width=True)

# -------------------------
# News
# -------------------------
with tabs[4]:
    st.subheader('Live news (free RSS)')

    q = st.text_input('Query (ticker or topic)', value='SPY')
    max_items = st.slider('Max items', 3, 30, 10, 1)

    if st.button('Fetch news'):
        try:
            items = fetch_google_news_rss(q, session=session, max_items=int(max_items))
            for it in items:
                st.markdown(f"- [{it.title}]({it.link}) — *{it.source}* ({it.pub_date})")
        except Exception as e:
            _status_line(f'News fetch failed: {e}', False)

# -------------------------
# Predictions log
# -------------------------
with tabs[5]:
    st.subheader('Prediction tracking')
    st.caption('We log each model signal with the close price at the time. Once the horizon date passes, we score it.')

    if st.button('Score matured predictions now'):
        scored_count, msg = score_matured_predictions(paths.pred_log_csv, paths.data_dir)
        _status_line(f'{msg} (scored {scored_count}).', True)

    preds = load_predictions(paths.pred_log_csv)
    if preds.empty:
        st.info('No predictions logged yet (generate signals in Watchlist tab).')
    else:
        # summary
        done = preds.dropna(subset=['actual_class'])
        if not done.empty:
            acc = (done['pred_class'] == done['actual_class']).mean()
            st.metric('Accuracy (matured preds)', f"{acc:.2%}")
            st.metric('Matured predictions', int(len(done)))
        st.dataframe(preds.sort_values('timestamp_utc', ascending=False).head(400), use_container_width=True)

    st.divider()
    st.subheader('Backfill Training Data')
    st.caption('Generate historical prediction data to create training examples. This looks back N weeks and creates synthetic predictions with known outcomes.')

    with st.expander('Backfill Settings', expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            backfill_weeks = st.number_input(
                'Weeks back to generate',
                min_value=1,
                max_value=104,
                value=16,
                step=1,
                help='How many weeks of historical data to generate (e.g., 16 = ~4 months)'
            )
            backfill_max_tickers = st.number_input(
                'Max tickers to process',
                min_value=1,
                max_value=100,
                value=12,
                step=1,
                help='Limit number of tickers to avoid long processing times'
            )

        with col2:
            backfill_baseline = st.text_input(
                'Baseline ticker',
                value='SPY',
                help='Ticker to use as baseline for outperformance calculation'
            ).strip().upper()
            backfill_source_tag = st.text_input(
                'Source tag',
                value='retro',
                help='Tag to identify generated rows (e.g., "retro", "backfill")'
            ).strip()

        with col3:
            backfill_backup = st.checkbox(
                'Create backup first',
                value=True,
                help='Create a .bak copy of predictions log before modifying'
            )

        st.info(f'This will process up to {backfill_max_tickers} tickers from your watchlist/universe, going back {backfill_weeks} weeks.')

        if st.button('🔄 Run Backfill', type='primary'):
            with st.spinner('Backfilling training data... This may take a minute.'):
                try:
                    stats = run_backfill(
                        pred_log_path=str(paths.pred_log_csv),
                        cache_dir=str(paths.data_dir),
                        baseline=backfill_baseline,
                        weeks_back=int(backfill_weeks),
                        max_tickers=int(backfill_max_tickers),
                        watchlist_path=str(paths.watchlist_csv),
                        universe_path=str(paths.universe_csv),
                        source_tag=backfill_source_tag,
                        create_backup=backfill_backup,
                    )

                    st.success('✅ Backfill complete!')
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric('Outcomes updated', stats['updated'])
                    col_b.metric('New rows added', stats['added'])
                    col_c.metric('Total rows', stats['total'])
                    col_d.metric('Known outcomes', stats['known'])

                    if stats['added'] > 0:
                        st.info(f'Added {stats["added"]} historical prediction rows. These can now be used for training meta-models or analyzing prediction accuracy over time.')

                except Exception as e:
                    st.error(f'Backfill failed: {e}')
                    import traceback
                    st.code(traceback.format_exc())


st.caption('Tip: keep universe.csv updated with your thesis inputs; the model learns from prices; the tracker shows if it helps.')
