import streamlit as st
import time
import sys
import os
from src.predictor import WinPredictor
from minimal_monitor import MatchState, RUN_MODEL, WICKET_MODEL

st.set_page_config(page_title="Willow-MC Live", page_icon="🏏")

st.title("🏏 Willow-MC Live Monitor")

match_id = st.sidebar.text_input("Cricbuzz Match ID", value="100001")
n_sims = st.sidebar.number_input("Simulations", value=5000, step=1000)

if st.sidebar.button("Start Tracking"):
    predictor = WinPredictor(run_model_path=RUN_MODEL, wicket_model_path=WICKET_MODEL)
    match_state = MatchState(match_id)
    placeholder = st.empty()

    while True:
        match_state.update()
        
        with placeholder.container():
            if not match_state.batting_team:
                st.info(f"Connecting to Match {match_id}...")
            else:
                st.header(f"{match_state.batting_team} vs {match_state.bowling_team}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Score", f"{match_state.score}/{match_state.wickets}")
                c2.metric("Overs", match_state.overs)
                c3.metric("CRR", match_state.crr)

                if match_state.target:
                    st.subheader(f"Target: {match_state.target}")

                state = {
                    "score": match_state.score,
                    "wickets": match_state.wickets,
                    "balls": predictor.get_balls_bowled(match_state.overs),
                    "target": match_state.target
                }

                with st.spinner("Simulating..."):
                    if match_state.target:
                        win_prob = predictor.simulate(state, n_sims=n_sims)
                        st.metric("Win Probability", f"{round(win_prob * 100, 1)}%")
                    else:
                        proj_score = predictor.simulate(state, n_sims=n_sims)
                        chase_state = {"score": 0, "wickets": 0, "balls": 0, "target": int(proj_score) + 1}
                        chase_win_prob = predictor.simulate(chase_state, n_sims=n_sims)
                        win_prob = 1.0 - chase_win_prob
                        
                        col_p1, col_p2 = st.columns(2)
                        col_p1.metric("Projected Score", int(proj_score))
                        col_p2.metric("Win Probability", f"{round(win_prob * 100, 1)}%")

                st.caption(f"Last updated: {match_state.last_updated}")
        
        time.sleep(10)
