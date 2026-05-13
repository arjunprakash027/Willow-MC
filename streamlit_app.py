import streamlit as st
import time
import sys
import os
from src.predictor import WinPredictor
from src.monitor_utils import (
    MatchState, 
    RUN_MODEL_IPL, WICKET_MODEL_IPL,
    RUN_MODEL_T20, WICKET_MODEL_T20,
    RUN_MODEL_ODI, WICKET_MODEL_ODI
)

st.set_page_config(page_title="Willow-MC Live", page_icon="🏏")

st.title("WillowMC Live Monitor")

match_id = st.sidebar.text_input("Cricbuzz Match ID", value="100001")
match_format = st.sidebar.selectbox("Match Format", ["IPL", "T20 International", "ODI"])
n_sims = st.sidebar.number_input("Simulations", value=5000, step=1000)

if st.sidebar.button("Start Tracking"):
    if match_format == "IPL":
        rm, wm = RUN_MODEL_IPL, WICKET_MODEL_IPL
    elif match_format == "T20 International":
        rm, wm = RUN_MODEL_T20, WICKET_MODEL_T20
    else:
        rm, wm = RUN_MODEL_ODI, WICKET_MODEL_ODI

    predictor = WinPredictor(run_model_path=rm, wicket_model_path=wm)
    match_state = MatchState(match_id)
    placeholder = st.empty()

    while True:
        match_state.update()
        
        with placeholder.container():
            if not match_state.batting_team:
                st.info(f"Connecting to Match {match_id}...")
            else:
                st.header(f"{match_state.batting_team} vs {match_state.bowling_team}")
                
                cname1, cname2 = st.columns(2)

                with cname1:
                    st.write(f"Batting : {match_state.batting_team}")
                with cname2:
                    st.write(f"Bowling : {match_state.bowling_team}")
                
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
                        st.metric(
                            label=f"Win Probability for {match_state.batting_team}", 
                            value=f"{round(win_prob * 100, 1)}%"
                        )

                    else:
                        proj_score = predictor.simulate(state, n_sims=n_sims)
                        chase_state = {"score": 0, "wickets": 0, "balls": 0, "target": int(proj_score) + 1}
                        chase_win_prob = predictor.simulate(chase_state, n_sims=n_sims)
                        win_prob = 1.0 - chase_win_prob
                        
                        col_p1, col_p2 = st.columns(2)
                        col_p1.metric("Projected Score", int(proj_score))

                        col_p2.metric(
                            label=f"Win Probability for {match_state.batting_team}", 
                            value=f"{round(win_prob * 100, 1)}%"
                        )


                st.caption(f"Last updated: {match_state.last_updated}")
        
        time.sleep(10)
