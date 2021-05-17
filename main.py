from accuracy_fairness_dilemma.afd import afd
from ponder.home import ponder_home
import streamlit as st

PAGES = {
    "Home" : ponder_home,
    "Accuracy-Fairness Dilemma" : afd
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()
