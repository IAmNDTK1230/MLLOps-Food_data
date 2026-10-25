"""A small, centralized visual layer for the Streamlit demo."""

import streamlit as st

CSS = """
<style>
:root {color-scheme: dark;}
.stApp {background: #0E1117; color: #F1F5F9;}
[data-testid="stSidebar"] {background: #151B24; border-right: 1px solid #2A3442;}
[data-testid="stSidebar"] .stRadio > label {display: none;}
[data-testid="stSidebar"] [role="radiogroup"] {gap: .36rem;}
[data-testid="stSidebar"] [role="radio"] {padding: .48rem .7rem; border-radius: .65rem;}
[data-testid="stSidebar"] [role="radio"]:has(input:checked) {background: #253857;}
.block-container {max-width: 1380px; padding-top: 1.8rem; padding-bottom: 3rem;}
h1,h2,h3 {letter-spacing: -.025em; color: #F1F5F9;}
h1 {font-size: 2.15rem !important; font-weight: 750 !important;}
h2 {font-size: 1.35rem !important; margin-top: 1.25rem !important;}
h3 {font-size: 1.08rem !important;}
[data-testid="stVerticalBlockBorderWrapper"] {background: #161B22; border-color: #2A3442 !important; border-radius: .85rem;}
[data-testid="stMetric"] {background: #1A202C; border: 1px solid #2A3442; border-radius: .85rem; padding: 1rem; min-height: 116px;}
[data-testid="stMetricLabel"] {color: #9AA4B2;}
[data-testid="stMetricValue"] {font-weight: 720; letter-spacing: -.03em;}
[data-testid="stDataFrame"] {border: 1px solid #2A3442; border-radius: .7rem; overflow: hidden;}
.foody-hero {padding: 1.15rem 1.3rem; background: linear-gradient(120deg,#1B2940,#161B22 65%); border: 1px solid #32425A; border-radius: 1rem; margin-bottom: 1.25rem;}
.foody-eyebrow {font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; color: #82ABF2; font-weight: 700; margin-bottom: .3rem;}
.foody-hero h1 {margin: 0 0 .25rem 0;}
.foody-muted {color: #9AA4B2; font-size: .9rem; margin: .2rem 0;}
.foody-badge {display: inline-block; padding: .25rem .57rem; margin: .25rem .3rem .05rem 0; border: 1px solid #3B5274; background: #22314A; color: #BDD7FF; border-radius: 999px; font-size: .76rem; font-weight: 600;}
.foody-badge.good {background: #15352B; border-color: #286A50; color: #93E5B8;}
.foody-badge.warn {background: #382D1D; border-color: #77602B; color: #F5D38A;}
.foody-badge.muted {background: #252C36; border-color: #3E4855; color: #B2BDCA;}
.foody-card {background: #161B22; border: 1px solid #2A3442; border-radius: .85rem; padding: 1rem 1.15rem; margin: .35rem 0 .8rem;}
.foody-card h3 {margin: 0 0 .45rem;}
.foody-card p {margin: .25rem 0; line-height: 1.55;}
.foody-empty {background: #1A202C; border: 1px dashed #435064; border-radius: .85rem; padding: 1rem 1.2rem; color: #CBD5E1;}
.foody-empty strong {display: block; color: #F1F5F9; margin-bottom: .2rem;}
.foody-answer {border-left: 3px solid #4F8EF7;}
.foody-section-note {font-size: .84rem; color: #9AA4B2; margin-top: -.35rem; margin-bottom: .6rem;}
div.stButton > button {border-radius: .55rem; border-color: #3A4E69;}
div.stButton > button[kind="primary"] {background: #4F8EF7; border-color: #4F8EF7; color: #0E1117;}
</style>
"""


def apply_styles():
    st.markdown(CSS, unsafe_allow_html=True)
