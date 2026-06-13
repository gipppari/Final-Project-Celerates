import json
import importlib.util
import time
from html import escape
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


DATASET_URL = "https://docs.google.com/spreadsheets/d/1aX6GynZaDd3leFEWfU16wnpxP-PjgyCiaetOkHvCooQ/export?format=csv"
REFERENCE_DATE = pd.Timestamp("2024-12-31")
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEGMENT_COLORS = {
    "High Value Customers": "#10b981",
    "Loyal Customers": "#3b82f6",
    "Frequent Buyers": "#8b5cf6",
    "Potential Loyalists": "#06b6d4",
    "New Customers": "#f59e0b",
    "At Risk Customers": "#f97316",
    "Lost Customers": "#ef4444",
    "Regular Customers": "#64748b",
}
SEGMENT_STRATEGY = {
    "High Value Customers": (
        "Top-spending recent customers.",
        "Protect this cohort with VIP access, personal offers, and premium service follow-ups.",
    ),
    "Loyal Customers": (
        "Frequent buyers with strong recent activity.",
        "Use milestone rewards, referral bonuses, and early product previews.",
    ),
    "Frequent Buyers": (
        "High checkout frequency with moderate recency or value.",
        "Increase basket size through bundles, volume discounts, and companion items.",
    ),
    "Potential Loyalists": (
        "Recent buyers who are close to becoming repeat customers.",
        "Send onboarding sequences, review vouchers, and second-purchase incentives.",
    ),
    "New Customers": (
        "Fresh buyers with limited purchase history.",
        "Focus on welcome campaigns, education, and low-friction repeat purchase prompts.",
    ),
    "At Risk Customers": (
        "Past valuable customers with declining recency.",
        "Launch win-back campaigns with personalized discounts and friction surveys.",
    ),
    "Lost Customers": (
        "Low-recency, low-frequency buyers.",
        "Keep acquisition cost low with seasonal newsletters and broad reactivation offers.",
    ),
    "Regular Customers": (
        "Middle-tier customers with stable but unexceptional behavior.",
        "Maintain newsletter cadence and target them with category-based promotions.",
    ),
}


st.set_page_config(
    page_title="Customer Behavior Analytics & Recommendation System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #0f172a;
            --muted: #64748b;
            --soft: #f1f5f9;
            --panel: #ffffff;
            --line: #e2e8f0;
            --blue: #0ea5e9;
            --violet: #4f46e5;
            --nav: #0f172a;
        }
        .stApp {
            background: #eef3f8;
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(238, 243, 248, 0.86);
            backdrop-filter: blur(10px);
        }
        [data-testid="stSidebar"] {
            background: var(--nav);
        }
        [data-testid="stSidebar"] * {
            color: #dbeafe;
        }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: #94a3b8 !important;
            font-size: 0.78rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #24324d;
        }
        .side-nav {
            display: flex;
            flex-direction: column;
            gap: 0.58rem;
            margin-bottom: 1.25rem;
        }
        .side-nav a {
            text-decoration: none !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 54px;
            justify-content: flex-start;
            text-align: left;
            border-radius: 12px;
            border: 1px solid transparent;
            background: transparent;
            color: #cbd5e1;
            padding: 0.65rem 0.78rem;
            box-shadow: none;
            font-size: 0.82rem;
            font-weight: 750;
        }
        [data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] p {
            margin: 0;
            line-height: 1.12;
        }
        .side-btn-title {
            color: #cbd5e1;
            font-size: 0.86rem;
            font-weight: 850;
        }
        .side-btn-subtitle {
            color: #6f86a7;
            font-size: 0.66rem;
            font-weight: 500;
            margin-top: 0.2rem;
        }
        [data-testid="stSidebar"] .stButton > button:hover .side-btn-title,
        [data-testid="stSidebar"] .stButton > button:hover .side-btn-subtitle {
            color: #ffffff;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: #14213a;
            border-color: #273650;
            color: #ffffff;
        }
        [data-testid="stSidebar"] .stButton > button:active,
        [data-testid="stSidebar"] .stButton > button:focus {
            border-color: #4f35e8;
            color: #ffffff;
        }
        .side-nav-item {
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: 0.72rem;
            align-items: center;
            min-height: 54px;
            border-radius: 12px;
            padding: 0.65rem 0.78rem;
            color: #cbd5e1;
            border: 1px solid transparent;
            transition: background 150ms ease, border 150ms ease, transform 150ms ease;
        }
        .side-nav-item:hover {
            background: #14213a;
            border-color: #273650;
            transform: translateX(2px);
        }
        .side-nav-item.active {
            background: linear-gradient(135deg, #603cff, #4f35e8);
            border-color: #735cf7;
            box-shadow: 0 10px 26px rgba(79, 53, 232, 0.22);
        }
        .side-nav-icon {
            width: 28px;
            height: 28px;
            border-radius: 9px;
            display: grid;
            place-items: center;
            color: #93a9c9;
            background: transparent;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.8rem;
            font-weight: 900;
        }
        .side-nav-icon svg {
            width: 18px;
            height: 18px;
            stroke: currentColor;
            stroke-width: 1.9;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .side-nav-item.active .side-nav-icon {
            background: rgba(255, 255, 255, 0.14);
            color: #ffffff;
        }
        .side-nav-title {
            color: #cbd5e1;
            font-size: 0.86rem;
            font-weight: 850;
            line-height: 1.15;
        }
        .side-nav-subtitle {
            color: #6f86a7;
            font-size: 0.66rem;
            line-height: 1.15;
            margin-top: 0.18rem;
        }
        .side-nav-item.active .side-nav-title,
        .side-nav-item.active .side-nav-subtitle {
            color: #ffffff;
        }
        .block-container {
            padding-top: 4rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }
        .page-head {
            padding-top: 0.35rem;
        }
        .dash-eyebrow {
            color: #49627f;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            line-height: 1.35;
        }
        .dash-title {
            color: #020617;
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1.1;
            margin: 0.25rem 0 1.25rem;
        }
        .dashboard-card {
            background: #ffffff;
            border: 1px solid rgba(226, 232, 240, 0.9);
            border-radius: 13px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            padding: 1.25rem;
        }
        .metric-card {
            min-height: 112px;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .metric-value {
            color: #111827;
            font-size: 1.58rem;
            font-weight: 900;
            margin-top: 0.25rem;
            letter-spacing: 0;
        }
        .metric-pill {
            display: inline-block;
            margin-top: 0.35rem;
            padding: 0.2rem 0.5rem;
            border-radius: 8px;
            background: #ecfdf5;
            color: #059669;
            font-size: 0.72rem;
            font-weight: 800;
        }
        .metric-icon {
            color: #52617a;
            width: 34px;
            height: 34px;
            display: grid;
            place-items: center;
            align-self: center;
        }
        .metric-icon svg {
            width: 28px;
            height: 28px;
            stroke: #52617a;
            stroke-width: 1.8;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .card-title {
            color: #111827;
            font-weight: 900;
            font-size: 1rem;
            margin: 0;
        }
        .card-subtitle {
            color: #64748b;
            font-size: 0.83rem;
            margin: 0.15rem 0 0.75rem;
        }
        .product-row {
            display: grid;
            grid-template-columns: 34px 1fr auto;
            gap: 0.75rem;
            align-items: center;
            padding: 0.52rem 0;
            border-bottom: 1px solid #eef2f7;
        }
        .product-rank {
            color: #334155;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.78rem;
            font-weight: 900;
        }
        .product-name {
            color: #1f2937;
            font-size: 0.82rem;
            font-weight: 850;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .product-meta {
            color: #64748b;
            font-size: 0.68rem;
            text-align: right;
        }
        .bar-track {
            grid-column: 2 / 4;
            height: 5px;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            background: #0ea5e9;
            border-radius: 999px;
        }
        .section-gap {
            height: 1.15rem;
        }
        .side-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.35rem 0 1.25rem;
            border-bottom: 1px solid #1f2b42;
            margin-bottom: 1.4rem;
        }
        .side-logo {
            width: 32px;
            height: 32px;
            border-radius: 9px;
            display: grid;
            place-items: center;
            background: #17233c;
            color: #93c5fd;
            border: 1px solid #334155;
            font-size: 0.9rem;
        }
        .side-card {
            background: #111c33;
            border: 1px solid #24324d;
            border-radius: 13px;
            padding: 1rem;
            margin-top: 1.5rem;
        }
        .side-label {
            color: #8aa0c0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        .side-big {
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 900;
            margin-top: 0.25rem;
        }
        .side-progress {
            height: 4px;
            border-radius: 99px;
            background: #25334f;
            overflow: hidden;
            margin-top: 0.8rem;
        }
        .side-progress div {
            height: 100%;
            background: #6d5dfc;
        }
        .side-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }
        .side-mini {
            background: #0b1428;
            border: 1px solid #1f2b42;
            border-radius: 8px;
            padding: 0.65rem;
        }
        .side-footer {
            color: #8aa0c0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.72rem;
            letter-spacing: 0.15em;
            text-align: center;
            margin-top: 5rem;
        }
        .top-filter-pill {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 7px;
            padding: 0.45rem 0.8rem;
            color: #334155;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.72rem;
            text-align: center;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
            margin-top: 0.25rem;
        }
        .control-title {
            color: #020617;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }
        .model-card {
            border: 1px solid #0f172a;
            background: #ffffff;
            border-radius: 13px;
            padding: 1.1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .model-card-title {
            color: #111827;
            font-size: 0.94rem;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }
        .seg-metric-card {
            background: #ffffff;
            border: 1px solid #dfe7f1;
            border-radius: 12px;
            padding: 1rem;
            min-height: 92px;
        }
        .seg-index-card {
            background: #ffffff;
            border: 1px solid #dfe7f1;
            border-radius: 13px;
            padding: 1rem;
            min-height: 548px;
        }
        .cohort-scroll {
            max-height: 292px;
            overflow-y: auto;
            padding-right: 0.25rem;
            margin-top: 0.65rem;
        }
        .cohort-scroll::-webkit-scrollbar,
        .draft-body::-webkit-scrollbar {
            width: 7px;
        }
        .cohort-scroll::-webkit-scrollbar-thumb,
        .draft-body::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border-radius: 999px;
        }
        .cohort-row {
            display: grid;
            grid-template-columns: 18px 1fr auto;
            gap: 0.55rem;
            align-items: center;
            padding: 0.62rem 0.75rem;
            border-radius: 9px;
            background: #f8fafc;
            border: 1px solid #e7edf5;
            margin-bottom: 0.45rem;
        }
        .cohort-row.active {
            background: #0f172a;
            color: #ffffff;
            border-color: #0f172a;
        }
        div[role="radiogroup"] label {
            background: #f8fafc;
            border: 1px solid #e7edf5;
            border-radius: 9px;
            color: #334155;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.45rem;
            font-size: 0.78rem;
            font-weight: 850;
            width: 100%;
            min-height: 2.55rem;
            display: flex !important;
            align-items: center;
            justify-content: flex-start;
            white-space: nowrap;
        }
        div[role="radiogroup"] label:hover {
            background: #eef4fb;
            border-color: #d8e2ee;
            color: #0f172a;
        }
        div[role="radiogroup"] label:has(input:checked) {
            background: #0f172a;
            border-color: #0f172a;
            color: #ffffff;
        }
        div[role="radiogroup"] label input {
            appearance: none;
            width: 10px;
            height: 10px;
            min-width: 10px;
            border-radius: 999px;
            border: 0;
            background: #94a3b8;
            margin-right: 0.55rem;
        }
        div[role="radiogroup"] {
            max-height: 292px;
            overflow-y: auto;
            overflow-x: hidden;
            padding-right: 0.25rem;
            display: flex;
            flex-direction: column;
            flex-wrap: nowrap;
        }
        div[role="radiogroup"]::-webkit-scrollbar {
            width: 7px;
        }
        div[role="radiogroup"]::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border-radius: 999px;
        }
        .cohort-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
        }
        .cohort-name {
            font-size: 0.78rem;
            font-weight: 900;
        }
        .cohort-count {
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            color: #8aa0b8;
            font-size: 0.64rem;
            font-weight: 800;
        }
        .focus-card {
            background: #ffffff;
            border: 1px solid #dfe7f1;
            border-radius: 13px;
            padding: 1.1rem;
            min-height: 548px;
        }
        .focus-kicker {
            color: #10b981;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.66rem;
            font-weight: 900;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        .playbook {
            background: #f3f6ff;
            border: 1px solid #dfe6ff;
            border-radius: 10px;
            padding: 0.85rem;
            color: #334155;
            font-size: 0.78rem;
        }
        .draft-card {
            background: #f8fafc;
            border: 1px solid #dbe4ef;
            border-radius: 10px;
            overflow: hidden;
            min-height: 238px;
        }
        .draft-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #eef4ff;
            border-bottom: 1px solid #dbe4ef;
            padding: 0.55rem 0.7rem;
        }
        .draft-title {
            color: #64748b;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.66rem;
            font-weight: 900;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }
        .draft-badge {
            background: #ede9fe;
            color: #4f46e5;
            border-radius: 6px;
            padding: 0.18rem 0.35rem;
            font-size: 0.58rem;
            font-weight: 900;
        }
        .draft-body {
            padding: 0.7rem;
            color: #64748b;
            font-size: 0.74rem;
            line-height: 1.55;
            max-height: 205px;
            overflow-y: auto;
        }
        .behavior-card {
            background: #f8fafc;
            border: 1px solid #dfe7f1;
            border-radius: 10px;
            padding: 0.85rem;
            min-height: 80px;
        }
        .relative-copy {
            padding-top: 0.25rem;
            min-width: 0;
        }
        .relative-copy .card-subtitle {
            font-size: 0.68rem;
            line-height: 1.38;
            margin-top: 0.22rem;
        }
        .finger-card {
            background: #ffffff;
            border: 1px solid #dfe7f1;
            border-radius: 12px;
            padding: 1rem;
        }
        .finger-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.8rem;
            align-items: center;
            margin: 0.62rem 0;
        }
        .finger-label {
            color: #334155;
            font-size: 0.76rem;
            font-weight: 750;
        }
        .finger-value {
            color: #64748b;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.72rem;
            font-weight: 800;
        }
        .finger-track {
            grid-column: 1 / 3;
            height: 5px;
            border-radius: 999px;
            background: #e8eef6;
            overflow: hidden;
        }
        .finger-fill {
            height: 100%;
            border-radius: 999px;
        }
        .directory-card {
            background: #ffffff;
            border: 1px solid #dfe7f1;
            border-radius: 13px;
            overflow: hidden;
        }
        .directory-head {
            display: grid;
            grid-template-columns: 1fr minmax(280px, 360px) 92px;
            gap: 0.75rem;
            align-items: center;
            padding: 1rem 1.1rem;
            border-bottom: 1px solid #dfe7f1;
        }
        .directory-count {
            background: #e8eef6;
            color: #334155;
            border-radius: 8px;
            padding: 0.55rem 0.7rem;
            text-align: center;
            font-size: 0.7rem;
            font-weight: 900;
        }
        .directory-grid {
            display: grid;
            grid-template-columns: 2.15fr 1.35fr 1.2fr 1.05fr 0.9fr 0.9fr 0.95fr;
            align-items: center;
        }
        .directory-th {
            background: #f1f5f9;
            border-bottom: 1px solid #dfe7f1;
            color: #64748b;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.65rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 0.75rem 0.85rem;
            overflow-wrap: anywhere;
        }
        .directory-row {
            border-bottom: 1px solid #edf2f7;
            min-height: 72px;
        }
        .directory-cell {
            padding: 0.72rem 0.85rem;
            color: #334155;
            font-size: 0.78rem;
        }
        .client-name-wrap {
            display: grid;
            grid-template-columns: 38px 1fr;
            gap: 0.75rem;
            align-items: center;
        }
        .avatar {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            background: #eef4fb;
            border: 1px solid #d8e2ee;
            color: #64748b;
            font-weight: 900;
            font-size: 0.78rem;
        }
        .club-badge {
            display: inline-block;
            color: #f97316;
            background: #fff7ed;
            border: 1px solid #fdba74;
            border-radius: 5px;
            padding: 0.08rem 0.28rem;
            margin-left: 0.35rem;
            font-size: 0.55rem;
            font-weight: 900;
        }
        .client-sub {
            color: #8aa0b8;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.66rem;
            margin-top: 0.15rem;
        }
        .dossier-btn {
            display: inline-block;
            border: 1px solid #d8e2ee;
            border-radius: 8px;
            padding: 0.45rem 0.7rem;
            color: #0f172a;
            background: #ffffff;
            font-weight: 800;
            font-size: 0.72rem;
            text-align: center;
        }
        .profile-card {
            background:#ffffff;
            border:1px solid #dfe7f1;
            border-radius:13px;
            padding:1.2rem;
        }
        .profile-avatar {
            width:52px;
            height:52px;
            border-radius:12px;
            display:grid;
            place-items:center;
            background:#f59e0b;
            color:#ffffff;
            font-weight:900;
            font-size:1rem;
        }
        .profile-metric {
            background:#ffffff;
            border:1px solid #0f172a;
            border-radius:13px;
            padding:1rem;
            min-height:110px;
        }
        .matrix-row {
            display:flex;
            justify-content:space-between;
            gap:1rem;
            padding:0.72rem 0;
            border-bottom:1px solid #edf2f7;
            color:#334155;
            font-size:0.78rem;
        }
        .receipt-grid {
            display:grid;
            grid-template-columns:1.05fr 1.25fr 0.9fr 0.9fr 1fr 0.95fr;
            align-items:center;
        }
        .receipt-th {
            background:#f8fafc;
            border-bottom:1px solid #e2e8f0;
            color:#64748b;
            font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
            font-size:0.62rem;
            font-weight:900;
            letter-spacing:0.11em;
            text-transform:uppercase;
            padding:0.72rem 0.85rem;
        }
        .receipt-cell {
            padding:0.75rem 0.85rem;
            border-bottom:1px solid #edf2f7;
            color:#334155;
            font-size:0.78rem;
        }
        .match-card {
            background:#f8fafc;
            border:1px solid #e2e8f0;
            border-radius:10px;
            padding:0.85rem;
            margin-bottom:0.55rem;
        }
        .trend-hero {
            display:flex;
            gap:0.9rem;
            align-items:center;
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:13px;
            padding:1.15rem 1.25rem;
            box-shadow:0 1px 2px rgba(15,23,42,0.04);
        }
        .trend-spark {
            width:28px;
            height:28px;
            display:grid;
            place-items:center;
            color:#5b35f5;
            font-weight:900;
            font-size:1.05rem;
        }
        .trend-chip-row {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:1rem;
            border-bottom:1px solid #e2e8f0;
            padding-bottom:0.9rem;
            margin-bottom:1rem;
        }
        .metric-toggle {
            display:inline-flex;
            gap:0.55rem;
            background:#ffffff;
            border:1px solid #dfe7f1;
            border-radius:11px;
            padding:0.45rem 0.65rem;
            color:#334155;
            font-weight:900;
            font-size:0.74rem;
        }
        .metric-toggle a {
            color:#0f172a !important;
            text-decoration:none !important;
            border-radius:8px;
            padding:0.18rem 0.28rem;
            white-space:nowrap;
        }
        .metric-toggle a.active {
            color:#4f46e5 !important;
            background:#eef2ff;
        }
        div[data-testid="stSegmentedControl"] {
            display:flex;
            justify-content:flex-end;
        }
        div[data-testid="stSegmentedControl"] [role="radiogroup"] {
            display:flex !important;
            flex-direction:row !important;
            flex-wrap:nowrap !important;
            width:max-content !important;
            max-width:none !important;
            margin-left:auto;
            border:1px solid #dbe5f0;
            border-radius:12px;
            padding:0.22rem;
            background:#ffffff;
            gap:0.15rem;
        }
        div[data-testid="stSegmentedControl"] [role="radio"] {
            white-space:nowrap !important;
            min-width:max-content !important;
            border-radius:9px !important;
            padding:0.24rem 0.45rem !important;
            font-size:0.68rem !important;
            font-weight:900 !important;
            color:#334155 !important;
        }
        div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
            color:#4f46e5 !important;
            background:#eef2ff !important;
        }
        .st-key-trend_metric_picker [data-testid="stButtonGroup"] {
            display:flex !important;
            justify-content:flex-end !important;
            width:100% !important;
        }
        .st-key-trend_metric_picker [data-testid="stButtonGroup"] > div {
            display:inline-flex !important;
            flex-direction:row !important;
            flex-wrap:nowrap !important;
            align-items:center !important;
            width:max-content !important;
            margin-left:auto !important;
            border:1px solid #dbe5f0 !important;
            border-radius:12px !important;
            padding:0.22rem !important;
            gap:0.12rem !important;
            background:#ffffff !important;
        }
        .st-key-trend_metric_picker button[data-testid^="stBaseButton-segmented_control"] {
            width:auto !important;
            min-width:max-content !important;
            height:28px !important;
            padding:0.2rem 0.42rem !important;
            border:0 !important;
            border-radius:8px !important;
            box-shadow:none !important;
            white-space:nowrap !important;
        }
        .st-key-trend_metric_picker button[data-testid^="stBaseButton-segmented_control"] p {
            font-size:0.68rem !important;
            font-weight:900 !important;
            white-space:nowrap !important;
        }
        .st-key-trend_metric_picker button[data-testid="stBaseButton-segmented_controlActive"] {
            background:#eef2ff !important;
            color:#4f46e5 !important;
        }
        .trend-grid {
            display:grid;
            grid-template-columns:88px repeat(12, minmax(64px, 1fr));
            gap:5px;
            align-items:stretch;
            overflow-x:auto;
            padding:0.1rem;
        }
        .trend-head, .trend-week {
            background:#f8fafc;
            border:1px solid #e2e8f0;
            color:#64748b;
            font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
            font-size:0.64rem;
            font-weight:900;
            text-transform:uppercase;
            display:grid;
            place-items:center;
            min-height:40px;
            border-radius:6px;
        }
        .trend-week {
            color:#0f172a;
            text-transform:none;
            justify-content:start;
            padding-left:0.7rem;
        }
        .trend-cell {
            min-height:60px;
            border-radius:9px;
            display:grid;
            place-items:center;
            text-decoration:none !important;
            color:#ffffff !important;
            border:1px solid rgba(255,255,255,0.8);
            box-shadow:inset 0 0 0 1px rgba(255,255,255,0.18);
        }
        .trend-cell:hover {
            transform:translateY(-1px);
            outline:2px solid #ffffff;
            box-shadow:0 6px 18px rgba(79,70,229,0.22);
        }
        .trend-cell.active {
            outline:2px solid #ffffff;
            box-shadow:0 0 0 2px #635bff, 0 10px 24px rgba(79,70,229,0.24);
        }
        .trend-cell-main {
            font-weight:950;
            font-size:0.78rem;
            line-height:1.1;
        }
        .trend-cell-sub {
            color:rgba(255,255,255,0.78);
            font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
            font-size:0.58rem;
            font-weight:850;
            margin-top:0.15rem;
            text-align:center;
        }
        .trend-scale {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:1rem;
            padding:1rem 0.2rem 0;
            color:#94a3b8;
            font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
            font-size:0.64rem;
            font-weight:900;
            text-transform:uppercase;
        }
        .gradient-pill {
            width:110px;
            height:9px;
            border-radius:999px;
            background:linear-gradient(90deg,#eef2ff,#a5b4fc,#4338ca,#1d1bcf);
            display:inline-block;
            margin-left:0.5rem;
        }
        .trend-profile {
            background:#ffffff;
            border:1px solid #dfe7f1;
            border-radius:13px;
            padding:1.15rem 1.25rem;
            display:grid;
            grid-template-columns:1.15fr 1fr 1fr 1.05fr;
            gap:1.25rem;
            align-items:start;
        }
        .trend-profile > div:not(:last-child) {
            border-right:1px solid #e2e8f0;
            padding-right:1rem;
        }
        .trend-mini-box {
            background:#f8fafc;
            border:1px solid #e2e8f0;
            border-radius:10px;
            padding:0.8rem;
        }
        .trend-progress {
            height:7px;
            border-radius:999px;
            background:#eef2f7;
            overflow:hidden;
            margin:0.35rem 0 0.75rem;
        }
        .trend-progress span {
            display:block;
            height:100%;
            border-radius:999px;
        }
        .audit-pill {
            display:inline-block;
            color:#047857;
            background:#dcfce7;
            border:1px solid #bbf7d0;
            border-radius:8px;
            padding:0.36rem 0.6rem;
            font-size:0.72rem;
            font-weight:900;
            margin:0.25rem 0.35rem 0.25rem 0;
        }
        .audit-note {
            background:#f8fafc;
            border:1px solid #e2e8f0;
            border-radius:10px;
            padding:0.85rem;
            color:#64748b;
            font-size:0.78rem;
            line-height:1.5;
        }
        .ai-board-hero {
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:13px;
            padding:1.2rem 1.35rem;
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:1rem;
            box-shadow:0 1px 2px rgba(15,23,42,0.04);
        }
        .ai-board-title {
            color:#4f46e5;
            font-weight:900;
            font-size:0.95rem;
        }
        .ai-action {
            background:#5b35f5;
            color:#ffffff;
            border-radius:9px;
            padding:0.72rem 1rem;
            font-weight:900;
            font-size:0.78rem;
            white-space:nowrap;
            box-shadow:0 8px 20px rgba(79,70,229,0.22);
        }
        .st-key-ai_revaluate_btn button {
            background:#5b35f5 !important;
            color:#ffffff !important;
            border:0 !important;
            border-radius:9px !important;
            min-height:46px !important;
            font-weight:900 !important;
            font-size:0.78rem !important;
            box-shadow:0 8px 20px rgba(79,70,229,0.22) !important;
        }
        .st-key-ai_revaluate_btn button p {
            white-space:nowrap !important;
        }
        .st-key-ai_revaluate_btn button:hover {
            background:#4f35e8 !important;
            color:#ffffff !important;
        }
        .ai-insight-card {
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:13px;
            padding:1rem 1.15rem;
            min-height:126px;
            box-shadow:0 1px 2px rgba(15,23,42,0.04);
        }
        .ai-card-head {
            display:flex;
            align-items:center;
            gap:0.7rem;
            border-bottom:1px solid #e2e8f0;
            padding-bottom:0.8rem;
            margin-bottom:0.85rem;
        }
        .ai-index {
            min-width:32px;
            height:24px;
            border-radius:7px;
            display:grid;
            place-items:center;
            font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
            font-size:0.66rem;
            font-weight:900;
        }
        .ai-index.violet { color:#4f46e5;background:#eef2ff;border:1px solid #ddd6fe; }
        .ai-index.pink { color:#c026d3;background:#fae8ff;border:1px solid #f5d0fe; }
        .ai-index.green { color:#059669;background:#dcfce7;border:1px solid #bbf7d0; }
        .ai-index.amber { color:#d97706;background:#fef3c7;border:1px solid #fde68a; }
        .ai-insight-title {
            color:#0f172a;
            font-weight:900;
            font-size:0.82rem;
        }
        .ai-insight-body {
            color:#526987;
            font-size:0.78rem;
            line-height:1.55;
        }
        .ai-loading-card {
            min-height:420px;
            display:grid;
            place-items:center;
            text-align:center;
            color:#0f172a;
        }
        .ai-loader {
            width:56px;
            height:56px;
            border-radius:999px;
            display:grid;
            place-items:center;
            margin:0 auto 1.1rem;
            border:3px solid #e0e7ff;
            border-top-color:#5b35f5;
            animation:ai-spin 0.9s linear infinite;
            color:#5b35f5;
            font-weight:900;
            font-size:1.2rem;
        }
        @keyframes ai-spin {
            to { transform:rotate(360deg); }
        }
        .ai-loading-title {
            font-weight:900;
            font-size:1rem;
            color:#334155;
        }
        .ai-loading-subtitle {
            margin-top:0.35rem;
            color:#94a3b8;
            font-size:0.86rem;
            line-height:1.45;
        }
        .profile-picker-row {
            display: grid;
            grid-template-columns: minmax(130px, 1fr) minmax(175px, 1.35fr);
            gap: 0.75rem;
            align-items: center;
            padding: 0.55rem 0.35rem;
            border-bottom: 1px solid #edf2f7;
        }
        .profile-picker-name {
            color:#0f172a;
            font-weight:900;
            font-size:0.82rem;
            white-space:normal;
            line-height:1.25;
        }
        .profile-picker-meta {
            text-align:right;
            min-width:0;
        }
        .profile-picker-empty {
            color:#64748b;
            font-size:0.82rem;
            padding:0.8rem 0.2rem;
        }
        div[data-baseweb="select"] > div {
            border-radius: 11px;
        }
        div[data-testid="stPopoverBody"] div[data-testid="stButton"] button {
            min-height: 2.25rem;
            border-radius: 10px;
            justify-content: flex-start;
            font-weight: 800;
            color: #0f172a;
            background: #ffffff;
            border-color: #dbe5f0;
        }
        div[data-testid="stPopoverBody"] div[data-testid="stButton"] button:hover {
            border-color:#7c6cff;
            background:#f8f7ff;
            color:#4f46e5;
        }
        div[data-testid="stPopoverBody"] {
            max-height: 390px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_currency(value: float) -> str:
    return f"${value:,.2f}"


@st.dialog("Client Outreach Dossier", width="large")
def show_client_dossier(row_data: dict) -> None:
    name = str(row_data.get("Customer_Name") or f"Client {row_data.get('Customer_ID', '')}")
    customer_id = str(row_data.get("Customer_ID", ""))
    initials = "".join(part[0].upper() for part in name.replace("-", " ").split()[:2]) or "CL"
    loyalty_badge = "LOYALTY MEMBER" if row_data.get("Loyalty_Member") else "STANDARD MEMBER"
    discount_badge = "DISCOUNT SENSITIVE" if row_data.get("Discount_Used") else "FULL PRICE BUYER"
    st.markdown(
        f"""
        <div style="background:#0f172a;color:white;border-radius:14px 14px 0 0;padding:1.1rem 1.35rem;margin:-1rem -1rem 1rem;">
            <div style="display:flex;gap:1rem;align-items:center;">
                <div style="width:54px;height:54px;border-radius:14px;background:#5b35f5;display:grid;place-items:center;font-size:1.2rem;font-weight:900;">{escape(initials)}</div>
                <div>
                    <div style="font-size:1.25rem;font-weight:900;">{escape(name)} <span style="font-size:0.72rem;background:#111c33;border:1px solid #334155;border-radius:5px;padding:0.2rem 0.4rem;">ID: {escape(customer_id)}</span></div>
                    <div style="margin-top:0.25rem;color:#dbeafe;">Resident address mapped in <b>{escape(str(row_data.get('Location', 'Unknown')))}</b></div>
                </div>
            </div>
            <div style="border-top:1px solid #334155;margin-top:1rem;padding-top:0.85rem;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.75rem;">
                RFM CLUSTER ALIGNMENT:
                <span style="background:#10b981;color:white;border-radius:999px;padding:0.25rem 0.55rem;margin-left:0.5rem;">{escape(str(row_data.get('rfmSegment', 'Segment'))).upper()}</span>
                <span style="background:#0f766e;color:white;border-radius:999px;padding:0.25rem 0.55rem;margin-left:0.35rem;">{loyalty_badge}</span>
                <span style="background:#92400e;color:white;border-radius:999px;padding:0.25rem 0.55rem;margin-left:0.35rem;">{discount_badge}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    gender_age = f"{row_data.get('Gender', '-')}, {int(row_data.get('Age', 0))} yrs"
    income = str(row_data.get("Income_Class") or row_data.get("Income_Level") or "-")
    marital = str(row_data.get("Marital_Status") or "-")
    education = str(row_data.get("Education_Level") or "-")
    lifetime = fmt_currency(float(row_data.get("Purchase_Amount", 0)))
    frequency = f"{int(row_data.get('Frequency_of_Purchase', 0))} Checkouts"
    loyalty = f"{float(row_data.get('Product_Rating', 0)):.1f} / 5 Rating"
    decision = f"{float(row_data.get('recencyDays', 0)):.0f} days recency"
    favorite = str(row_data.get("Favorite_Category", "-"))
    satisfaction = float(row_data.get("Customer_Satisfaction", 0))
    def mini_card(label: str, value: str, color: str = "#0f172a") -> str:
        return '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;min-width:0;">' + f'<div class="metric-label">{escape(label)}</div><div style="font-weight:900;color:{color};font-size:0.9rem;margin-top:0.3rem;white-space:normal;overflow-wrap:anywhere;">{escape(value)}</div></div>'

    html = (
        '<div style="padding:0.2rem 0 0.4rem;">'
        '<div style="color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.72rem;font-weight:900;letter-spacing:0.12em;text-transform:uppercase;margin:0.65rem 0;">Demographics Demarcation</div>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;">'
        + mini_card("Gender & Age", gender_age)
        + mini_card("Income Bracket", income)
        + mini_card("Marital Status", marital)
        + mini_card("Education", education)
        + '</div><div style="color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.72rem;font-weight:900;letter-spacing:0.12em;text-transform:uppercase;margin:1rem 0 0.65rem;">Transactional Behavior Variables</div>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;">'
        + mini_card("Lifetime Purchase", lifetime)
        + mini_card("Purchase Frequency", frequency, "#4f46e5")
        + mini_card("Brand Loyalty Rating", loyalty)
        + mini_card("Decisive Duration", decision)
        + '</div><div style="color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.72rem;font-weight:900;letter-spacing:0.12em;text-transform:uppercase;margin:1rem 0 0.65rem;">Digital Journey Fingerprints</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.9rem;">'
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.95rem;display:grid;grid-template-columns:34px 1fr;gap:0.75rem;"><div style="width:30px;height:30px;border-radius:8px;background:#eef2ff;color:#4f46e5;display:grid;place-items:center;font-weight:900;">A</div><div><div style="font-weight:900;color:#0f172a;">Acquisition & Device Traits</div>'
        + f'<div style="color:#64748b;font-size:0.82rem;line-height:1.45;margin-top:0.25rem;">Checks out primarily in <b>{escape(favorite)}</b>, with behavior concentrated in this cohort.</div></div></div>'
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.95rem;display:grid;grid-template-columns:34px 1fr;gap:0.75rem;"><div style="width:30px;height:30px;border-radius:8px;background:#ecfdf5;color:#10b981;display:grid;place-items:center;font-weight:900;">T</div><div><div style="font-weight:900;color:#0f172a;">Time & Return Behaviors</div>'
        + f'<div style="color:#64748b;font-size:0.82rem;line-height:1.45;margin-top:0.25rem;">Last order recency is <b>{float(row_data.get("recencyDays", 0)):.0f} days</b> with satisfaction <b>{satisfaction:.1f}/10</b>.</div></div></div>'
        '</div><div style="border-top:1px solid #e2e8f0;margin:1rem 0 0.7rem;"></div><div style="color:#10b981;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.78rem;">Live profiling synchronize success</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    if st.button("Close Dossier", use_container_width=True):
        st.rerun()
    return
    st.markdown(
        f"""
        <div style="padding:0.2rem 0 0.4rem;">
            <div style="color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.72rem;font-weight:900;letter-spacing:0.12em;text-transform:uppercase;margin:0.65rem 0;">Demographics Demarcation</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;">
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;">
                    <div class="metric-label">Gender & Age</div>
                    <div style="font-weight:900;color:#0f172a;font-size:0.9rem;margin-top:0.3rem;">{escape(gender_age)}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;">
                    <div class="metric-label">Income Bracket</div>
                    <div style="font-weight:900;color:#0f172a;font-size:0.9rem;margin-top:0.3rem;">{escape(income)}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;">
                    <div class="metric-label">Marital Status</div>
                    <div style="font-weight:900;color:#0f172a;font-size:0.9rem;margin-top:0.3rem;">{escape(marital)}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;">
                    <div class="metric-label">Education</div>
                    <div style="font-weight:900;color:#0f172a;font-size:0.9rem;margin-top:0.3rem;">{escape(education)}</div>
                </div>
            </div>

            <div style="color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.72rem;font-weight:900;letter-spacing:0.12em;text-transform:uppercase;margin:1rem 0 0.65rem;">Transactional Behavior Variables</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;">
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;">
                    <div class="metric-label">Lifetime Purchase</div>
                    <div style="font-weight:900;color:#0f172a;font-size:0.9rem;margin-top:0.3rem;">{escape(lifetime)}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;">
                    <div class="metric-label">Purchase Frequency</div>
                    <div style="font-weight:900;color:#4f46e5;font-size:0.9rem;margin-top:0.3rem;">{escape(frequency)}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;">
                    <div class="metric-label">Brand Loyalty Rating</div>
                    <div style="font-weight:900;color:#0f172a;font-size:0.9rem;margin-top:0.3rem;">{escape(loyalty)}</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.8rem;">
                    <div class="metric-label">Decisive Duration</div>
                    <div style="font-weight:900;color:#0f172a;font-size:0.9rem;margin-top:0.3rem;">{escape(decision)}</div>
                </div>
            </div>

            <div style="color:#94a3b8;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.72rem;font-weight:900;letter-spacing:0.12em;text-transform:uppercase;margin:1rem 0 0.65rem;">Digital Journey Fingerprints</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.9rem;">
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.95rem;display:grid;grid-template-columns:34px 1fr;gap:0.75rem;">
                    <div style="width:30px;height:30px;border-radius:8px;background:#eef2ff;color:#4f46e5;display:grid;place-items:center;font-weight:900;">A</div>
                    <div>
                        <div style="font-weight:900;color:#0f172a;">Acquisition & Device Traits</div>
                        <div style="color:#64748b;font-size:0.82rem;line-height:1.45;margin-top:0.25rem;">Checks out primarily in <b>{escape(favorite)}</b>, with behavior concentrated in this cohort.</div>
                    </div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:11px;padding:0.95rem;display:grid;grid-template-columns:34px 1fr;gap:0.75rem;">
                    <div style="width:30px;height:30px;border-radius:8px;background:#ecfdf5;color:#10b981;display:grid;place-items:center;font-weight:900;">T</div>
                    <div>
                        <div style="font-weight:900;color:#0f172a;">Time & Return Behaviors</div>
                        <div style="color:#64748b;font-size:0.82rem;line-height:1.45;margin-top:0.25rem;">Last order recency is <b>{float(row_data.get('recencyDays', 0)):.0f} days</b> with satisfaction <b>{satisfaction:.1f}/10</b>.</div>
                    </div>
                </div>
            </div>
            <div style="border-top:1px solid #e2e8f0;margin:1rem 0 0.7rem;"></div>
            <div style="color:#10b981;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.78rem;">● Live profiling synchronize success</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Close Dossier", use_container_width=True):
        st.rerun()


def clean_bool(value) -> bool:
    return str(value).strip().upper() in {"TRUE", "1", "YES", "Y"}


def safe_numeric(series: pd.Series, fallback: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fallback)


@st.cache_data(show_spinner=False)
def fetch_dataset_from_url() -> pd.DataFrame:
    response = requests.get(DATASET_URL, timeout=20)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def load_raw_dataset() -> pd.DataFrame:
    try:
        return fetch_dataset_from_url()
    except Exception as exc:
        st.error(f"Dataset gagal dimuat dari Google Sheets: {exc}")
        st.info("Pastikan koneksi internet aktif atau ganti sumber dataset di kode.")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def normalize_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    df = raw_df.copy()
    expected_defaults = {
        "Customer_ID": "",
        "Customer_Name": "",
        "Product_Name": "",
        "Age": 35,
        "Gender": "Other",
        "Income_Level": "Middle",
        "Marital_Status": "Married",
        "Education_Level": "Bachelor's",
        "Income_Class": "",
        "Location": "City Centric",
        "Purchase_Category": "Unknown",
        "Purchase_Amount": 150.0,
        "Purchase_Channel": "Online",
        "Brand_Loyalty": 3,
        "Product_Rating": 4,
        "Research_Time_Hours": 2.0,
        "Social_Media_Influence": "Low",
        "Discount_Sensitivity": "Somewhat Sensitive",
        "Return_Rate": 0.0,
        "Customer_Satisfaction": 7,
        "Engagement_with_Ads": "Low",
        "Device_Used_for_Shopping": "Smartphone",
        "Payment_Method": "Credit Card",
        "Time_of_Purchase": "2024-06-01",
        "Discount_Used": False,
        "Loyalty_Member": False,
        "Purchase_Intent": "Impulsive",
        "Shipping_Preference": "Standard",
        "Time_to_Decision": 2.0,
        "Purchase_Month": 6,
        "Purchase_Month_Name": "Jun",
        "Purchase_Year": 2024,
        "Purchase_DayOfWeek": "Wednesday",
        "Purchase_Quarter": 2,
    }
    for col, default in expected_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["Customer_ID"] = df["Customer_ID"].fillna("").astype(str)
    missing_ids = df["Customer_ID"].str.strip().eq("")
    df.loc[missing_ids, "Customer_ID"] = [f"CUST-{1000 + i}" for i in np.where(missing_ids)[0]]

    df["Purchase_Category"] = df["Purchase_Category"].fillna("Unknown").astype(str)
    df["Purchase_Category"] = df["Purchase_Category"].replace(
        {"Travel & Leisure (Flights": "Travel & Leisure", "Packages)": "Travel & Leisure"}
    )

    df["Purchase_Amount"] = safe_numeric(df["Purchase_Amount"], 150.0).clip(lower=0.01).round(2)
    df["Device_Used_for_Shopping"] = df["Device_Used_for_Shopping"].fillna("Smartphone").astype(str)
    bad_device = df["Device_Used_for_Shopping"].eq("High")
    df.loc[bad_device, "Device_Used_for_Shopping"] = np.where(np.arange(len(df))[bad_device] % 2 == 0, "Smartphone", "Desktop")
    df["Discount_Sensitivity"] = df["Discount_Sensitivity"].fillna("Somewhat Sensitive").replace({"Medium": "Somewhat Sensitive"})
    df["Purchase_Channel"] = df["Purchase_Channel"].fillna("Online").replace({"7": "Mixed"})

    numeric_defaults = {
        "Age": 35,
        "Brand_Loyalty": 3,
        "Product_Rating": 4,
        "Research_Time_Hours": 2.0,
        "Return_Rate": 0.0,
        "Customer_Satisfaction": 7,
        "Time_to_Decision": 2.0,
        "Purchase_Month": 6,
        "Purchase_Year": 2024,
        "Purchase_Quarter": 2,
    }
    for col, fallback in numeric_defaults.items():
        df[col] = safe_numeric(df[col], fallback)

    text_defaults = {
        "Customer_Name": "",
        "Product_Name": "",
        "Gender": "Other",
        "Income_Level": "Middle",
        "Marital_Status": "Married",
        "Education_Level": "Bachelor's",
        "Income_Class": "",
        "Location": "City Centric",
        "Social_Media_Influence": "Low",
        "Engagement_with_Ads": "Low",
        "Payment_Method": "Credit Card",
        "Purchase_Intent": "Impulsive",
        "Shipping_Preference": "Standard",
        "Purchase_Month_Name": "Jun",
        "Purchase_DayOfWeek": "Wednesday",
    }
    for col, fallback in text_defaults.items():
        df[col] = df[col].fillna(fallback).astype(str).replace({"": fallback})

    df["Income_Class"] = np.where(df["Income_Class"].eq(""), df["Income_Level"], df["Income_Class"])
    df["Income_Level"] = np.where(df["Income_Level"].eq(""), df["Income_Class"], df["Income_Level"])
    df["Time_of_Purchase"] = pd.to_datetime(df["Time_of_Purchase"], errors="coerce").fillna(pd.Timestamp("2024-06-01"))
    df["Purchase_Month"] = df["Time_of_Purchase"].dt.month
    df["Purchase_Month_Name"] = df["Time_of_Purchase"].dt.strftime("%b")
    df["Purchase_Year"] = df["Time_of_Purchase"].dt.year
    df["Purchase_DayOfWeek"] = df["Time_of_Purchase"].dt.day_name()
    df["Purchase_Quarter"] = df["Time_of_Purchase"].dt.quarter
    df["Discount_Used"] = df["Discount_Used"].apply(clean_bool)
    df["Loyalty_Member"] = df["Loyalty_Member"].apply(clean_bool)

    df["recencyDays"] = (REFERENCE_DATE - df["Time_of_Purchase"]).dt.days.clip(lower=0).fillna(120)
    customer_summary = (
        df.groupby("Customer_ID", as_index=False)
        .agg(totalSpend=("Purchase_Amount", "sum"), count=("Purchase_Amount", "size"), minRecency=("recencyDays", "min"))
    )

    def recency_score(days: float) -> int:
        if days <= 30:
            return 5
        if days <= 90:
            return 4
        if days <= 180:
            return 3
        if days <= 270:
            return 2
        return 1

    def frequency_score(count: int) -> int:
        if count >= 10:
            return 5
        if count >= 5:
            return 4
        if count >= 3:
            return 3
        if count >= 2:
            return 2
        return 1

    def monetary_score(total: float) -> int:
        if total >= 2000:
            return 5
        if total >= 1000:
            return 4
        if total >= 500:
            return 3
        if total >= 250:
            return 2
        return 1

    def segment(row) -> str:
        r, f, m = row["recencyScore"], row["frequencyScore"], row["monetaryScore"]
        if r >= 4 and m >= 4:
            return "High Value Customers"
        if f >= 4 and r >= 3:
            return "Loyal Customers"
        if f >= 4 and r <= 2:
            return "Frequent Buyers"
        if r >= 4 and f >= 2:
            return "Potential Loyalists"
        if r >= 4 and f == 1:
            return "New Customers"
        if r <= 2 and (f >= 3 or m >= 3):
            return "At Risk Customers"
        if r <= 2 and f <= 2:
            return "Lost Customers"
        return "Regular Customers"

    customer_summary["recencyScore"] = customer_summary["minRecency"].apply(recency_score)
    customer_summary["frequencyScore"] = customer_summary["count"].apply(frequency_score)
    customer_summary["monetaryScore"] = customer_summary["totalSpend"].apply(monetary_score)
    customer_summary["rfmSegment"] = customer_summary.apply(segment, axis=1)

    df = df.merge(customer_summary, on="Customer_ID", how="left")
    df["Frequency_of_Purchase"] = df["count"]
    df["recencyDays"] = df["minRecency"]
    return df


def init_filter_state() -> None:
    defaults = {
        "filter_preset": "Full Year 2024",
        "filter_start": date(2024, 1, 1),
        "filter_end": date(2024, 12, 31),
        "filter_category": "All Categories",
        "filter_segment": "All Segments",
        "filter_gender": "All Genders",
        "filter_device": "All Devices",
        "filter_income": "All Levels",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_filter_state(df: pd.DataFrame) -> pd.DataFrame:
    ranges = {
        "Full Year 2024": (date(2024, 1, 1), date(2024, 12, 31)),
        "Q1": (date(2024, 1, 1), date(2024, 3, 31)),
        "Q2": (date(2024, 4, 1), date(2024, 6, 30)),
        "Q3": (date(2024, 7, 1), date(2024, 9, 30)),
        "Q4": (date(2024, 10, 1), date(2024, 12, 31)),
    }
    preset = st.session_state.get("filter_preset", "Full Year 2024")
    if preset == "Custom":
        start = st.session_state.get("filter_start", date(2024, 1, 1))
        end = st.session_state.get("filter_end", date(2024, 12, 31))
    else:
        start, end = ranges[preset]
        st.session_state["filter_start"] = start
        st.session_state["filter_end"] = end

    filtered = df[(df["Time_of_Purchase"].dt.date >= start) & (df["Time_of_Purchase"].dt.date <= end)]
    return apply_non_time_filters(filtered)


def apply_non_time_filters(df: pd.DataFrame) -> pd.DataFrame:
    category = st.session_state.get("filter_category", "All Categories")
    segment = st.session_state.get("filter_segment", "All Segments")
    gender = st.session_state.get("filter_gender", "All Genders")
    device = st.session_state.get("filter_device", "All Devices")
    income = st.session_state.get("filter_income", "All Levels")

    filtered = df
    if category != "All Categories":
        filtered = filtered[filtered["Purchase_Category"].eq(category)]
    if segment != "All Segments":
        filtered = filtered[filtered["rfmSegment"].eq(segment)]
    if gender != "All Genders":
        filtered = filtered[filtered["Gender"].eq(gender)]
    if device != "All Devices":
        filtered = filtered[filtered["Device_Used_for_Shopping"].eq(device)]
    if income != "All Levels":
        filtered = filtered[filtered["Income_Level"].eq(income)]
    return filtered


def render_filter_header(df: pd.DataFrame, page: str) -> pd.DataFrame:
    init_filter_state()
    left, right = st.columns([1.2, 0.75])
    with left:
        st.markdown(
            f"""
            <div class="page-head">
            <div class="dash-eyebrow">{escape(page)} Page</div>
            <div class="dash-title">{escape(page)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        rows_col, filter_col = st.columns([1, 0.92])
        filtered_preview = apply_filter_state(df)
        rows_col.markdown(
            f"""<div class="top-filter-pill">{len(filtered_preview):,}/{len(df):,} Rows</div>""",
            unsafe_allow_html=True,
        )
        with filter_col:
            with st.popover("Filters", use_container_width=True):
                st.markdown('<div class="control-title">Control Center</div>', unsafe_allow_html=True)
                st.caption("TIME RANGE SELECTION")
                st.radio(
                    "Time range",
                    ["Full Year 2024", "Q1", "Q2", "Q3", "Q4", "Custom"],
                    key="filter_preset",
                    label_visibility="collapsed",
                )
                if st.session_state["filter_preset"] == "Custom":
                    st.date_input("Start date", key="filter_start")
                    st.date_input("End date", key="filter_end")

                def options(column: str, all_label: str) -> list[str]:
                    return [all_label] + sorted(df[column].dropna().astype(str).unique().tolist())

                st.caption("PRODUCT CATEGORY")
                st.selectbox("Product category", options("Purchase_Category", "All Categories"), key="filter_category", label_visibility="collapsed")
                st.caption("RFM CLUSTER SEGMENT")
                st.selectbox("RFM cluster segment", options("rfmSegment", "All Segments"), key="filter_segment", label_visibility="collapsed")
                st.caption("CUSTOMER PROFILE")
                st.selectbox("Gender", options("Gender", "All Genders"), key="filter_gender")
                st.selectbox("Device", options("Device_Used_for_Shopping", "All Devices"), key="filter_device")
                st.selectbox("Income level", options("Income_Level", "All Levels"), key="filter_income")

                if st.button("Reset to Defaults", use_container_width=True):
                    for key in [
                        "filter_preset",
                        "filter_start",
                        "filter_end",
                        "filter_category",
                        "filter_segment",
                        "filter_gender",
                        "filter_device",
                        "filter_income",
                    ]:
                        st.session_state.pop(key, None)
                    init_filter_state()
                    st.rerun()
    st.divider()
    filtered = apply_filter_state(df)
    return filtered


def customer_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby("Customer_ID")
        .agg(
            Customer_Name=("Customer_Name", "first"),
            Age=("Age", "first"),
            Gender=("Gender", "first"),
            Location=("Location", "first"),
            Marital_Status=("Marital_Status", "first"),
            Education_Level=("Education_Level", "first"),
            Income_Level=("Income_Level", "first"),
            Income_Class=("Income_Class", "first"),
            Purchase_Channel=("Purchase_Channel", "first"),
            Payment_Method=("Payment_Method", "first"),
            Device_Used_for_Shopping=("Device_Used_for_Shopping", "first"),
            Return_Rate=("Return_Rate", "mean"),
            Purchase_Intent=("Purchase_Intent", "first"),
            rfmSegment=("rfmSegment", "first"),
            Purchase_Amount=("Purchase_Amount", "sum"),
            Frequency_of_Purchase=("Purchase_Amount", "size"),
            recencyDays=("recencyDays", "min"),
            Product_Rating=("Product_Rating", "mean"),
            Customer_Satisfaction=("Customer_Satisfaction", "mean"),
            Favorite_Category=("Purchase_Category", lambda x: x.value_counts().index[0]),
            Discount_Sensitivity=("Discount_Sensitivity", "first"),
            Discount_Used=("Discount_Used", "max"),
            Loyalty_Member=("Loyalty_Member", "first"),
        )
        .reset_index()
    )
    grouped["Customer_Name"] = grouped.apply(
        lambda r: r["Customer_Name"] if str(r["Customer_Name"]).strip() else f"Client {r['Customer_ID']}", axis=1
    )
    return grouped.sort_values("Purchase_Amount", ascending=False)


def render_sidebar_summary(records: pd.DataFrame, filtered: pd.DataFrame) -> None:
    total_rows = len(records)
    active_rows = len(filtered)
    coverage = (active_rows / total_rows * 100) if total_rows else 0
    sales = filtered["Purchase_Amount"].sum()
    avg_spend = filtered["Purchase_Amount"].mean() if active_rows else 0
    st.sidebar.markdown(
        f"""
        <div class="side-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="side-label">Active Data Segment</div>
                <div style="color:#34d399;font-size:0.68rem;">&bull; LIVE</div>
            </div>
            <div class="side-big">{active_rows:,} <span style="font-size:0.72rem;color:#8aa0c0;">/ {total_rows:,} rows</span></div>
            <div style="display:flex;justify-content:space-between;color:#8aa0c0;font-size:0.68rem;margin-top:0.65rem;">
                <span>Coverage</span><span>{coverage:.0f}%</span>
            </div>
            <div class="side-progress"><div style="width:{min(coverage, 100):.1f}%"></div></div>
        </div>
        <div style="margin-top:1.1rem;" class="side-label">Selected Cohort Metrics</div>
        <div class="side-grid">
            <div class="side-mini">
                <div class="side-label" style="font-size:0.58rem;">Cohort Sales</div>
                <div style="color:#fff;font-weight:900;">{escape(fmt_currency(sales).replace('.00', ''))}</div>
            </div>
            <div class="side-mini">
                <div class="side-label" style="font-size:0.58rem;">Average Spend</div>
                <div style="color:#fff;font-weight:900;">{escape(fmt_currency(avg_spend).replace('.00', ''))}</div>
            </div>
        </div>
        <div style="margin-top:1.1rem;" class="side-label">Active Constraints</div>
        <div class="side-mini" style="margin-top:0.55rem;text-align:center;color:#8aa0c0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;">
            Complete 2024 Scope Active
        </div>
        <div class="side-footer">SYSTEM ONLINE - {total_rows:,} RECORDS</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav() -> str:
    pages = [
        ("Executive Dashboard", "KPI Cards & Sales Demographics", "dashboard", ":material/analytics:"),
        ("Customer Segmentation", "RFM Clusters & Target Strategy", "layers", ":material/layers:"),
        ("Customer Profiles", "Individual Profiles & Metrics Search", "profile", ":material/manage_accounts:"),
        ("Sales Trend Analysis", "Seasonality & Next 30-Day Forecasts", "calendar", ":material/calendar_month:"),
        ("AI Insights", "Automated Strategic Advisory", "brain", ":material/neurology:"),
    ]
    nav_icons = {
        "dashboard": '<svg viewBox="0 0 24 24"><path d="M4 19V9"/><path d="M10 19V5"/><path d="M16 19v-8"/><path d="M22 19V3"/></svg>',
        "layers": '<svg viewBox="0 0 24 24"><path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="m3 13 9 5 9-5"/><path d="m3 18 9 5 9-5"/></svg>',
        "profile": '<svg viewBox="0 0 24 24"><circle cx="8" cy="8" r="3"/><path d="M3 20a5 5 0 0 1 10 0"/><path d="M15 7h6"/><path d="M15 12h6"/><path d="M15 17h4"/></svg>',
        "calendar": '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4"/><path d="M8 3v4"/><path d="M3 11h18"/></svg>',
        "brain": '<svg viewBox="0 0 24 24"><path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 5 2V4Z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-5 2V4Z"/><path d="M9 9H6"/><path d="M15 9h3"/><path d="M9 15H6"/><path d="M15 15h3"/></svg>',
    }
    valid_names = [name for name, _, _, _ in pages]
    st.session_state.setdefault("nav_page", "Executive Dashboard")
    query_page = st.query_params.get("page", "")
    if isinstance(query_page, list):
        query_page = query_page[0] if query_page else ""
    if query_page in valid_names:
        st.session_state["nav_page"] = query_page
    if st.session_state["nav_page"] not in valid_names:
        st.session_state["nav_page"] = "Executive Dashboard"
    current_page = st.session_state["nav_page"]

    st.sidebar.markdown('<div class="side-nav">', unsafe_allow_html=True)
    for name, subtitle, icon, button_icon in pages:
        icon_html = nav_icons.get(icon, "")
        if name == current_page:
            st.sidebar.markdown(
                f"""
                <div class="side-nav-item active">
                    <div class="side-nav-icon">{icon_html}</div>
                    <div>
                        <div class="side-nav-title">{escape(name)}</div>
                        <div class="side-nav-subtitle">{escape(subtitle)}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            label = f"**{name}**\n\n:small[{subtitle}]"
            if st.sidebar.button(label, key=f"nav_{name}", icon=button_icon, use_container_width=True):
                st.session_state["nav_page"] = name
                st.query_params["page"] = name
                st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    return current_page


def render_dashboard(records: pd.DataFrame, all_records: pd.DataFrame) -> None:
    revenue = records["Purchase_Amount"].sum()
    orders = len(records)
    customers = records["Customer_ID"].nunique()
    aov = revenue / orders if orders else 0
    avg_satisfaction = records["Customer_Satisfaction"].mean() if orders else 0
    unique_customers = customer_table(records)
    loyalty_count = int(unique_customers["Customer_ID"].nunique()) if not unique_customers.empty else 0
    loyalty_ratio = records["Loyalty_Member"].mean() * 100 if orders else 0
    retention = all_records["Loyalty_Member"].mean() * 100 if len(all_records) else 0
    q1 = all_records.loc[all_records["Purchase_Quarter"].eq(1), "Purchase_Amount"].sum()
    q4 = all_records.loc[all_records["Purchase_Quarter"].eq(4), "Purchase_Amount"].sum()
    growth = ((q4 - q1) / q1 * 100) if q1 else 0

    def metric_card(label: str, value: str, pill: str, icon: str, pill_color: str = "green") -> str:
        pill_styles = {
            "green": "background:#ecfdf5;color:#059669;",
            "blue": "background:#eef2ff;color:#4f46e5;",
            "amber": "background:#fffbeb;color:#d97706;",
            "slate": "background:#f1f5f9;color:#64748b;",
        }
        icons = {
            "revenue": '<svg viewBox="0 0 24 24"><path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7H14a3.5 3.5 0 0 1 0 7H6"/></svg>',
            "cart": '<svg viewBox="0 0 24 24"><circle cx="9" cy="20" r="1.6"/><circle cx="18" cy="20" r="1.6"/><path d="M3 4h2l2.4 11.2a2 2 0 0 0 2 1.6h8.8a2 2 0 0 0 1.9-1.4L22 8H7"/></svg>',
            "trend": '<svg viewBox="0 0 24 24"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg>',
            "users": '<svg viewBox="0 0 24 24"><path d="M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"/><circle cx="9.5" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        }
        return f"""
        <div class="dashboard-card metric-card">
            <div>
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-value">{escape(value)}</div>
                <div class="metric-pill" style="{pill_styles.get(pill_color, pill_styles['green'])}">{escape(pill)}</div>
            </div>
            <div class="metric-icon">{icons.get(icon, icons["revenue"])}</div>
        </div>
        """

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total Revenue", fmt_currency(revenue), f"{growth:+.1f}% YoY Target", "revenue", "green"), unsafe_allow_html=True)
    c2.markdown(metric_card("Total Sales", f"{orders:,}", "Orders cataloged", "cart", "slate"), unsafe_allow_html=True)
    c3.markdown(metric_card("Avg Order Value", fmt_currency(aov), f"Satisfaction: {avg_satisfaction:.1f}/10", "trend", "amber"), unsafe_allow_html=True)
    c4.markdown(metric_card("Customer Loyalty", f"{loyalty_count:,} Active", f"{loyalty_ratio:.1f}% Member Ratio", "users", "blue"), unsafe_allow_html=True)
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])
    monthly = records.groupby("Purchase_Month_Name", as_index=False).agg(revenue=("Purchase_Amount", "sum"), orders=("Purchase_Amount", "size"))
    monthly = pd.DataFrame({"Purchase_Month_Name": MONTH_ORDER}).merge(monthly, how="left", on="Purchase_Month_Name")
    monthly["revenue"] = monthly["revenue"].fillna(0)
    monthly["orders"] = monthly["orders"].fillna(0)
    monthly["order"] = monthly["Purchase_Month_Name"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    monthly = monthly.sort_values("order")
    with left:
        st.markdown(
            """
            <div class="dashboard-card">
                <p class="card-title">Monthly Revenue Performance</p>
                <p class="card-subtitle">Breakdown of gross transaction volumes by month</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly["Purchase_Month_Name"],
                y=monthly["revenue"],
                mode="lines",
                name="revenue",
                line=dict(color="#0ea5e9", width=3),
                fill="tozeroy",
                fillcolor="rgba(14, 165, 233, 0.12)",
                hovertemplate="%{x}<br>Revenue: $%{y:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=330,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER, showgrid=False, tickfont=dict(color="#8aa0b8")),
            yaxis=dict(gridcolor="#edf2f7", zeroline=False, tickfont=dict(color="#8aa0b8")),
        )
        st.plotly_chart(fig, use_container_width=True)

    cat = records.groupby("Purchase_Category", as_index=False).agg(revenue=("Purchase_Amount", "sum"), orders=("Purchase_Amount", "size"))
    cat = cat.sort_values("revenue", ascending=False).head(8)
    with right:
        st.markdown(
            """
            <div class="dashboard-card">
                <p class="card-title">Category Market Share</p>
                <p class="card-subtitle">Gross revenue allocation per product family</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        donut_colors = ["#0284c7", "#0ea5e9", "#38bdf8", "#14b8a6", "#2dd4bf", "#94a3b8", "#6366f1", "#8b5cf6"]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=cat["Purchase_Category"],
                    values=cat["revenue"],
                    hole=0.62,
                    marker=dict(colors=donut_colors, line=dict(color="white", width=4)),
                    textinfo="none",
                    hovertemplate="%{label}<br>$%{value:,.2f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            height=330,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", y=-0.05, font=dict(size=10, color="#334155")),
            annotations=[
                dict(text=f"<b>TOTAL</b><br>{fmt_currency(revenue).replace('.00', '')}", x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#0f172a"))
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

    top_products = records.assign(
        Product_Display=lambda d: np.where(
            d["Product_Name"].astype(str).str.strip().ne(""),
            d["Product_Name"],
            d["Purchase_Category"] + " Spec-" + d["Device_Used_for_Shopping"],
        )
    )
    top_products = top_products.groupby(["Product_Display", "Purchase_Category"], as_index=False).agg(
        revenue=("Purchase_Amount", "sum"), orders=("Purchase_Amount", "size")
    ).sort_values("revenue", ascending=False).head(10)
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([0.85, 1.65])
    with col_a:
        st.markdown(
            """
            <div class="dashboard-card">
                <p class="card-title">Top 10 Active Product Profiles</p>
                <p class="card-subtitle">Identified profiles by aggregate sales volume</p>
            """,
            unsafe_allow_html=True,
        )
        max_product_revenue = max(float(top_products["revenue"].max()), 1.0)
        rows_html = []
        for idx, row in enumerate(top_products.itertuples(), start=1):
            pct = min(float(row.revenue) / max_product_revenue * 100, 100)
            rows_html.append(
                f"""
                <div class="product-row">
                    <div class="product-rank">{idx:02d}</div>
                    <div class="product-name" title="{escape(str(row.Product_Display))}">{escape(str(row.Product_Display))}</div>
                    <div class="product-meta"><b>{escape(fmt_currency(row.revenue).replace('.00', ''))}</b><br>{int(row.orders)} orders</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
                </div>
                """
            )
        st.markdown("".join(rows_html) + "</div>", unsafe_allow_html=True)

    customers_df = customer_table(records)
    top_revenue_customers = customers_df.head(5).copy()
    top_revenue_customers["Demographics"] = top_revenue_customers.apply(lambda r: f"{int(r['Age'])} y/o ({r['Gender']})", axis=1)
    top_revenue_customers["Order Amount"] = top_revenue_customers["Purchase_Amount"].map(fmt_currency)
    top_revenue_customers = top_revenue_customers[["Customer_ID", "Demographics", "Location", "rfmSegment", "Order Amount"]]

    top_frequency_customers = customers_df.sort_values("Frequency_of_Purchase", ascending=False).head(5).copy()
    top_frequency_customers["Average Satisfaction"] = top_frequency_customers["Customer_Satisfaction"].map(lambda v: f"{v:.1f} / 10")
    top_frequency_customers["Loyalty Frequency"] = top_frequency_customers["Frequency_of_Purchase"].map(lambda v: f"{int(v)} purchases")
    top_frequency_customers = top_frequency_customers[
        ["Customer_ID", "Favorite_Category", "rfmSegment", "Average Satisfaction", "Loyalty Frequency"]
    ]

    with col_b:
        st.markdown(
            """
            <div class="dashboard-card">
                <p class="card-title">Top Revenue Contributing Customers</p>
                <p class="card-subtitle">Highest-paying customer accounts sorted by order contribution</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(top_revenue_customers, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class="dashboard-card" style="margin-top:1rem;">
                <p class="card-title">Top Frequent Customers Order Index</p>
                <p class="card-subtitle">High frequency buyer cohorts and categorical distribution indicators</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(top_frequency_customers, use_container_width=True, hide_index=True)


def render_segmentation(records: pd.DataFrame) -> None:
    customers = customer_table(records)
    if customers.empty:
        st.warning("No customer records match the selected filters.")
        return

    segment_stats = customers.groupby("rfmSegment", as_index=False).agg(
        customers=("Customer_ID", "nunique"),
        revenue=("Purchase_Amount", "sum"),
        avg_age=("Age", "mean"),
        avg_recency=("recencyDays", "mean"),
        avg_frequency=("Frequency_of_Purchase", "mean"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
        loyalty_ratio=("Loyalty_Member", "mean"),
        discount_ratio=("Discount_Used", "mean"),
    )
    segment_stats["revenue_pct"] = segment_stats["revenue"] / max(segment_stats["revenue"].sum(), 1) * 100
    segment_stats["customer_pct"] = segment_stats["customers"] / max(customers["Customer_ID"].nunique(), 1) * 100
    segment_stats = segment_stats.sort_values("revenue", ascending=False)

    valid_segments = segment_stats["rfmSegment"].tolist()
    st.session_state.setdefault("active_segment", valid_segments[0])
    if st.session_state["active_segment"] not in valid_segments:
        st.session_state["active_segment"] = valid_segments[0]
    active_segment = st.session_state["active_segment"]
    active_stats = segment_stats[segment_stats["rfmSegment"].eq(active_segment)].iloc[0]
    desc, action = SEGMENT_STRATEGY.get(active_segment, SEGMENT_STRATEGY["Regular Customers"])
    active_customers = customers[customers["rfmSegment"].eq(active_segment)].copy()
    active_records = records[records["Customer_ID"].isin(active_customers["Customer_ID"])]
    total_customers = customers["Customer_ID"].nunique()
    total_revenue = customers["Purchase_Amount"].sum()

    st.markdown(
        f"""
        <div class="model-card">
            <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center;">
                <div>
                    <div class="model-card-title">Interactive Multi-Dimensional Client Cohorts Model</div>
                    <div class="card-subtitle" style="max-width:820px;">
                        Automatically categorizes customers based on Recency, Frequency, and Monetary Values. Explore demographic fingerprints,
                        purchase behavior variables, and correlation charts to formulate customized loyalty programs.
                    </div>
                </div>
                <div class="dashboard-card" style="min-width:150px;text-align:center;padding:0.85rem;">
                    <div class="metric-label">Database Coverage</div>
                    <div style="font-weight:900;font-size:1.25rem;color:#0f172a;">{total_customers:,} Users active</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metric_payload = [
        ("Cohort Size", f"{int(active_stats['customers']):,}", f"{active_stats['customer_pct']:.1f}% active customer share"),
        ("Total Monetization", fmt_currency(active_stats["revenue"]), f"{active_stats['revenue_pct']:.1f}% gross sales contribution"),
        ("Engagement Traits", f"{active_stats['avg_frequency']:.1f} Buy Cyl.", f"Avg purchases, satisfaction {active_stats['avg_satisfaction']:.1f}/10"),
        ("Retentive Loyalty Program", f"{active_stats['loyalty_ratio'] * 100:.1f}%", "Subscribed loyalty penetration"),
    ]
    for col, (label, value, sub) in zip([m1, m2, m3, m4], metric_payload):
        col.markdown(
            f"""
            <div class="seg-metric-card">
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-value" style="font-size:1.25rem;">{escape(value)}</div>
                <div class="card-subtitle">{escape(sub)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    left, right = st.columns([0.95, 1.75])
    with left:
        with st.container(border=True):
            st.markdown(
                """
                <p class="card-title">Target Cohorts Index</p>
                <p class="card-subtitle">Click a cohort segment below to load tactical recommendations, demographic fingerprints, and dedicated rosters.</p>
                <div style="border-top:1px solid #e2e8f0;margin:0.85rem 0 0.6rem;"></div>
                """,
                unsafe_allow_html=True,
            )
            count_lookup = {row.rfmSegment: int(row.customers) for row in segment_stats.itertuples()}
            selected_segment = st.radio(
                "Choose cohort",
            valid_segments,
            index=valid_segments.index(active_segment),
            format_func=lambda name: f"{name} - {count_lookup.get(name, 0)} customers",
            key="active_segment_radio",
            label_visibility="collapsed",
        )
            if selected_segment != active_segment:
                st.session_state["active_segment"] = selected_segment
                st.rerun()
            st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:0.9rem 0;'>", unsafe_allow_html=True)
            donut_labels = segment_stats["rfmSegment"].tolist()
            donut_values = segment_stats["customers"].tolist()
            muted_colors = ["#cbd5e1", "#67e8f9", "#fca5a5", "#fdba74", "#fcd34d", "#93c5fd", "#ddd6fe", "#a7f3d0"]
            donut_colors = [
                SEGMENT_COLORS.get(label, "#10b981") if label == active_segment else muted_colors[idx % len(muted_colors)]
                for idx, label in enumerate(donut_labels)
            ]
            total_donut = max(sum(donut_values), 1)
            cursor = 0.0
            gradient_parts = []
            for value, color in zip(donut_values, donut_colors):
                start = cursor
                cursor += (value / total_donut) * 100
                gradient_parts.append(f"{color} {start:.2f}% {cursor:.2f}%")
            gradient = ", ".join(gradient_parts)
            st.markdown(
                f"""
                <div style="display:flex;gap:0.85rem;align-items:center;min-width:0;">
                    <div style="width:82px;height:82px;min-width:82px;border-radius:999px;background:conic-gradient({gradient});display:grid;place-items:center;">
                        <div style="width:48px;height:48px;border-radius:999px;background:#ffffff;display:grid;place-items:center;color:#0f172a;font-weight:900;font-size:0.68rem;">
                            {active_stats['customer_pct']:.1f}%
                        </div>
                    </div>
                    <div class="relative-copy">
                        <div style="font-weight:900;color:#0f172a;">Relative Distribution</div>
                        <div class="card-subtitle">This cohort encapsulates <b>{active_stats['customer_pct']:.1f}%</b> of total audience and drives <b>{active_stats['revenue_pct']:.1f}%</b> of total monetary revenue.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        subject = f"Exclusive First-Look: The Elite {active_segment.split()[0]} Collection Inside..."
        st.markdown(
            f"""
            <div class="focus-card">
                <div style="display:flex;gap:0.85rem;align-items:center;border-bottom:1px solid #e2e8f0;padding-bottom:0.8rem;">
                    <div style="width:48px;height:48px;border-radius:13px;background:#ecfdf5;border:1px solid #bbf7d0;display:grid;place-items:center;color:#10b981;font-weight:900;">AF</div>
                    <div>
                        <div class="focus-kicker">Active Focus Segment</div>
                        <h3 style="margin:0.15rem 0 0;color:#0f172a;">{escape(active_segment)}</h3>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;">
                    <div>
                        <div class="metric-label">Cohort Profile Analysis</div>
                        <div style="color:#334155;font-size:0.86rem;line-height:1.55;margin:0.7rem 0 1rem;">{escape(desc)}</div>
                        <div class="playbook">
                            <div class="metric-label" style="color:#4f46e5;">Marketing Playbook Deployment</div>
                            <div style="margin-top:0.5rem;">{escape(action)}</div>
                        </div>
                    </div>
                    <div class="draft-card">
                        <div class="draft-head">
                            <div class="draft-title">Draft Outreach Campaign</div>
                            <div class="draft-badge">Cohort Filter Target</div>
                        </div>
                        <div class="draft-body">
                            <div><b>Subject:</b> {escape(subject)}</div>
                            <div style="border-top:1px solid #e2e8f0;margin:0.55rem 0;padding-top:0.55rem;"><b>Audience:</b> All {escape(active_segment)} Members</div>
                            <p>Dear Valued Shopper,</p>
                            <p>As one of our most valued client patrons, we are delighted to invite you to a private campaign tailored to your purchase history and segment behavior.</p>
                            <p>{escape(action)}</p>
                            <p>Use your priority access benefit during the next checkout window.</p>
                        </div>
                    </div>
                </div>
                <div style="border-top:1px solid #e2e8f0;margin-top:1.2rem;padding-top:1rem;">
                    <div class="metric-label">Key Segment Behavior Profiles</div>
                    <div class="side-grid" style="grid-template-columns:repeat(3,1fr);margin-top:0.8rem;">
                        <div class="behavior-card">
                            <div class="metric-label">Avg. Recency</div>
                            <div style="color:#0f172a;font-weight:900;font-size:1.05rem;">{active_stats['avg_recency']:.1f} Days</div>
                        </div>
                        <div class="behavior-card">
                            <div class="metric-label">CSAT Satisfaction</div>
                            <div style="color:#0f172a;font-weight:900;font-size:1.05rem;">{active_stats['avg_satisfaction']:.1f} / 10</div>
                        </div>
                        <div class="behavior-card">
                            <div class="metric-label">Promo Usage Ratio</div>
                            <div style="color:#0f172a;font-weight:900;font-size:1.05rem;">{active_stats['discount_ratio'] * 100:.1f}%</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="card-title">Target Cohort Fingerprint Breakdown</p>
        <p class="card-subtitle">Examine specific demographic traits, channels, and behavioral indexes for {escape(active_segment)}.</p>
        """,
        unsafe_allow_html=True,
    )

    def distribution_card(title: str, series: pd.Series, color: str, note: str, order: list[str] | None = None) -> str:
        clean = series.dropna().astype(str)
        counts = clean.value_counts()
        if order:
            ordered = [(label, int(counts.get(label, 0))) for label in order if int(counts.get(label, 0)) > 0]
            rest = [(label, int(count)) for label, count in counts.items() if label not in order]
            items = ordered + rest
        else:
            items = [(label, int(count)) for label, count in counts.items()]
        items = items[:4]
        total = max(int(len(clean)), 1)
        rows = []
        for label, count in items:
            pct = count / total * 100
            rows.append(
                f'<div class="finger-row">'
                f'<div class="finger-label">{escape(label)}</div>'
                f'<div class="finger-value">{pct:.1f}%</div>'
                f'<div class="finger-track"><div class="finger-fill" style="width:{pct:.1f}%;background:{color};"></div></div>'
                f'</div>'
            )
        return (
            '<div class="finger-card">'
            f'<div class="metric-label">{escape(title)}</div>'
            f'{"".join(rows)}'
            f'<div class="card-subtitle" style="margin-top:0.8rem;">{escape(note)}</div>'
            '</div>'
        )

    def geo_card(series: pd.Series) -> str:
        clean = series.dropna().astype(str)
        counts = clean.value_counts().head(4)
        total = max(int(len(clean)), 1)
        rows = []
        for idx, (label, count) in enumerate(counts.items(), start=1):
            pct = count / total * 100
            rows.append(
                f'<div class="finger-row" style="grid-template-columns:auto 1fr auto;">'
                f'<div class="finger-value" style="background:#eef2f7;border-radius:6px;padding:0.1rem 0.3rem;">#{idx}</div>'
                f'<div class="finger-label">{escape(label)}</div>'
                f'<div class="finger-value">{int(count)} ({pct:.1f}%)</div>'
                f'</div>'
            )
        return (
            '<div class="finger-card">'
            '<div class="metric-label">Geographical Hotspots</div>'
            f'{"".join(rows)}'
            '<div class="card-subtitle" style="margin-top:0.8rem;">Top client address matches</div>'
            '</div>'
        )

    income_series = active_customers["Income_Class"].replace(
        {"Low": "Low Segment", "Middle": "Middle Segment", "High": "High Segment", "Very High": "Very High Segment"}
    )
    channel_series = active_customers["Purchase_Channel"].replace(
        {"In-Store": "In-Store Channel", "Online": "Online Channel", "Mixed": "Mixed Channel"}
    )

    f1, f2, f3, f4 = st.columns(4)
    f1.markdown(distribution_card("Gender Distribution", active_customers["Gender"], "#635bff", "Excludes null transaction types"), unsafe_allow_html=True)
    f2.markdown(distribution_card("Income Level Distribution", income_series, "#10b981", "Household segment index", ["Low Segment", "Middle Segment", "High Segment", "Very High Segment"]), unsafe_allow_html=True)
    f3.markdown(distribution_card("Acquisition Channel Mix", channel_series, "#f59e0b", "Purchasing medium metrics", ["In-Store Channel", "Online Channel", "Mixed Channel"]), unsafe_allow_html=True)
    f4.markdown(geo_card(active_customers["Location"]), unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dashboard-card">
            <p class="card-title">Multi-Dimensional Scatter Correlation Explorer</p>
            <p class="card-subtitle">Configure custom variable definitions to map correlation ratios across all segments instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    axis_options = {
        "Recency Days": "recencyDays",
        "Monetary Spend ($)": "Purchase_Amount",
        "Orders Count": "Frequency_of_Purchase",
        "Satisfaction": "Customer_Satisfaction",
        "Age": "Age",
    }
    ax1, ax2 = st.columns([1, 1])
    x_label = ax1.selectbox("X-axis variable", list(axis_options.keys()), index=0)
    y_label = ax2.selectbox("Y-axis variable", list(axis_options.keys()), index=1)
    fig = px.scatter(
        customers,
        x=axis_options[x_label],
        y=axis_options[y_label],
        size="Frequency_of_Purchase",
        color="rfmSegment",
        color_discrete_map=SEGMENT_COLORS,
        hover_name="Customer_Name",
        hover_data=["Customer_ID", "Location", "Favorite_Category"],
    )
    fig.update_layout(
        height=430,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        xaxis=dict(gridcolor="#edf2f7"),
        yaxis=dict(gridcolor="#edf2f7"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        head_left, head_search, head_count = st.columns([1.7, 0.85, 0.38])
        head_left.markdown(
            f"""
            <p class="card-title">Cohort Client Directory</p>
            <p class="card-subtitle">Live roster of customers indexed under the <b>{escape(active_segment)}</b> segment.</p>
            """,
            unsafe_allow_html=True,
        )
        query = head_search.text_input("Find name, ID, or city...", key="segment_directory_search", label_visibility="collapsed", placeholder="Find name, ID, or city...")
        directory = active_customers.copy()
        if query:
            q = query.lower()
            directory = directory[
                directory["Customer_ID"].str.lower().str.contains(q, na=False)
                | directory["Customer_Name"].str.lower().str.contains(q, na=False)
                | directory["Location"].str.lower().str.contains(q, na=False)
            ]
        head_count.markdown(
            f"""<div class="directory-count">{len(directory):,} RESULTS</div>""",
            unsafe_allow_html=True,
        )

        def initials(name: str) -> str:
            parts = [p for p in str(name).replace("-", " ").split() if p]
            return "".join(p[0].upper() for p in parts[:2]) or "CL"

        st.markdown(
            '<div class="directory-card">'
            '<div class="directory-grid">'
            '<div class="directory-th">Customer Name & ID</div>'
            '<div class="directory-th">Demographics</div>'
            '<div class="directory-th">Location</div>'
            '<div class="directory-th">Lifetime Spend</div>'
            '<div class="directory-th">Orders Count</div>'
            '<div class="directory-th">Satisfaction</div>'
            '<div class="directory-th">Outreach Dossier</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        for idx, row in enumerate(directory.head(60).itertuples()):
            name = str(row.Customer_Name).strip() or f"Client {row.Customer_ID}"
            education = str(getattr(row, "Education_Level", "") or "Bachelor's")
            marital = str(getattr(row, "Marital_Status", "") or "Single")
            club = '<span class="club-badge">CLUB MEMBER</span>' if bool(row.Loyalty_Member) else ""
            cols = st.columns([2.15, 1.35, 1.2, 1.05, 0.9, 0.9, 0.95])
            cols[0].markdown(
                f'<div class="client-name-wrap"><div class="avatar">{escape(initials(name))}</div><div><b>{escape(name)}</b>{club}<div class="client-sub">ID: {escape(str(row.Customer_ID))}</div></div></div>',
                unsafe_allow_html=True,
            )
            cols[1].markdown(f'<b>{escape(str(row.Gender))}, {int(row.Age)}yrs</b><div class="client-sub">{escape(education)} | {escape(marital)}</div>', unsafe_allow_html=True)
            cols[2].markdown(f'<b>{escape(str(row.Location))}</b><div class="client-sub">Zone Centric</div>', unsafe_allow_html=True)
            cols[3].markdown(f'<b>{escape(fmt_currency(row.Purchase_Amount))}</b>', unsafe_allow_html=True)
            cols[4].markdown(f'{int(row.Frequency_of_Purchase)} orders', unsafe_allow_html=True)
            cols[5].markdown(f'<span style="color:#f59e0b;">&#9733;</span> <b>{row.Customer_Satisfaction:.1f}/10</b>', unsafe_allow_html=True)
            if cols[6].button("Inspect dossier", key=f"dossier_{row.Customer_ID}_{idx}", use_container_width=True):
                show_client_dossier(row._asdict())
            st.markdown("<div style='border-bottom:1px solid #edf2f7;margin:0.35rem 0;'></div>", unsafe_allow_html=True)


def product_recommendations(category: str, sensitivity: str, income: str) -> list[dict]:
    is_sensitive = any(word in str(sensitivity).lower() for word in ["highly", "very", "somewhat"])
    is_high_income = income in {"High", "Very High"}
    products = {
        "Gardening & Outdoors": [("Premium Ergo Pruning Shears", 49), ("Smart Soil Moisture Sensor", 35), ("All-Weather Planter Set", 89)],
        "Food & Beverages": [("Single-Origin Coffee Beans", 28), ("Extra Virgin Olive Oil", 32.5), ("Ceremonial Matcha", 39)],
        "Office Supplies": [("Vegan Leather Desk Pad", 39), ("Memory Foam Seat Cushion", 45), ("Aluminium Fountain Pen Set", 65)],
        "Home Appliances": [("Compact Espresso Machine", 189), ("Silent Room Air Purifier", 120), ("Smart Electric Kettle", 59)],
        "Furniture": [("Solid Oak Side Table", 145), ("Sit-Stand Desk Converter", 199), ("Comfort Accent Cushion", 38)],
        "Books": [("The Pragmatic Strategist", 34), ("Visual Data Design", 48), ("Thinking in Algorithms", 29.99)],
        "Sports & Outdoors": [("Titanium Water Bottle", 55), ("Waterproof Hiking Backpack", 89), ("Non-Slip Yoga Mat", 42)],
        "Mobile Accessories": [("Magnetic Wireless Charger", 69), ("Ultra-Slim Protect Case", 45), ("Pro Earbuds", 129)],
        "Luxury Goods": [("Oud Scented Candle Set", 75), ("Silver Monogram Cufflinks", 110), ("Calfskin Travel Cardholder", 95)],
    }
    defaults = [("Aluminum Desk Organiser", 39.99), ("Thermal Mug", 29.99), ("USB-C Travel Charger", 49.99)]
    picked = products.get(category, defaults)
    rows = []
    for i, (name, base_price) in enumerate(picked[:3]):
        price = base_price * 1.15 if is_high_income else base_price
        discount = [20, 15, 10][i] if is_sensitive else 0
        rows.append(
            {
                "Product": f"Lux {name}" if is_high_income else name,
                "Price": fmt_currency(price),
                "Discount": f"{discount}%",
                "Reason": [
                    f"Strong match for favorite category: {category}.",
                    "Discount-optimized recommendation." if is_sensitive else f"Aligned with {income} income tier.",
                    "Complements the customer's device and purchase pattern.",
                ][i],
            }
        )
    return rows


def render_recommendations(records: pd.DataFrame) -> None:
    customers = customer_table(records)
    if customers.empty:
        st.warning("No customer records match the selected filters.")
        return

    def initials(name: str) -> str:
        parts = [p for p in str(name).split() if p]
        return "".join(p[0].upper() for p in parts[:2]) or "CP"

    with st.container(border=True):
        hleft, hright = st.columns([1.35, 0.8])
        hleft.markdown(
            """
            <p class="card-title">Customer Dossier & Portfolio Metrics</p>
            <p class="card-subtitle">Conduct high-fidelity customer queries to inspect individual purchasing velocity, RFM demographic cohorts, custom marketing personas, and comprehensive lifetime receipts.</p>
            """,
            unsafe_allow_html=True,
        )
        sorted_customers = customers.sort_values(["Customer_Name", "Customer_ID"]).copy()
        customer_lookup = sorted_customers.set_index("Customer_ID")

        st.session_state.setdefault("profile_customer_id", sorted_customers["Customer_ID"].iloc[0])
        if st.session_state["profile_customer_id"] not in set(sorted_customers["Customer_ID"]):
            st.session_state["profile_customer_id"] = sorted_customers["Customer_ID"].iloc[0]
        selected_id = st.session_state["profile_customer_id"]
        selected_preview = customer_lookup.loc[selected_id]
        selected_name = str(selected_preview["Customer_Name"]).strip() or f"Client {selected_id}"
        hright.markdown('<div class="metric-label">Search Customer Profile</div>', unsafe_allow_html=True)
        with hright.popover(f"Browsing: {selected_name}", use_container_width=True):
            query = st.text_input(
                "Find customer",
                key="profile_customer_query",
                placeholder="Find name, ID, or city...",
                label_visibility="collapsed",
            )
            filtered_customers = sorted_customers
            if query:
                q = query.strip().lower()
                filtered_customers = sorted_customers[
                    sorted_customers["Customer_Name"].astype(str).str.lower().str.contains(q, na=False)
                    | sorted_customers["Customer_ID"].astype(str).str.lower().str.contains(q, na=False)
                    | sorted_customers["Location"].astype(str).str.lower().str.contains(q, na=False)
                    | sorted_customers["rfmSegment"].astype(str).str.lower().str.contains(q, na=False)
                ]
            if filtered_customers.empty:
                st.markdown('<div class="profile-picker-empty">No matching customers found.</div>', unsafe_allow_html=True)
            for idx, row in enumerate(filtered_customers.head(90).itertuples()):
                cid = row.Customer_ID
                cname = str(row.Customer_Name).strip() or f"Client {cid}"
                picked = cid == selected_id
                cols = st.columns([1.05, 1.55], vertical_alignment="center")
                with cols[0]:
                    if st.button(cname, key=f"profile_pick_{cid}_{idx}", use_container_width=True, type="primary" if picked else "secondary"):
                        st.session_state["profile_customer_id"] = cid
                        st.rerun()
                cols[1].markdown(
                    f"""
                    <div class="profile-picker-meta">
                        <span class="club-badge" style="margin-left:0;color:#4f46e5;background:#eef2ff;border-color:#ddd6fe;">{escape(str(row.rfmSegment).upper())}</span>
                        <div class="client-sub">ID: {escape(str(cid))}</div>
                        <div class="client-sub">{escape(str(row.Location))} | {int(row.Age)} yrs</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='border-bottom:1px solid #edf2f7;margin:0.15rem 0 0.2rem;'></div>", unsafe_allow_html=True)

    selected = customers[customers["Customer_ID"].eq(selected_id)].iloc[0]
    history = records[records["Customer_ID"].eq(selected_id)].sort_values("Time_of_Purchase", ascending=False)
    name = str(selected["Customer_Name"]).strip() or f"Client {selected_id}"
    aov = selected["Purchase_Amount"] / max(selected["Frequency_of_Purchase"], 1)
    coupon_pct = history["Discount_Used"].mean() * 100 if len(history) else 0
    recs = product_recommendations(selected["Favorite_Category"], selected["Discount_Sensitivity"], selected["Income_Level"])

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    left, right = st.columns([0.92, 1.88])
    with left:
        st.markdown(
            f"""
            <div class="profile-card" style="border-top:6px solid #f59e0b;">
                <div style="display:grid;grid-template-columns:58px 1fr;gap:0.85rem;align-items:center;">
                    <div class="profile-avatar">{escape(initials(name))}</div>
                    <div>
                        <div style="font-weight:900;color:#0f172a;font-size:1rem;">{escape(name)}</div>
                        <div class="client-sub">ID: {escape(str(selected_id))}</div>
                        <span class="club-badge" style="margin-left:0;margin-top:0.35rem;">LOYALTY</span>
                        <span class="club-badge" style="color:#92400e;">{escape(str(selected['rfmSegment']).upper())}</span>
                    </div>
                </div>
                <div style="border-top:1px solid #e2e8f0;margin:1rem 0;padding-top:0.7rem;color:#334155;font-size:0.82rem;">
                    <b>{escape(str(selected['Location']))}</b><br>Age {int(selected['Age'])} ({escape(str(selected['Gender']))})
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.9rem;">
                    <div><div class="metric-label">Marital Status</div><b>{escape(str(selected['Marital_Status']))}</b></div>
                    <div><div class="metric-label">Education</div><b>{escape(str(selected['Education_Level']))}</b></div>
                    <div><div class="metric-label">Income Tier</div><b>{escape(str(selected['Income_Class']))}</b></div>
                    <div><div class="metric-label">Tenure</div><b>{int(selected['Frequency_of_Purchase'])} orders</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="profile-card">
                <div class="metric-label">Behavior & Channels Matrix</div>
                <div class="matrix-row"><span>Favorite Category</span><b>{escape(str(selected['Favorite_Category']))}</b></div>
                <div class="matrix-row"><span>Payment Config</span><b>{escape(str(selected['Payment_Method']))}</b></div>
                <div class="matrix-row"><span>Checkout Device</span><b>{escape(str(selected['Device_Used_for_Shopping']))}</b></div>
                <div class="matrix-row"><span>Preferred Channel</span><b>{escape(str(selected['Purchase_Channel']))}</b></div>
                <div class="matrix-row"><span>Coupon Conversion</span><b>{coupon_pct:.0f}% checkouts</b></div>
                <div class="matrix-row"><span>Return Propensity</span><b style="color:#e11d48;">{float(selected['Return_Rate']) * 100:.1f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Lifetime Value LTV", fmt_currency(selected["Purchase_Amount"]), "Accumulative purchase total"),
            ("Receipt Averages", fmt_currency(aov), f"{int(selected['Frequency_of_Purchase'])} Orders"),
            ("CSAT Index", f"{selected['Customer_Satisfaction']:.1f} /10", "Critical - Satisfied - Champion"),
            ("Loyalty Rating", f"{selected['Product_Rating']:.1f} /10", "Repurchase cadence profile"),
        ]
        for col, (label, value, sub) in zip([m1, m2, m3, m4], metrics):
            col.markdown(
                f'<div class="profile-metric"><div class="metric-label">{escape(label)}</div><div class="metric-value" style="font-size:1.22rem;">{escape(value)}</div><div class="card-subtitle">{escape(sub)}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-card"><p class="card-title">Purchase Ledger & Completed Checkouts</p><p class="card-subtitle">Historical transaction receipts extracted from dataset logs for this profile.</p></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="directory-card"><div class="receipt-grid">'
            '<div class="receipt-th">Purchase Date</div><div class="receipt-th">Line Category</div><div class="receipt-th">Discount Card</div><div class="receipt-th">Score Given</div><div class="receipt-th">Shipping</div><div class="receipt-th">Order Amount</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        rows_html = []
        for row in history.head(5).itertuples():
            discount = "% Used" if row.Discount_Used else "None"
            rows_html.append(
                '<div class="receipt-grid">'
                f'<div class="receipt-cell">{pd.to_datetime(row.Time_of_Purchase).strftime("%b %d, %Y")}</div>'
                f'<div class="receipt-cell"><b>{escape(str(row.Purchase_Category))}</b></div>'
                f'<div class="receipt-cell">{escape(discount)}</div>'
                f'<div class="receipt-cell">{float(row.Product_Rating):.0f} / 5 Rating</div>'
                f'<div class="receipt-cell">{escape(str(row.Shipping_Preference))}</div>'
                f'<div class="receipt-cell"><b>{escape(fmt_currency(row.Purchase_Amount))}</b></div>'
                '</div>'
            )
        st.markdown("".join(rows_html) + "</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        b1.markdown(
            f"""
            <div class="profile-card">
                <div class="metric-label">Target Customer Archetype</div>
                <h4>Pragmatic Utility Buyer</h4>
                <p class="card-subtitle">This user exhibits <b>{'sensitive' if coupon_pct > 40 else 'not sensitive'}</b> discount behavior. Their purchase lifecycle demonstrates a <b>{escape(str(selected['Purchase_Intent']).lower() if 'Purchase_Intent' in selected else 'needs-based')}</b> purchase focus.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        match_rows = []
        for item in recs:
            match_rows.append(
                f'<div class="match-card"><div style="display:flex;justify-content:space-between;gap:0.8rem;"><b>{escape(item["Product"])}</b><b>{escape(item["Price"])}</b></div><div class="card-subtitle">{escape(item["Reason"])}</div></div>'
            )
        b2.markdown(
            '<div class="profile-card"><div class="metric-label" style="color:#f97316;">Personalized Matches & Affinity</div>'
            + "".join(match_rows)
            + '</div>',
            unsafe_allow_html=True,
        )


def render_trends(records: pd.DataFrame) -> tuple[list[str], list[str]]:
    monthly = records.groupby("Purchase_Month_Name", as_index=False).agg(revenue=("Purchase_Amount", "sum"), orders=("Purchase_Amount", "size"))
    monthly["order"] = monthly["Purchase_Month_Name"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    monthly = monthly.sort_values("order")
    monthly["aov"] = monthly["revenue"] / monthly["orders"].replace(0, np.nan)
    peak_months = monthly.sort_values("revenue", ascending=False)["Purchase_Month_Name"].head(2).tolist()

    weekday = records.groupby("Purchase_DayOfWeek", as_index=False).agg(revenue=("Purchase_Amount", "sum"), orders=("Purchase_Amount", "size"))
    weekday["order"] = weekday["Purchase_DayOfWeek"].map({d: i for i, d in enumerate(DAY_ORDER)})
    weekday = weekday.sort_values("order")
    peak_days = weekday.sort_values("revenue", ascending=False)["Purchase_DayOfWeek"].head(2).tolist()

    st.markdown(
        """
        <div class="trend-hero">
            <div class="trend-spark">✣</div>
            <div>
                <div style="color:#4f46e5;font-weight:900;">Seasonality Clustering & Performance Heatmap</div>
                <div class="card-subtitle" style="margin-top:0.25rem;">Identify optimal transaction periods using our multi-dimensional grid heatmap. Toggle indicators to scale color-density metrics across Weeks of the Month vs Months to uncover cohort habits.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    trend_records = records.copy()
    trend_records["Week_Index"] = ((trend_records["Time_of_Purchase"].dt.day.sub(1) // 7) + 1).clip(upper=5)
    matrix = trend_records.groupby(["Week_Index", "Purchase_Month_Name"], as_index=False).agg(
        revenue=("Purchase_Amount", "sum"),
        orders=("Purchase_Amount", "size"),
        aov=("Purchase_Amount", "mean"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
        loyalty_ratio=("Loyalty_Member", "mean"),
        promo_ratio=("Discount_Used", "mean"),
    )
    all_cells = pd.MultiIndex.from_product([range(1, 6), MONTH_ORDER], names=["Week_Index", "Purchase_Month_Name"]).to_frame(index=False)
    matrix = all_cells.merge(matrix, how="left", on=["Week_Index", "Purchase_Month_Name"]).fillna(
        {"revenue": 0, "orders": 0, "aov": 0, "avg_satisfaction": 0, "loyalty_ratio": 0, "promo_ratio": 0}
    )
    mode_options = ["Total Revenue", "Purchase Counts", "Ticket AOV"]
    query_metric = st.query_params.get("trend_metric", "")
    if isinstance(query_metric, list):
        query_metric = query_metric[0] if query_metric else ""
    initial_metric = query_metric if query_metric in mode_options else st.session_state.get("trend_metric_mode", "Total Revenue")
    if "trend_metric_picker" not in st.session_state or st.session_state["trend_metric_picker"] not in mode_options:
        st.session_state["trend_metric_picker"] = initial_metric

    def current_trend_cell() -> tuple[int, str]:
        if "trend_selected_cell" in st.session_state:
            stored = st.session_state["trend_selected_cell"]
            if isinstance(stored, tuple) and len(stored) == 2 and stored[1] in MONTH_ORDER:
                return int(stored[0]), stored[1]
        raw = st.query_params.get("trend_cell", "")
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        if raw and "__" in str(raw):
            month, week = str(raw).split("__", 1)
            if month in MONTH_ORDER and week.isdigit():
                return int(week), month
        return 1, MONTH_ORDER[0]

    selected_week, selected_month = current_trend_cell()
    selected_cell = matrix[(matrix["Week_Index"].eq(selected_week)) & (matrix["Purchase_Month_Name"].eq(selected_month))].iloc[0]
    heatmap_card = st.container(border=True)
    header_left, header_right = heatmap_card.columns([1.25, 0.95], vertical_alignment="center")
    header_left.markdown(
        """
        <span class="metric-label" style="background:#f1f5f9;border-radius:6px;padding:0.32rem 0.55rem;color:#334155;">Heat Vector Index</span>
        <span class="card-subtitle" style="margin-left:0.55rem;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;">5 Weeks x 12 Months Matrix</span>
        <p class="card-title" style="margin-top:0.7rem;">Weekly & Monthly Cohort Performance Matrix</p>
        <p class="card-subtitle">Discover underlying distribution velocities by cross-referencing customer checkout rates.</p>
        """,
        unsafe_allow_html=True,
    )
    metric_mode = header_right.segmented_control(
        "Heatmap metric",
        mode_options,
        key="trend_metric_picker",
        label_visibility="collapsed",
    )
    st.session_state["trend_metric_mode"] = metric_mode
    metric_col = {"Total Revenue": "revenue", "Purchase Counts": "orders", "Ticket AOV": "aov"}[metric_mode]
    max_metric = max(float(matrix[metric_col].max()), 1.0)
    min_metric = float(matrix[metric_col].min())
    metric_span = max(max_metric - min_metric, 1.0)
    heatmap_palette = {
        "Total Revenue": (30, 27, 207),
        "Purchase Counts": (16, 185, 129),
        "Ticket AOV": (124, 58, 237),
    }
    heatmap_outline = {
        "Total Revenue": "#635bff",
        "Purchase Counts": "#10b981",
        "Ticket AOV": "#7c3aed",
    }
    heatmap_rgb = heatmap_palette[metric_mode]
    heatmap_active = heatmap_outline[metric_mode]

    head_html = [
        '<div style="border-top:1px solid #e2e8f0;margin:0.6rem 0 1rem;"></div>',
        '<div class="trend-grid">',
        '<div class="trend-head">Week</div>',
    ]
    head_html.extend(f'<div class="trend-head">{month.upper()}</div>' for month in MONTH_ORDER)
    for week in range(1, 6):
        head_html.append(f'<div class="trend-week">Week {week}</div>')
        for month in MONTH_ORDER:
            row = matrix[(matrix["Week_Index"].eq(week)) & (matrix["Purchase_Month_Name"].eq(month))].iloc[0]
            intensity = (float(row[metric_col]) - min_metric) / metric_span
            alpha = 0.08 + (intensity ** 1.75 * 0.92)
            bg = f"rgba({heatmap_rgb[0]}, {heatmap_rgb[1]}, {heatmap_rgb[2]}, {alpha:.3f})"
            is_active = week == selected_week and month == selected_month
            value = fmt_currency(float(row["revenue"])).replace(".00", "") if metric_mode == "Total Revenue" else (f"{int(row['orders'])} trx" if metric_mode == "Purchase Counts" else fmt_currency(float(row["aov"])))
            sub = f"{int(row['orders'])} trx"
            head_html.append(
                f'<a class="trend-cell {"active" if is_active else ""}" target="_self" href="?page=Sales%20Trend%20Analysis&trend_cell={month}__{week}" style="background:{bg};">'
                f'<div><div class="trend-cell-main">{escape(value)}</div><div class="trend-cell-sub">{escape(sub)}</div></div></a>'
            )
    head_html.extend(
        [
            '</div>',
            '<div class="trend-scale"><span>Low Intensity ○</span><span>Active Gradient Scale:<span class="gradient-pill"></span></span><span style="color:#4338ca;">● Peak Volume Range</span></div>',
        ]
    )
    button_css = [
        '<style>',
        '.trend-week-native{height:60px;border:1px solid #e2e8f0;background:#f8fafc;border-radius:7px;display:grid;place-items:center;color:#0f172a;font-weight:900;font-size:0.68rem;}',
        '.trend-head-native{height:38px;border:1px solid #e2e8f0;background:#f8fafc;border-radius:7px;display:grid;place-items:center;color:#64748b;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-weight:900;font-size:0.62rem;}',
    ]
    for week in range(1, 6):
        for month in MONTH_ORDER:
            row = matrix[(matrix["Week_Index"].eq(week)) & (matrix["Purchase_Month_Name"].eq(month))].iloc[0]
            intensity = (float(row[metric_col]) - min_metric) / metric_span
            alpha = 0.08 + (intensity ** 1.75 * 0.92)
            bg = f"rgba({heatmap_rgb[0]}, {heatmap_rgb[1]}, {heatmap_rgb[2]}, {alpha:.3f})"
            key = f"trend_cell_{month}_{week}"
            active_style = f"box-shadow:0 0 0 2px {heatmap_active},0 10px 24px rgba({heatmap_rgb[0]},{heatmap_rgb[1]},{heatmap_rgb[2]},0.24) !important;" if week == selected_week and month == selected_month else ""
            text_color = "#0f172a" if intensity < 0.24 else "#ffffff"
            button_css.append(
                f'.st-key-{key} button{{height:60px;width:100%;border-radius:9px !important;border:1px solid rgba(255,255,255,0.8) !important;background:{bg} !important;color:#fff !important;{active_style}}}'
                f'.st-key-{key} button p{{color:{text_color} !important;font-weight:950 !important;font-size:0.74rem !important;line-height:1.15 !important;white-space:pre-line !important;text-align:center !important;}}'
            )
    button_css.append('</style>')
    heatmap_card.markdown("".join(button_css), unsafe_allow_html=True)
    heatmap_card.markdown('<div style="border-top:1px solid #e2e8f0;margin:0.6rem 0 1rem;"></div>', unsafe_allow_html=True)

    header_cols = heatmap_card.columns([0.9] + [1] * 12, gap="small")
    header_cols[0].markdown('<div class="trend-head-native">Week</div>', unsafe_allow_html=True)
    for col, month in zip(header_cols[1:], MONTH_ORDER):
        col.markdown(f'<div class="trend-head-native">{month.upper()}</div>', unsafe_allow_html=True)

    for week in range(1, 6):
        row_cols = heatmap_card.columns([0.9] + [1] * 12, gap="small")
        row_cols[0].markdown(f'<div class="trend-week-native">Week {week}</div>', unsafe_allow_html=True)
        for col, month in zip(row_cols[1:], MONTH_ORDER):
            row = matrix[(matrix["Week_Index"].eq(week)) & (matrix["Purchase_Month_Name"].eq(month))].iloc[0]
            value = fmt_currency(float(row["revenue"])).replace(".00", "") if metric_mode == "Total Revenue" else (f"{int(row['orders'])} trx" if metric_mode == "Purchase Counts" else fmt_currency(float(row["aov"])))
            if col.button(f"{value}\n{int(row['orders'])} trx", key=f"trend_cell_{month}_{week}", use_container_width=True):
                st.session_state["trend_selected_cell"] = (week, month)
                st.rerun()

    heatmap_card.markdown(
        f'<div class="trend-scale"><span>Low Intensity o</span><span>Active Gradient Scale:<span class="gradient-pill" style="background:linear-gradient(90deg,rgba({heatmap_rgb[0]},{heatmap_rgb[1]},{heatmap_rgb[2]},0.08),rgba({heatmap_rgb[0]},{heatmap_rgb[1]},{heatmap_rgb[2]},0.28),rgba({heatmap_rgb[0]},{heatmap_rgb[1]},{heatmap_rgb[2]},0.68),rgb({heatmap_rgb[0]},{heatmap_rgb[1]},{heatmap_rgb[2]}));"></span></span><span style="color:{heatmap_active};">* Peak Volume Range</span></div>',
        unsafe_allow_html=True,
    )

    cell_records = trend_records[
        trend_records["Week_Index"].eq(selected_week) & trend_records["Purchase_Month_Name"].eq(selected_month)
    ]
    if cell_records.empty:
        cell_records = trend_records[trend_records["Purchase_Month_Name"].eq(selected_month)]
    gender_normalized = cell_records["Gender"].astype(str).str.strip().str.title()
    female_pct = float(gender_normalized.eq("Female").mean() * 100) if len(cell_records) else 0
    male_pct = max(0.0, 100.0 - female_pct)
    loyalty_pct = float(cell_records["Loyalty_Member"].mean() * 100) if len(cell_records) else 0
    promo_pct = float(cell_records["Discount_Used"].mean() * 100) if len(cell_records) else 0
    satisfaction = float(cell_records["Customer_Satisfaction"].mean()) if len(cell_records) else 0
    top_device = str(cell_records["Device_Used_for_Shopping"].mode().iloc[0]) if len(cell_records) else "Desktop"
    top_cats = cell_records["Purchase_Category"].astype(str).value_counts().head(3)
    top_total = max(float(len(cell_records)), 1.0)
    vertical_rows = []
    for cat, value in top_cats.items():
        pct = float(value) / top_total * 100
        vertical_rows.append(
            f'<div style="display:flex;justify-content:space-between;font-weight:900;font-size:0.78rem;color:#334155;"><span>{escape(str(cat))}</span><span>{pct:.0f}%</span></div>'
            f'<div class="trend-progress"><span style="width:{pct:.1f}%;background:#4f46e5;"></span></div>'
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="trend-profile">
            <div>
                <div class="metric-label" style="color:#635bff;">Selected Segment Profile</div>
                <h3 style="margin:0.55rem 0 0.4rem;color:#0f172a;">{escape(selected_month)} — Week {selected_week}</h3>
                <p class="card-subtitle">Extracted buyer behaviors matching this exact cross-tab. Tracks shopper aesthetics during this seasonal intersection.</p>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;margin-top:0.9rem;">
                    <div class="trend-mini-box"><div class="metric-label">Friction Total</div><div class="metric-value" style="font-size:1.15rem;">{fmt_currency(float(selected_cell['revenue'])).replace('.00', '')}</div></div>
                    <div class="trend-mini-box"><div class="metric-label">Cohort Count</div><div class="metric-value" style="font-size:1.15rem;">{int(selected_cell['orders'])} trx</div></div>
                </div>
            </div>
            <div>
                <div class="metric-label">Aesthetic & Gender Splits</div>
                <div style="display:flex;justify-content:space-between;margin-top:0.9rem;font-size:0.78rem;"><span>Gender (Female / Male)</span><b>{female_pct:.0f}% / {male_pct:.0f}%</b></div>
                <div class="trend-progress"><span style="width:{female_pct:.1f}%;background:#ec4899;"></span></div>
                <div style="display:flex;justify-content:space-between;font-size:0.78rem;"><span>Loyalty Member Share</span><b>{loyalty_pct:.0f}%</b></div>
                <div class="trend-progress"><span style="width:{loyalty_pct:.1f}%;background:#10b981;"></span></div>
                <div class="trend-mini-box" style="display:flex;justify-content:space-between;align-items:center;"><span style="color:#64748b;">▯ Shopping device:</span><b>{escape(top_device)}</b></div>
            </div>
            <div>
                <div class="metric-label">Promotion & Satisfaction Benchmarks</div>
                <div class="trend-mini-box" style="margin-top:0.9rem;"><div class="metric-label">Satisfaction Index</div><span style="color:#f97316;font-weight:900;font-size:1.25rem;">{satisfaction:.1f}</span> <span class="card-subtitle">/ 10 Avg Score</span></div>
                <div style="display:flex;justify-content:space-between;margin-top:0.85rem;font-size:0.78rem;"><span>Promo Coupon Usage Rate</span><b>{promo_pct:.0f}%</b></div>
                <div class="trend-progress"><span style="width:{promo_pct:.1f}%;background:#7c3aed;"></span></div>
            </div>
            <div>
                <div class="metric-label">Top Selling Verticals</div>
                <div style="margin-top:0.9rem;">{''.join(vertical_rows) if vertical_rows else '<div class="card-subtitle">No category activity in this cell.</div>'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    chart_col, audit_col = st.columns([1.75, 0.9])
    with chart_col:
        st.markdown(
            """
            <div class="dashboard-card">
                <p class="card-title">Convert Rhythms by Weekday Distribution</p>
                <p class="card-subtitle">Analyze raw revenue velocities by weekday to balance dynamic marketing and stock budgets.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = px.bar(weekday, x="Purchase_DayOfWeek", y="revenue", color_discrete_sequence=["#4f46e5"])
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br><span style='color:#4f46e5;'>Cohort value : %{y:$,.2f}</span><extra></extra>",
            marker_line_width=0,
        )
        fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=15, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title=None,
            yaxis_title=None,
            xaxis=dict(categoryorder="array", categoryarray=DAY_ORDER, gridcolor="#edf2f7"),
            yaxis=dict(gridcolor="#edf2f7"),
            showlegend=False,
            hoverlabel=dict(
                bgcolor="#0f172a",
                bordercolor="#0f172a",
                font=dict(color="#ffffff", size=12),
                align="left",
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with audit_col:
        st.markdown(
            f"""
            <div class="dashboard-card" style="min-height:360px;">
                <p class="card-title">Seasonality Audit Report</p>
                <p class="card-subtitle">Broad chronological summaries outlining buyer peaks recorded in our transaction files.</p>
                <div class="metric-label" style="margin-top:1rem;">Peak Revenue Quarters</div>
                <div><span class="audit-pill">🏆 {escape(peak_months[0] if peak_months else 'Peak')} Peak</span><span class="audit-pill">🏆 {escape(peak_months[1] if len(peak_months) > 1 else 'Next')} Peak</span></div>
                <p class="card-subtitle">Slower purchasing velocities historically standard in: {escape(', '.join(monthly.sort_values('revenue')['Purchase_Month_Name'].head(2).tolist()))}.</p>
                <div style="border-top:1px solid #e2e8f0;margin:1rem 0;"></div>
                <div class="metric-label">Weekly Conversion Paces</div>
                <div><span class="audit-pill" style="color:#4f46e5;background:#eef2ff;border-color:#ddd6fe;">⚡ {escape(peak_days[0] if peak_days else 'Top Day')} Volume</span><span class="audit-pill" style="color:#4f46e5;background:#eef2ff;border-color:#ddd6fe;">⚡ {escape(peak_days[1] if len(peak_days) > 1 else 'Next Day')} Volume</span></div>
                <p class="card-subtitle">Weekends present deeper evaluation rates, while midweeks act as impulse purchasing peaks.</p>
                <div class="audit-note" style="margin-top:1rem;"><div class="metric-label" style="color:#4f46e5;">Cross-Analytical Recommendation</div>Cross-reference heat patterns with active promo intervals. Scheduling discounts during standard weekday valleys assists in lifting baseline velocities.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return peak_months, peak_days


@st.cache_data(show_spinner=False)
def generate_insights(records: pd.DataFrame, peak_months: list[str], peak_days: list[str]) -> dict:
    customers = customer_table(records)
    total_revenue = records["Purchase_Amount"].sum()
    total_customers = records["Customer_ID"].nunique()
    aov = records["Purchase_Amount"].mean() if len(records) else 0
    avg_satisfaction = records["Customer_Satisfaction"].mean() if len(records) else 0
    segments = customers["rfmSegment"].value_counts(normalize=True).mul(100).round(1).to_dict()
    categories = records.groupby("Purchase_Category")["Purchase_Amount"].sum().sort_values(ascending=False)
    top_category = categories.index[0] if len(categories) else "the leading category"
    second_category = categories.index[1] if len(categories) > 1 else None
    category_share = (categories / max(categories.sum(), 1) * 100).round(1)
    top_category_share = float(category_share.iloc[0]) if len(category_share) else 0
    top_five_share = float(category_share.head(5).sum()) if len(category_share) else 0
    top_device = records["Device_Used_for_Shopping"].mode().iloc[0] if "Device_Used_for_Shopping" in records and len(records) else "Smartphones and Tablets"
    second_device = records["Device_Used_for_Shopping"].value_counts().index[1] if "Device_Used_for_Shopping" in records and records["Device_Used_for_Shopping"].nunique() > 1 else "Tablets"
    loyalty_ratio = customers["Loyalty_Member"].mean() * 100 if "Loyalty_Member" in customers and len(customers) else 0
    high_value_pct = float(segments.get("High Value Customers", 0))
    loyal_pct = float(segments.get("Loyal Customers", 0))
    frequent_pct = float(segments.get("Frequent Buyers", 0))
    at_risk_pct = float(segments.get("At Risk Customers", 0))
    lost_pct = float(segments.get("Lost Customers", 0))
    low_rating_pct = float((records["Product_Rating"].lt(4).mean() * 100) if "Product_Rating" in records and len(records) else 0)
    weak_months = records.groupby("Purchase_Month_Name")["Purchase_Amount"].sum().sort_values().head(2).index.tolist()
    weak_days = records.groupby("Purchase_DayOfWeek")["Purchase_Amount"].sum().sort_values().head(2).index.tolist()

    fallback = {
        "Customer Insights": (
            f"Our customer base exhibits a premium purchasing profile, demonstrated by an Average Order Value (AOV) of "
            f"{fmt_currency(aov)} across {len(records):,} transactions and {total_customers:,} active customers, generating "
            f"{fmt_currency(total_revenue)} in total revenue. Demographically, the strongest behavioral signals come from "
            f"tech-adapted cohorts using {top_device} and {second_device}, with a loyalty penetration of {loyalty_ratio:.1f}%. "
            f"However, the average product satisfaction rating of {avg_satisfaction:.1f}/10 reveals a possible post-purchase "
            "experience disconnect. This suggests that retention depends less on acquisition volume and more on improving "
            "fulfillment quality, product clarity, and high-touch customer reassurance after checkout."
        ),
        "Sales Insights": (
            f"Sales data reveals clear temporal peaks that should dictate marketing spend allocation. Seasonally, revenue spikes "
            f"during {', '.join(peak_months) or 'the strongest seasonal windows'}, while slower purchasing velocities historically "
            f"appear in {', '.join(weak_months) or 'lower-demand months'}. Weekly behavior is led by {', '.join(peak_days) or 'the strongest weekdays'}, "
            f"while {', '.join(weak_days) or 'lower-performing weekdays'} represent softer conversion periods. This pattern favors "
            "front-loading premium offers near peak windows while using lighter discount nudges, reminders, and email campaigns "
            "to lift standard weekday valleys."
        ),
        "Product Insights": (
            f"Our product portfolio is heavily fragmented, with the top five categories representing {top_five_share:.1f}% of total "
            f"revenue. {top_category} leads the catalog at {top_category_share:.1f}%"
            + (f", followed by {second_category}. " if second_category else ". ")
            + f"At the same time, {low_rating_pct:.1f}% of transaction ratings fall below 4/5, indicating that assortment breadth may "
            "be diluting product confidence. The immediate action is to protect high-margin categories, bundle complementary items, "
            "audit weak SKUs, and improve product descriptions so cross-sell recommendations do not amplify low-satisfaction items."
        ),
        "Segment Insights": (
            f"Our RFM segmentation reveals a concentrated but unstable customer structure: {high_value_pct:.1f}% of the database is "
            f"classified as High Value Customers, while Loyal Customers ({loyal_pct:.1f}%) and Frequent Buyers ({frequent_pct:.1f}%) "
            f"remain comparatively thin. At the same time, At Risk and Lost Customers account for {at_risk_pct + lost_pct:.1f}% of the "
            "base, which means the business has both high-value acquisition strength and visible churn pressure. High-value users "
            "should receive VIP service and early access, while at-risk groups need targeted win-back campaigns, satisfaction recovery, "
            "and discount triggers calibrated to their purchase history."
        ),
        "Recommendation Insights": (
            f"To maximize the {fmt_currency(aov)} AOV and counteract satisfaction drag, the recommendation engine should pivot toward "
            "quality-first personalization. First, pair high-margin categories with complementary products during peak months "
            f"({', '.join(peak_months) or 'seasonal highs'}), especially around {top_category}. Second, suppress products with weak ratings "
            "from automated add-ons until catalog issues are resolved. Third, use income level, discount sensitivity, favorite category, "
            "and device behavior to determine whether a customer should see premium bundles, value bundles, or replenishment-style offers."
        ),
    }

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return fallback

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an executive e-commerce analytics advisor. Return only strict JSON with keys "
                    "customerInsights, salesInsights, productInsights, segmentInsights, recommendationInsights. "
                    "Each value must be one detailed strategic advisor paragraph of 90-130 words. "
                    "Use concrete metrics, risks, and recommended actions; do not be generic.",
                ),
                ("human", "{analytics_payload}"),
            ]
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.35,
        )
        chain = prompt | llm | StrOutputParser()
        analytics_payload = json.dumps(
            {
                "total_revenue": round(float(total_revenue), 2),
                "total_customers": int(total_customers),
                "total_transactions": int(len(records)),
                "aov": round(float(aov), 2),
                "avg_satisfaction": round(float(avg_satisfaction), 2),
                "segments": segments,
                "top_categories": categories.head(8).to_dict(),
                "peak_months": peak_months,
                "peak_days": peak_days,
                "weak_months": weak_months,
                "weak_days": weak_days,
                "top_device": str(top_device),
                "second_device": str(second_device),
                "loyalty_ratio": round(float(loyalty_ratio), 1),
                "low_rating_pct": round(float(low_rating_pct), 1),
            }
        )
        ai_text = chain.invoke({"analytics_payload": analytics_payload}).strip()
        if ai_text.startswith("```"):
            ai_text = ai_text.strip("`").replace("json\n", "", 1).strip()
        parsed = json.loads(ai_text)
        return {
            "Customer Insights": parsed.get("customerInsights", fallback["Customer Insights"]),
            "Sales Insights": parsed.get("salesInsights", fallback["Sales Insights"]),
            "Product Insights": parsed.get("productInsights", fallback["Product Insights"]),
            "Segment Insights": parsed.get("segmentInsights", fallback["Segment Insights"]),
            "Recommendation Insights": parsed.get("recommendationInsights", fallback["Recommendation Insights"]),
        }
    except Exception:
        return fallback


def ai_runtime_status() -> tuple[str, str, str]:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return "Local Fallback", "GEMINI_API_KEY not configured", "#64748b"
    if importlib.util.find_spec("langchain_google_genai") is None:
        return "Local Fallback", "LangChain Google GenAI package not installed", "#d97706"
    return "Gemini + LangChain Active", "Live generative AI enabled", "#10b981"


def render_ai_insights(records: pd.DataFrame, peak_months: list[str], peak_days: list[str]) -> None:
    title_map = [
        ("Customer Insights", "Customer Portfolios & Behavior", "01", "violet"),
        ("Sales Insights", "Sales Seasonality & Demand Peaks", "02", "pink"),
        ("Product Insights", "Product Assortment Optimization", "03", "green"),
        ("Segment Insights", "Cohort Activation Strategies (RFM)", "04", "amber"),
        ("Recommendation Insights", "AI Personal Recommendation Actions & Cross-Selling", "05", "violet"),
    ]

    engine_label, engine_detail, engine_color = ai_runtime_status()
    with st.container(border=True):
        hero_left, hero_right = st.columns([1.55, 0.52], vertical_alignment="center")
        hero_left.markdown(
            f"""
            <div class="ai-board-title">AI Executive Advisor Board</div>
            <div class="card-subtitle">Strategic marketing blueprints generated live via LangChain-orchestrated Gemini semantic models.</div>
            <div style="display:inline-flex;align-items:center;gap:0.45rem;margin-top:0.7rem;border:1px solid #dbe5f0;border-radius:999px;background:#ffffff;padding:0.32rem 0.62rem;font-size:0.74rem;font-weight:800;color:#334155;">
                <span style="width:0.48rem;height:0.48rem;border-radius:999px;background:{engine_color};display:inline-block;"></span>
                {escape(engine_label)}
                <span style="color:#94a3b8;font-weight:700;">{escape(engine_detail)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if hero_right.button("Re-evaluate Strategic Models", key="ai_revaluate_btn", use_container_width=True):
            generate_insights.clear()
            st.session_state["ai_evaluating"] = True
            st.rerun()
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    if st.session_state.get("ai_evaluating", False):
        st.markdown(
            """
            <div class="ai-loading-card">
                <div>
                    <div class="ai-loader">+</div>
                    <div class="ai-loading-title">Reviewing RFM demographic correlation matrices...</div>
                    <div class="ai-loading-subtitle">Synthesizing category revenue shares and peak seasonality<br>coordinates to model recommendations.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(1.8)
        st.session_state["ai_evaluating"] = False
        generate_insights.clear()
        st.rerun()

    insights = generate_insights(records, peak_months, peak_days)

    def ai_card(source_key: str, display_title: str, index: str, color: str) -> str:
        body = insights.get(source_key, "")
        return (
            '<div class="ai-insight-card">'
            '<div class="ai-card-head">'
            f'<div class="ai-index {escape(color)}">{escape(index)}</div>'
            f'<div class="ai-insight-title">{escape(display_title)}</div>'
            '</div>'
            f'<div class="ai-insight-body">{escape(str(body))}</div>'
            '</div>'
        )

    row1 = st.columns(2)
    row1[0].markdown(ai_card(*title_map[0]), unsafe_allow_html=True)
    row1[1].markdown(ai_card(*title_map[1]), unsafe_allow_html=True)
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    row2 = st.columns(2)
    row2[0].markdown(ai_card(*title_map[2]), unsafe_allow_html=True)
    row2[1].markdown(ai_card(*title_map[3]), unsafe_allow_html=True)
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    st.markdown(ai_card(*title_map[4]), unsafe_allow_html=True)


def main() -> None:
    inject_theme()
    st.sidebar.markdown(
        """
        <div class="side-brand">
            <div class="side-logo">[]</div>
            <div>
                <div style="font-weight:900;color:#fff;font-size:0.95rem;">E-Commerce Portfolio</div>
                <div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.68rem;font-weight:800;letter-spacing:0.14em;color:#93c5fd;">ANALYTICS & AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_df = load_raw_dataset()
    records = normalize_dataset(raw_df)
    if records.empty:
        st.stop()

    page = render_sidebar_nav()
    filtered = render_filter_header(records, page)
    render_sidebar_summary(records, filtered)
    if filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
        st.stop()

    if page == "Executive Dashboard":
        render_dashboard(filtered, records)
    elif page == "Customer Segmentation":
        render_segmentation(filtered)
    elif page == "Customer Profiles":
        render_recommendations(filtered)
    elif page == "Sales Trend Analysis":
        render_trends(filtered)
    else:
        monthly = filtered.groupby("Purchase_Month_Name")["Purchase_Amount"].sum().sort_values(ascending=False)
        weekday = filtered.groupby("Purchase_DayOfWeek")["Purchase_Amount"].sum().sort_values(ascending=False)
        peak_months = monthly.head(2).index.tolist()
        peak_days = weekday.head(2).index.tolist()
        render_ai_insights(filtered, peak_months, peak_days)


if __name__ == "__main__":
    main()
