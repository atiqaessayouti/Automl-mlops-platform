from datetime import datetime, timezone
from pathlib import Path
import re

import streamlit as st


def apply_styles() -> None:
    stylesheet = Path(__file__).with_name("theme.css")
    if not stylesheet.is_file():
        st.error(f"La feuille de style de l’interface est introuvable : {stylesheet}")
        return
    st.markdown(
        f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, icon: str) -> None:
    with st.container(border=True, key="page-hero"):
        st.caption("AUTOML STUDIO  /  INTELLIGENCE ARTIFICIELLE & MLOPS")
        st.title(title, icon=icon)
        st.markdown(subtitle)


def metric_card(label: str, value: str, detail: str, key: str, icon: str) -> None:
    safe_key = re.sub(r"[^a-z0-9_-]+", "-", key.lower()).strip("-")
    with st.container(border=True, key=f"kpi-{safe_key}"):
        st.caption(f":material/{icon}: {label}")
        st.metric(label, value, label_visibility="collapsed")
        st.caption(detail)


def status_card(name: str, healthy: bool, detail: str, key: str) -> None:
    safe_key = re.sub(r"[^a-z0-9_-]+", "-", key.lower()).strip("-")
    with st.container(border=True, key=f"panel-{safe_key}"):
        st.markdown(f"#### {name}")
        if healthy:
            st.badge("Opérationnel", color="green", icon=":material/check_circle:")
        else:
            st.badge("Indisponible", color="red", icon=":material/error:")
        st.caption(detail)


def format_timestamp(value: int | float | None) -> str:
    if value is None:
        return "Non renseigné"
    timestamp = value / 1000 if value > 10_000_000_000 else value
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%d/%m/%Y à %H:%M UTC")


def format_count(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def format_decimal(value: float, precision: int = 2) -> str:
    return f"{value:.{precision}f}".replace(".", ",")


def format_percent(value: float, precision: int = 1) -> str:
    return f"{format_decimal(value * 100, precision)} %"
