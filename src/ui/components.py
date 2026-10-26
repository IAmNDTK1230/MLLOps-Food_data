"""Reusable, escaped presentation components; no business calculations."""

from html import escape

import streamlit as st


def header(title, subtitle, badges=()):
    pills = "".join(f'<span class="foody-badge">{escape(str(label))}</span>' for label in badges)
    st.markdown(
        '<div class="foody-hero"><div class="foody-eyebrow">Foody Intelligence Platform</div>'
        f'<h1>{escape(title)}</h1><div class="foody-muted">{escape(subtitle)}</div>{pills}</div>',
        unsafe_allow_html=True,
    )


def badge(label, tone="default"):
    tone = tone if tone in {"good", "warn", "muted"} else ""
    st.markdown(f'<span class="foody-badge {tone}">{escape(str(label))}</span>',
                unsafe_allow_html=True)


def badge_row(items):
    pills = "".join(
        f'<span class="foody-badge {tone if tone in {"good", "warn", "muted"} else ""}">'
        f'{escape(str(label))}</span>' for label, tone in items
    )
    st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)


def empty_state(title, detail=""):
    st.markdown(f'<div class="foody-empty"><strong>{escape(title)}</strong>'
                f'{escape(detail)}</div>', unsafe_allow_html=True)


def answer_card(title, answer, label=None, tone="default"):
    pill = (f'<span class="foody-badge {tone if tone in {"good", "warn", "muted"} else ""}">'
            f'{escape(label)}</span>') if label else ""
    st.markdown(f'<div class="foody-card foody-answer"><h3>{escape(title)}</h3>{pill}'
                f'<p>{escape(str(answer))}</p></div>', unsafe_allow_html=True)


def evidence_card(reference, text, metadata=""):
    st.markdown(f'<div class="foody-card"><h3>{escape(str(reference))}</h3>'
                f'<p>{escape(str(text))}</p><div class="foody-muted">{escape(str(metadata))}</div>'
                '</div>', unsafe_allow_html=True)


def money(value):
    return f"{float(value or 0):,.0f} ₫"


def percent(value):
    return "—" if value is None else f"{float(value):.2%}"


def text_sentiment_label(prediction):
    """The explorer badge uses the saved text-model class, never the rating."""
    if prediction not in {"negative", "neutral", "positive"}:
        raise ValueError("Unsupported sentiment model class")
    return f"Text sentiment: {prediction.title()}"
