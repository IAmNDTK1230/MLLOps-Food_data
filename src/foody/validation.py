"""Strict source reading: no encoding repair, inferred labels or date guessing."""

import csv
import io
from datetime import datetime

import pandas as pd

from src.foody.schema import MAPPING

DATE_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M")


def load_source(path):
    text = path.read_bytes().decode("utf-8-sig", errors="strict")
    header = next(csv.reader(io.StringIO(text)))
    if len(header) != len(set(header)) or set(header) != set(MAPPING):
        raise ValueError(f"Expected the five first-party Foody fields; found {header}")
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def parse_date(value):
    for pattern in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(value.strip(), pattern))
        except ValueError:
            pass
    return pd.NaT


def validate_processed(frame):
    if len(frame) != frame.source_row.nunique():
        raise ValueError("Source-row provenance must be unique")
    valid = frame.loc[frame.valid_for_ai]
    if valid.text.eq("").any() or valid.text.str.contains("\ufffd", regex=False).any():
        raise ValueError("Invalid text marked eligible")
    if valid.restaurant_key.isna().any():
        raise ValueError("Missing restaurant key marked eligible")
    if "source_classification" in frame or "Classification" in frame:
        raise ValueError("First-party crawl has no Classification")
