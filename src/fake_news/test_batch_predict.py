
"""Basic tests for batch CSV validation."""

from io import BytesIO

import pytest

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAKE_NEWS_DIR = PROJECT_ROOT / "src" / "fake_news"

if str(FAKE_NEWS_DIR) not in sys.path:
    sys.path.insert(0, str(FAKE_NEWS_DIR))

from batch_predict import load_headline_csv


def test_valid_csv():
    file = BytesIO(b"headline\nExample headline\n")
    dataframe, headlines = load_headline_csv(file)

    assert len(dataframe) == 1
    assert headlines.iloc[0] == "Example headline"


def test_missing_headline_column():
    file = BytesIO(b"title\nExample headline\n")

    with pytest.raises(ValueError, match="headline"):
        load_headline_csv(file)



def test_blank_headline_is_preserved():
    file = BytesIO(b'headline\n""\n')
    dataframe, headlines = load_headline_csv(file)

    assert len(dataframe) == 1
    assert headlines.iloc[0] == ""



def test_row_limit():
    file = BytesIO(b"headline\nOne\nTwo\n")

    with pytest.raises(ValueError, match="maximum"):
        load_headline_csv(file, max_rows=1)
