
import pytest

from src.fake_news.predict import predict_headline


def test_prediction_has_expected_fields():
    result = predict_headline(
        "Government announces new economic policy"
    )

    assert {
        "headline",
        "prediction",
        "confidence",
        "fake_probability",
        "real_probability",
    }.issubset(result.keys())


def test_prediction_label_is_valid():
    result = predict_headline(
        "Government announces new economic policy"
    )

    assert result["prediction"] in {"Fake", "Real"}


def test_probabilities_are_valid():
    result = predict_headline(
        "Government announces new economic policy"
    )

    assert 0.0 <= result["fake_probability"] <= 1.0
    assert 0.0 <= result["real_probability"] <= 1.0
    assert result["fake_probability"] + result["real_probability"] == pytest.approx(1.0)
    assert result["confidence"] == pytest.approx(
        max(result["fake_probability"], result["real_probability"])
    )


def test_empty_headline_is_rejected():
    with pytest.raises(ValueError):
        predict_headline("   ")
