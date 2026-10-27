"""Headless smoke test of the Streamlit app (no browser needed)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_loads_and_solves_exactly():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.metric[0].value == "233,545"
    next(b for b in at.button if b.label == "Optimise").click().run()
    assert not at.exception
    assert any("OPTIMAL" in m.value for m in at.markdown)
