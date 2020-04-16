from app.preprocess import load_audios


def test_load_audios() -> None:
    res = load_audios()
    assert len(res) == 30
