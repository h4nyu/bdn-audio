import typing as t


class Audio:
    id: str
    spectrogram: t.Any

    def __init__(self, id: str, spectrogram: t.Any) -> None:
        self.id = id
        self.spectrogram = spectrogram


Audios = t.List[Audio]
