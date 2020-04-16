import typing as t


class Audio:
    id: str
    waveform: t.Any

    def __init__(self, id: str, waveform: t.Any) -> None:
        self.id = id
        self.waveform = waveform


Audios = t.List[Audio]
