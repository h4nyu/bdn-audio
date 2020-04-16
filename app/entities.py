import typing as t


class Label:
    id: int
    category: str
    detail: str

    def __init__(self, id: int, category: str, detail: str) -> None:
        self.id = id
        self.category = category
        self.detail = detail


Labels = t.Dict[int, Label]


class Annotation:
    id: str
    label_ids: t.Sequence[int]

    def __init__(self, id: str, label_ids: t.Sequence[int]) -> None:
        self.id = id
        self.label_ids = label_ids


Annotations = t.List[Annotation]
