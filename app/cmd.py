import typer
from .pipeline import eda, train, dummy_aug, predict

app = typer.Typer()

app.command()(eda)
app.command()(train)
app.command()(dummy_aug)
app.command()(predict)
