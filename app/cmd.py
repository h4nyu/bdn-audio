import typer
from .pipeline import eda, train, dummy_aug, predict, summary

app = typer.Typer()

app.command()(eda)
app.command()(summary)
app.command()(train)
app.command()(dummy_aug)
app.command()(predict)
