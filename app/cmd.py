import typer
from .pipeline import eda, train

app = typer.Typer()

app.command()(eda)

app.command()(train)
