import typer
from .pipeline import eda, train, dummy_aug, predict, mel_to_audio

app = typer.Typer()

app.command()(eda)
app.command()(train)
app.command()(dummy_aug)
app.command()(predict)
app.command()(mel_to_audio)
