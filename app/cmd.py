import typer
from .pipeline import eda, train, dummy_aug, submit, mel_to_audio, pre_submit

app = typer.Typer()

app.command()(eda)
app.command()(train)
app.command()(dummy_aug)
app.command()(submit)
app.command()(mel_to_audio)
app.command()(pre_submit)
