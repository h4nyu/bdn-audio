from logging import getLogger, StreamHandler, Formatter, INFO, FileHandler

logger = getLogger()
logger.setLevel(INFO)
stream_handler = StreamHandler()
stream_handler.setLevel(INFO)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

file_handler = FileHandler(filename="app.log")
logger.addHandler(file_handler)
file_handler.setLevel(INFO)
file_handler.setFormatter(handler_format)
