import logging

root_logger = (
    logging.getLogger()
)  # By having no name here, this becomes the root logger. All logs are therefore passed up to this
root_logger.setLevel(logging.DEBUG)

logfmt_str = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = logging.Formatter(logfmt_str)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("logger.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

root_logger.addHandler(stream_handler)
root_logger.addHandler(file_handler)
