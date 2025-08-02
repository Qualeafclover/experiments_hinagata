import logging

def setup_logging(out_file="temp.log", format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s'):
    logging.basicConfig(
        filename=out_file,
        format=format,
        filemode='w',
        level=logging.DEBUG,
    )
    logging.getLogger().setLevel(logging.DEBUG)

    # Create a console handler manually
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(console_handler)
