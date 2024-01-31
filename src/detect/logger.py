import logging


def get_new_logger(name):
    """

    :param name: str name of your new logger
    :return: logging.Logger your new logger
    """

    if name in logging.root.manager.loggerDict:

        raise Exception("{} exists already".format(name))

    else:
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        return log


def get_simple_handler(level=logging.DEBUG):
    """

    Parameters
    ----------

    level logging level

    Returns
    -------

    StreamHandler

    """
    ch = logging.StreamHandler()
    ch.setLevel(level)

    return ch
