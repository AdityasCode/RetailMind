import sys


def stderr_print(text) -> None:
    """
    print to stderr instead of stdout
    :param text: text to print
    :return:
    """
    print(text, file=sys.stderr)