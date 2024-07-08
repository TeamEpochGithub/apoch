"""Setup the logger."""
import logging
import os
import sys
from types import TracebackType

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)


def log_exception(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None) -> None:
    """Log any uncaught exceptions except KeyboardInterrupts.

    Based on https://stackoverflow.com/a/16993115.

    :param exc_type: The type of the exception.
    :param exc_value: The exception instance.
    :param exc_traceback: The traceback.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("A wild %s appeared!", exc_type.__name__, exc_info=(exc_type, exc_value, exc_traceback))

def print_section_separator(title: str, spacing: int = 2) -> None:
    """Print a section separator.

    :param title: title of the section
    :param spacing: spacing between the sections
    """
    try:
        separator_length = os.get_terminal_size().columns
    except OSError:
        separator_length = 200
    separator_char = "="
    title_char = " "
    separator = separator_char * separator_length
    title_padding = (separator_length - len(title)) // 2
    centered_title = (
        f"{title_char * title_padding}{title}{title_char * title_padding}" if len(title) % 2 == 0 else f"{title_char * title_padding}{title}{title_char * (title_padding + 1)}"
    )
    print("\n" * spacing)  # noqa: T201
    print(f"{separator}\n{centered_title}\n{separator}")  # noqa: T201
    print("\n" * spacing)  # noqa: T201


sys.excepthook = log_exception
