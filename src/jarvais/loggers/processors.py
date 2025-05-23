# Imports remain the same
import contextlib
import datetime
from pathlib import Path
from typing import Optional

import pytz
from structlog.types import EventDict


class PathPrettifier:
    """
    A processor to convert absolute paths to relative paths based on a base directory.

    Args:
        base_dir (Optional[Path]): The base directory to which paths should be made relative. Defaults to the current working directory.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path.cwd()

    def __call__(
        self, _: object, __: object, event_dict: EventDict
    ) -> EventDict:
        """
        Process the event dictionary to convert Path objects to relative paths.

        Args:
            _: Unused positional argument.
            __: Unused positional argument.
            event_dict (EventDict): The event dictionary containing log information.

        Returns:
            EventDict: The modified event dictionary with relative paths.
        """
        if not isinstance(event_dict, dict):
            msg = "event_dict must be a dictionary"
            raise TypeError(msg)

        for key, path in event_dict.items():
            if isinstance(path, Path):
                with contextlib.suppress(ValueError):
                    relative_path = path.relative_to(self.base_dir)
                    event_dict[key] = str(relative_path)
        return event_dict


class JSONFormatter:
    """
    A processor to format the event dictionary for JSON output.
    """

    def __call__(
        self, _: object, __: object, event_dict: EventDict
    ) -> EventDict:
        """
        Process the event dictionary to separate core fields from extra fields.

        Args:
                _: Unused positional argument.
                __: Unused positional argument.
                event_dict (EventDict): The event dictionary containing log information.

        Returns:
                EventDict: The modified event dictionary with core fields and extra fields separated.
        """
        if not isinstance(event_dict, dict):
            msg = "event_dict must be a dictionary"
            raise TypeError(msg)

        core_fields = {"event", "level", "timestamp", "call", "exception"}
        extra_fields = {
            key: value
            for key, value in event_dict.items()
            if key not in core_fields
        }
        for key in extra_fields:
            del event_dict[key]

        event_dict["extra"] = extra_fields
        return event_dict


class CallPrettifier:
    """
    A processor to format call information in the event dictionary.

    Args:
        concise (bool): Whether to use a concise format for call information. Defaults to True.
    """

    def __init__(self, concise: bool = True) -> None:
        self.concise = concise

    def __call__(
        self, _: object, __: object, event_dict: EventDict
    ) -> EventDict:
        """
        Process the event dictionary to format call information.

        Args:
            _: Unused positional argument.
            __: Unused positional argument.
            event_dict (EventDict): The event dictionary containing log information.

        Returns:
                EventDict: The modified event dictionary with formatted call information.
        """
        if not isinstance(event_dict, dict):
            msg = "event_dict must be a dictionary"
            raise TypeError(msg)

        call = {
            "module": event_dict.pop("module", ""),
            "func_name": event_dict.pop("func_name", ""),
            "lineno": event_dict.pop("lineno", ""),
        }

        event_dict["call"] = (
            f"{call['module']}.{call['func_name']}:{call['lineno']}"
            if self.concise
            else call
        )
        return event_dict


class ESTTimeStamper:
    """
    A processor to add a timestamp in Eastern Standard Time to the event dictionary.

    Parameters
    ----------
        fmt (str): The format string for the timestamp.
        Defaults to "%Y-%m-%dT%H:%M:%S%z".

    Example
    -------
    >>> est_stamper = ESTTimeStamper()
    >>> event_dict = {}
    >>> est_stamper(None, None, event_dict)
    {'timestamp': '2023-10-01T12:00:00-0400'}

    format for just time:
    >>> est_stamper = ESTTimeStamper(fmt="%H:%M:%S")
    """

    def __init__(self, fmt: str = "%Y-%m-%dT%H:%M:%S%z") -> None:
        self.fmt = fmt
        self.est = pytz.timezone("US/Eastern")
        self._last_timestamp: str | None = None

    def __call__(
        self, _: object, __: object, event_dict: EventDict
    ) -> EventDict:
        """
        Process the event dictionary to add a timestamp in Eastern Standard Time,
        avoiding repetition if the timestamp matches the previous one.

        Parameters
        ----------
        _ : object
            Unused positional argument.
        __ : object
            Unused positional argument.
        event_dict : EventDict
            The event dictionary containing log information.

        Returns
        -------
        EventDict
            The modified event dictionary with the added timestamp if it's different
            from the previous one.
        """
        if not isinstance(event_dict, dict):
            msg = "event_dict must be a dictionary"
            raise TypeError(msg)

        now = datetime.datetime.now(self.est)
        current_timestamp = now.strftime(self.fmt)

        if current_timestamp != self._last_timestamp:
            event_dict["timestamp"] = current_timestamp
            self._last_timestamp = current_timestamp
        else:
            # the string should be the same LENGTH of chracters
            # so that the log file is not corrupted
            event_dict["timestamp"] = " " * len(current_timestamp)

        return event_dict