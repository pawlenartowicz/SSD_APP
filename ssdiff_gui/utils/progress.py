"""Progress hook utilities for mapping ssdiff iteration progress to Qt signals."""


def make_progress_cb(signal, start_pct: int, end_pct: int, phase: str):
    """Create a ssdiff progress_hook callback that maps iterations to a % range.

    Parameters
    ----------
    signal : Signal(int, str)
        Qt signal to emit progress updates on. Must accept (percent, message).
    start_pct, end_pct : int
        Progress bar range for this phase.
    phase : str
        Label shown in the progress dialog.
    """
    def cb(current: int, total: int, desc: str) -> None:
        if total > 0:
            fraction = current / total
            pct = start_pct + int(fraction * (end_pct - start_pct))
            signal.emit(pct, f"{phase}: {desc}")
    return cb
