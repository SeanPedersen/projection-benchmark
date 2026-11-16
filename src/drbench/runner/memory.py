"""Process memory probes from /proc (Linux).

`ru_maxrss` (getrusage/wait4) survives fork+exec, so a child reports at least its parent's RSS
at fork time. VmHWM belongs to the address space created by exec, and writing "5" to
/proc/self/clear_refs resets it. That lets a worker measure the peak of the timed phase alone.
"""

from pathlib import Path

PROC_STATUS = Path("/proc/self/status")
PROC_CLEAR_REFS = Path("/proc/self/clear_refs")
RESET_PEAK_RSS = "5"


def status_kb(field: str) -> int:
    for line in PROC_STATUS.read_text().splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1])
    raise KeyError(field)


def current_rss_kb() -> int:
    return status_kb("VmRSS")


def peak_rss_kb() -> int:
    return status_kb("VmHWM")


def reset_peak_rss() -> None:
    PROC_CLEAR_REFS.write_text(RESET_PEAK_RSS)
