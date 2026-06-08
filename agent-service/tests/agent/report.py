"""Pretty CLI output for the test suite."""
from __future__ import annotations

from dataclasses import dataclass, field


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class ScenarioReport:
    id: str
    passed: bool
    duration_s: float
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def print_header(total: int) -> None:
    print(f"{BOLD}Running {total} scenario(s){RESET}")
    print("=" * 60)


def print_scenario(idx: int, total: int, r: ScenarioReport) -> None:
    if r.skipped:
        status = f"{YELLOW}SKIP{RESET}"
    elif r.passed:
        status = f"{GREEN}PASS{RESET}"
    else:
        status = f"{RED}FAIL{RESET}"
    pad = f"[{idx}/{total}]"
    name = r.id.ljust(48, ".")
    print(f"{pad} {name} {status}  {r.duration_s:5.1f}s  ${r.cost_usd:.3f}")
    if r.skipped and r.skip_reason:
        print(f"        {DIM}{r.skip_reason}{RESET}")
    for f in r.failures:
        print(f"        {RED}x{RESET} {f}")


def print_summary(reports: list[ScenarioReport]) -> None:
    passed = sum(1 for r in reports if r.passed and not r.skipped)
    failed = sum(1 for r in reports if not r.passed and not r.skipped)
    skipped = sum(1 for r in reports if r.skipped)
    total_dur = sum(r.duration_s for r in reports)
    total_cost = sum(r.cost_usd for r in reports)
    print("=" * 60)
    parts = [f"{GREEN}{passed} PASS{RESET}", f"{RED}{failed} FAIL{RESET}"]
    if skipped:
        parts.append(f"{YELLOW}{skipped} SKIP{RESET}")
    mins = int(total_dur // 60)
    secs = int(total_dur % 60)
    print(
        f"Total: {' / '.join(parts)}  /  Duration: {mins}m {secs}s  /  Cost: ${total_cost:.2f}"
    )
