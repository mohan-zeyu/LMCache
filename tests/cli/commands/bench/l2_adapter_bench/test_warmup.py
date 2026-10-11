# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench.command import _strip_warmup
from lmcache.cli.commands.bench.l2_adapter_bench.result import BenchResult


# Each input round is (duration, successful_keys).
# None means timeout. Each round attempts four keys.
@pytest.mark.parametrize(
    "rounds,warmup,expected_durations,expected_timeouts,expected_rate",
    [
        # Zero warmup preserves the timeout.
        ([(None, 0), (0.1, 4), (0.2, 4)],
         0, [0.1, 0.2], 1, 2 / 3),

        # A warmup timeout removes no measurement duration.
        ([(None, 0), (0.1, 4), (0.2, 4)],
         1, [0.1, 0.2], 0, 1.0),

        # A measurement timeout remains counted.
        ([(0.1, 4), (None, 0), (0.2, 4)],
         1, [0.2], 1, 0.5),

        # Partial successes from a measurement timeout remain counted.
        ([(0.1, 4), (None, 2), (0.2, 4)],
         1, [0.2], 1, 0.75),

        # All measurement rounds time out.
        ([(0.1, 4), (None, 0), (None, 0)],
         1, [], 2, 0.0),

        # Ordinary successful rounds still work.
        ([(0.1, 4), (0.2, 4), (0.3, 4)],
         1, [0.2, 0.3], 0, 1.0),
    ],
)
def test_strip_warmup(
    rounds: list[tuple[float | None, int]],
    warmup: int,
    expected_durations: list[float],
    expected_timeouts: int,
    expected_rate: float,
) -> None:
    original = BenchResult(
        operation="lookup",
        in_flight=1,
        num_keys=4,
        data_size_bytes=0,
        round_durations=[d for d, _ in rounds if d is not None],
        success_counts=[hits for _, hits in rounds],
        timed_out_rounds=sum(d is None for d, _ in rounds),
        round_timed_out=[d is None for d, _ in rounds],
    )

    measured = _strip_warmup(original, warmup)

    assert measured.round_durations == expected_durations
    assert measured.timed_out_rounds == expected_timeouts
    assert measured.attempted_rounds == len(rounds) - warmup
    assert measured.total_keys == 4 * (len(rounds) - warmup)
    assert measured.actual_hit_rate == pytest.approx(expected_rate)
    assert len(measured.round_timed_out) == measured.attempted_rounds
    assert sum(measured.round_timed_out) == expected_timeouts

    # Filtering must preserve the original object.
    assert original.round_durations == [
        d for d, _ in rounds if d is not None
    ]
    assert original.round_timed_out == [d is None for d, _ in rounds]