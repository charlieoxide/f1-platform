import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from src.openf1 import (
    OpenF1Error,
    _build_url,
    _get,
    _http_get,
    build_lap_frame,
    coach_report_openf1,
    fastest_lap,
    resolve_driver,
    resolve_session,
    sector_check,
)
from tests.test_coach import REF, make_lap

T0 = datetime(2024, 5, 25, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.openf1._cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr("src.openf1.time.sleep", lambda s: None)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")


def serve(monkeypatch, handler):
    """handler(url) -> (status, payload); payload is JSON-encoded unless it
    is already bytes. Returns the list of URLs requested."""
    calls = []

    def fake_http_get(url):
        calls.append(url)
        status, payload = handler(url)
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return status, body

    monkeypatch.setattr("src.openf1._http_get", fake_http_get)
    return calls


# ------------------------------------------------------------- HTTP layer


def test_build_url_matches_the_query_that_worked_by_hand() -> None:
    url = _build_url(
        "car_data",
        {"session_key": 9519, "driver_number": 4},
        ["date>=2024-05-25T14:26:53.072", "date<2024-05-25T14:28:05.444"],
    )
    assert url == (
        "https://api.openf1.org/v1/car_data?session_key=9519&driver_number=4"
        "&date>=2024-05-25T14:26:53.072&date<2024-05-25T14:28:05.444"
    )


def test_filters_reach_the_server_as_raw_bytes_not_percent_encoded(
    monkeypatch,
) -> None:
    """httpx sent date%3E=...; curl (verified working) sent date>=....
    Use a real local socket to prove we send the curl form."""
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["path"] = self.path
            body = b'[{"a": 1}]'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(
            "src.openf1.BASE_URL", f"http://127.0.0.1:{server.server_port}"
        )
        data = _get(
            "car_data",
            {"session_key": 9519},
            raw=["date>=2024-05-25T14:26:53.072", "date<2024-05-25T14:28:05.444"],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert data == [{"a": 1}]
    assert seen["path"].endswith(
        "session_key=9519&date>=2024-05-25T14:26:53.072&date<2024-05-25T14:28:05.444"
    )


def test_unreachable_server_becomes_openf1_error() -> None:
    with pytest.raises(OpenF1Error, match="Could not reach OpenF1"):
        _http_get("http://127.0.0.1:1/nothing-listens-here")


def test_404_no_results_becomes_empty_list(monkeypatch) -> None:
    serve(monkeypatch, lambda u: (404, {"detail": "No results found."}))
    assert _get("laps", {"session_key": 1}) == []


def test_200_with_detail_dict_is_empty_or_an_error(monkeypatch) -> None:
    serve(monkeypatch, lambda u: (200, {"detail": "No results found."}))
    assert _get("laps", {"session_key": 1}) == []
    serve(monkeypatch, lambda u: (200, {"detail": "Invalid filter"}))
    with pytest.raises(OpenF1Error, match="Invalid filter"):
        _get("laps", {"session_key": 2})


def test_server_error_and_garbage_body_become_openf1_error(monkeypatch) -> None:
    serve(monkeypatch, lambda u: (500, {}))
    with pytest.raises(OpenF1Error, match="HTTP 500"):
        _get("laps", {"session_key": 3})
    serve(monkeypatch, lambda u: (200, b"<html>maintenance</html>"))
    with pytest.raises(OpenF1Error, match="not valid JSON"):
        _get("laps", {"session_key": 4})


def test_429_is_retried_then_succeeds(monkeypatch) -> None:
    replies = iter([(429, []), (429, []), (200, [{"a": 1}])])
    calls = serve(monkeypatch, lambda u: next(replies))
    assert _get("laps", {"session_key": 5}) == [{"a": 1}]
    assert len(calls) == 3


def test_429_forever_gives_up_with_clear_error(monkeypatch) -> None:
    serve(monkeypatch, lambda u: (429, []))
    with pytest.raises(OpenF1Error, match="rate limit"):
        _get("laps", {"session_key": 6})


def test_successful_responses_are_cached_and_corrupt_cache_is_ignored(
    monkeypatch, tmp_path
) -> None:
    calls = serve(monkeypatch, lambda u: (200, [{"x": 1}]))
    assert _get("laps", {"session_key": 7}) == [{"x": 1}]
    assert _get("laps", {"session_key": 7}) == [{"x": 1}]
    assert len(calls) == 1  # second call came from disk

    for f in (tmp_path / "cache").glob("*.json"):
        f.write_text("{not json")  # corrupt it
    assert _get("laps", {"session_key": 7}) == [{"x": 1}]
    assert len(calls) == 2  # refetched instead of trusting the bad file


def test_empty_results_are_not_cached(monkeypatch) -> None:
    calls = serve(monkeypatch, lambda u: (404, {"detail": "No results found."}))
    _get("laps", {"session_key": 8})
    _get("laps", {"session_key": 8})
    assert len(calls) == 2


# ---------------------------------------------------------- name resolving

SESSIONS = [
    {
        "session_key": 1,
        "meeting_key": 10,
        "country_name": "Monaco",
        "location": "Monaco",
        "circuit_short_name": "Monte Carlo",
    },
    {
        "session_key": 2,
        "meeting_key": 11,
        "country_name": "United States",
        "location": "Miami",
        "circuit_short_name": "Miami",
    },
    {
        "session_key": 3,
        "meeting_key": 12,
        "country_name": "United States",
        "location": "Austin",
        "circuit_short_name": "Austin",
    },
    {
        "session_key": 4,
        "meeting_key": 13,
        "country_name": "United Kingdom",
        "location": "Silverstone",
        "circuit_short_name": "Silverstone",
    },
]


def test_resolve_session_exact_match_on_country_or_location(monkeypatch) -> None:
    serve(monkeypatch, lambda u: (200, SESSIONS))
    assert resolve_session(2024, "Monaco", "Q")["session_key"] == 1
    assert resolve_session(2024, "Miami", "Q")["session_key"] == 2
    assert resolve_session(2024, "silverstone", "Q")["session_key"] == 4


def test_resolve_session_refuses_to_guess(monkeypatch) -> None:
    """The Austria-labelled-as-Britain lesson: no silent best guess."""
    serve(monkeypatch, lambda u: (200, SESSIONS))
    with pytest.raises(OpenF1Error, match="matches several"):
        resolve_session(2024, "United States", "Q")
    with pytest.raises(OpenF1Error, match="Silverstone"):
        resolve_session(2024, "Great Britain", "Q")  # lists real options


def test_resolve_session_rejects_unknown_session_type() -> None:
    with pytest.raises(ValueError, match="Unknown session type"):
        resolve_session(2024, "Monaco", "XYZ")


DRIVERS = [
    {"driver_number": 4, "name_acronym": "NOR"},
    {"driver_number": 1, "name_acronym": "VER"},
]


def test_resolve_driver_by_code_number_and_unknown(monkeypatch) -> None:
    serve(monkeypatch, lambda u: (200, DRIVERS))
    assert resolve_driver(9519, "nor") == 4
    assert resolve_driver(9519, "1") == 1  # a plain number needs no lookup
    with pytest.raises(OpenF1Error, match="Available: NOR, VER"):
        resolve_driver(9519, "XXX")


def test_fastest_lap_ignores_pit_out_and_untimed_laps(monkeypatch) -> None:
    laps = [
        {"lap_duration": None, "date_start": "2024-05-25T14:00:00+00:00"},
        {
            "lap_duration": 60.0,
            "date_start": "2024-05-25T14:01:00+00:00",
            "is_pit_out_lap": True,
        },
        {"lap_duration": 72.5, "date_start": "2024-05-25T14:10:00+00:00"},
        {"lap_duration": 71.4, "date_start": "2024-05-25T14:20:00+00:00"},
    ]
    serve(monkeypatch, lambda u: (200, laps))
    assert fastest_lap(9519, 4)["lap_duration"] == 71.4


# --------------------------------------------------------------- telemetry


def _samples(start_s, end_s, speed=180, step=0.264, brake=0):
    n = int((end_s - start_s) / step) + 1
    return [
        {
            "date": (T0 + timedelta(seconds=start_s + i * step)).isoformat(),
            "speed": speed,
            "throttle": 100,
            "brake": brake,
        }
        for i in range(n)
    ]


def test_build_lap_frame_integrates_distance_and_scales_brake() -> None:
    # 180 km/h = 50 m/s for a 60 s lap -> 3000 m; samples start 0.5 s before
    # the lap and run 0.5 s past its end, as lap_telemetry fetches them.
    frame = build_lap_frame(_samples(-0.5, 60.5, brake=100), T0, 60.0)
    assert frame["Distance"].iloc[0] == 0
    assert frame["Distance"].iloc[-1] == pytest.approx(3000, abs=1)
    assert frame["Brake"].max() == pytest.approx(1.0)  # 100 -> 1.0
    assert frame["Distance"].is_monotonic_increasing
    assert frame["Time"].iloc[0] == 0 and frame["Time"].iloc[-1] == 60.0


def _simulate_samples(lap_df, step=0.264):
    """Drive a synthetic lap and sample it at OpenF1's rate (~3.8 Hz)."""
    d = lap_df["Distance"].to_numpy()
    s, t, out = 0.0, 0.0, []
    while s < d[-1]:
        v = float(np.interp(s, d, lap_df["Speed"]))
        brake_here = np.interp(s, d, lap_df["Brake"].astype(float))
        out.append(
            {
                "date": (T0 + timedelta(seconds=t)).isoformat(),
                "speed": round(v),
                "throttle": round(float(np.interp(s, d, lap_df["Throttle"]))),
                "brake": 100 if brake_here >= 0.5 else 0,
            }
        )
        s += v / 3.6 * step
        t += step
    return out, t


def test_coarse_sampling_still_finds_a_clear_braking_difference() -> None:
    """At ~20 m between samples a 15 m difference is below resolution, so
    plant 60 m: the coach should report roughly that, and say 'later'."""
    from src.coach import compare_laps

    ref_s, ref_t = _simulate_samples(make_lap(REF))
    drv_s, drv_t = _simulate_samples(make_lap([(660, 700, 100, 730)]))
    result = compare_laps(
        build_lap_frame(drv_s, T0, drv_t), build_lap_frame(ref_s, T0, ref_t)
    )
    c = result["corners"][0]
    assert c.label == "Braking zone 1"  # OpenF1 has no corner map
    assert c.brake_delta_m == pytest.approx(60, abs=25)
    assert "later than the reference" in c.message


def test_coach_report_openf1_wires_everything_together(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.openf1.resolve_session", lambda y, e, s: {"session_key": 9519}
    )
    monkeypatch.setattr(
        "src.openf1.resolve_driver", lambda key, d: {"NOR": 4, "VER": 1}[d]
    )
    as_float = lambda df: df.assign(Brake=df["Brake"].astype(float))  # noqa: E731
    laps = {
        4: (as_float(make_lap([(615, 700, 100, 730)])), {"lap_duration": 71.4}),
        1: (as_float(make_lap(REF)), {"lap_duration": 71.6}),
    }
    monkeypatch.setattr("src.openf1.lap_telemetry", lambda key, num: laps[num])

    result = coach_report_openf1("2024 Monaco Q", "nor", "ver")
    assert result["lap_times_s"] == {"NOR": 71.4, "VER": 71.6}
    assert result["corners"]
    assert result["sector_check"] == []  # no sector splits supplied -> no check
    with pytest.raises(ValueError, match="two different drivers"):
        coach_report_openf1("2024 Monaco Q", "NOR", "nor")


def _with_time(lap):
    v = np.maximum(lap["Speed"].to_numpy() / 3.6, 1.0)
    d = lap["Distance"].to_numpy()
    t = np.concatenate([[0.0], np.cumsum(np.diff(d) / ((v[:-1] + v[1:]) / 2))])
    return lap.assign(Time=t, Brake=lap["Brake"].astype(float))


def test_sector_check_agrees_with_official_splits_for_a_uniform_slowdown() -> None:
    """Driver is uniformly 1% slower, so the true gap at each sector line is
    1% of the reference's cumulative time. The coach must reproduce that."""
    from src.coach import compare_laps

    ref = _with_time(make_lap([(600, 700, 100, 730), (1400, 1500, 100, 1530)]))
    total = float(ref["Time"].iloc[-1])
    drv = ref.assign(Time=ref["Time"] * 1.01)
    result = compare_laps(drv, ref)

    ref_lap = {f"duration_sector_{i}": total / 3 for i in (1, 2, 3)}
    drv_lap = {k: v * 1.01 for k, v in ref_lap.items()}
    checks = sector_check(result, ref_lap, drv_lap)

    assert [c["line"] for c in checks] == ["S1/S2", "S2/S3"]
    for c in checks:
        assert c["coach_s"] == pytest.approx(c["official_s"], abs=0.02)
    assert result["total_delta_s"] == pytest.approx(0.01 * total, abs=0.005)


def test_sector_check_is_skipped_when_clocks_do_not_match_or_data_missing() -> None:
    from src.coach import compare_laps

    ref = _with_time(make_lap(REF))
    result = compare_laps(ref, ref)
    assert sector_check(result, {}, {}) == []  # splits missing
    wrong = {f"duration_sector_{i}": 40.0 for i in (1, 2, 3)}  # 120 s vs ~26 s lap
    assert sector_check(result, wrong, wrong) == []
