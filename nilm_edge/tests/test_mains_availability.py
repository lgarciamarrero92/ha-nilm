"""Availability regression tests without HA or TensorFlow dependencies."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from energy_accumulator import EnergyAccumulator
from entity_slug import slugify_entity_suffix
from power_units import normalize_power_to_watts


def complete(coroutine):
    # All I/O below is immediate, so no OS event loop is required.
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Unexpected external I/O")


async def gather(*coroutines, **kwargs):
    return [await coroutine for coroutine in coroutines]


async def wait_for(coroutine, **kwargs):
    return await coroutine


class Session:
    def __init__(self):
        self.posts = []

    def post(self, url, headers, json):
        self.posts.append((url.rsplit("/", 1)[-1], json))
        return self

    async def __aenter__(self):
        return SimpleNamespace(status=200)

    async def __aexit__(self, *args):
        pass


class AvailabilityTests(unittest.TestCase):
    def test_pause_preserves_total_after_restart_and_skips_short_outage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "energy.json")
            accumulator = EnergyAccumulator(path)
            accumulator.update("washer", 1000, 0)
            before = accumulator.update("washer", 1000, 60).total_kwh
            accumulator.pause("washer")
            accumulator = EnergyAccumulator(path)
            resumed = accumulator.update("washer", 1000, 120)
            self.assertEqual(resumed.total_kwh, before)
            self.assertTrue(resumed.skipped_stale_gap)
            self.assertAlmostEqual(accumulator.update("washer", 1000, 180).total_kwh, before * 2)

    def test_unknown_and_unavailable_are_scoped_and_recover(self):
        # Execute the actual live loop and publisher, without importing the
        # add-on's startup dependencies or registering OS signal handlers.
        tree = ast.parse((SRC / "main.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                     and node.name in ("run_live_loop", "publish_disaggregation_dl")]
        for state in ("unknown", "unavailable"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                accumulator = EnergyAccumulator(str(Path(directory) / "energy.json"))
                affected = Mock()
                other = Mock()
                runtime = SimpleNamespace(
                    enabled_appliances_by_mains_bundle={"sensor.mains": {"online_v1": ["washer"]},
                                                       "sensor.other": {"online_v1": ["oven"]}},
                    runtimes={"sensor.mains": {"online_v1": affected}, "sensor.other": {"online_v1": other}},
                )

                async def predict(*args, **kwargs):
                    return {"appliances": {"washer": {"appliance_name": "washer", "bundle_id": "online_v1",
                                                       "power": 600, "onoff": 1}}}

                runtime.disaggregate_next = predict
                app = SimpleNamespace(TOKEN="test", HA_REST_API_URL="http://ha", HA_WS_URL="ws://ha",
                                      refquery_instance=runtime, current_config={},
                                      get_mains=lambda: [{"sensor_id": "sensor.mains"}, {"sensor_id": "sensor.other"}],
                                      get_mains_entry=lambda _: {"unit": "W"},
                                      get_sensor_max_gap_s=lambda: 300, get_primary_mains_sensor_id=lambda: "sensor.mains")
                ns = dict(app_state=app, energy_accumulator=accumulator, _slug=slugify_entity_suffix,
                          normalize_power_to_watts=normalize_power_to_watts, datetime=datetime, timezone=timezone,
                          aiohttp=SimpleNamespace(ClientSession=object), json=json,
                          time=SimpleNamespace(perf_counter=lambda: 0), running=True, LIVE_WS_TIMEOUT_MARGIN_S=5,
                          asyncio=SimpleNamespace(gather=gather, wait_for=wait_for, TimeoutError=TimeoutError))
                messages = [{"type": "auth_required"}, {"type": "auth_ok"}]
                for value in ("1000", state, "1000"):
                    messages.append({"event": {"data": {"new_state": {"entity_id": "sensor.mains", "state": value}}}})

                class Socket:
                    closed = False

                    async def recv(self):
                        message = messages.pop(0)
                        if not messages:
                            ns["running"] = False
                        return json.dumps(message)

                    async def send(self, message):
                        pass

                    async def close(self):
                        self.closed = True

                async def connect(url):
                    return Socket()

                ns["retry_websocket_connection"] = connect
                exec(compile(ast.Module(body=functions, type_ignores=[]), str(SRC / "main.py"), "exec"), ns)
                session = Session()
                complete(ns["run_live_loop"](session))
                for entity in ("sensor.nilm_washer_online_v1_power", "sensor.nilm_washer_online_v1_energy_consumed",
                               "binary_sensor.nilm_washer_online_v1_on"):
                    states = [payload["state"] for name, payload in session.posts if name == entity]
                    self.assertEqual(len(states), 3)
                    self.assertEqual(states[1], "unavailable")
                    self.assertNotEqual(states[2], "unavailable")
                self.assertFalse(any("oven" in name for name, _ in session.posts))
                affected.reset.assert_called_once()
                other.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
