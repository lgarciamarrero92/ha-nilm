import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from energy_accumulator import EnergyAccumulator


class EnergyAccumulatorTests(unittest.TestCase):
    def test_integrates_with_the_trapezoidal_method(self):
        with tempfile.TemporaryDirectory() as directory:
            accumulator = EnergyAccumulator(os.path.join(directory, "energy.json"))
            accumulator.update("sensor.nilm_kettle_energy_consumed", 100.0, 10.0)
            result = accumulator.update("sensor.nilm_kettle_energy_consumed", 200.0, 20.0)

        self.assertAlmostEqual(result.total_kwh, 1500.0 / 3_600_000.0)
        self.assertEqual(result.integration_gap_s, 10.0)
        self.assertFalse(result.skipped_stale_gap)

    def test_stale_gaps_are_not_integrated_and_become_a_new_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            accumulator = EnergyAccumulator(os.path.join(directory, "energy.json"), max_integration_gap_s=30.0)
            accumulator.update("sensor.nilm_kettle_energy_consumed", 100.0, 10.0)
            stale = accumulator.update("sensor.nilm_kettle_energy_consumed", 200.0, 50.0)
            result = accumulator.update("sensor.nilm_kettle_energy_consumed", 200.0, 60.0)

        self.assertEqual(stale.total_kwh, 0.0)
        self.assertTrue(stale.skipped_stale_gap)
        self.assertAlmostEqual(result.total_kwh, 2000.0 / 3_600_000.0)

    def test_persists_the_total_and_ignores_out_of_order_readings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "energy.json")
            accumulator = EnergyAccumulator(path)
            accumulator.update("sensor.nilm_kettle_energy_consumed", 100.0, 10.0)
            expected = accumulator.update("sensor.nilm_kettle_energy_consumed", 100.0, 20.0).total_kwh

            restored = EnergyAccumulator(path)
            out_of_order = restored.update("sensor.nilm_kettle_energy_consumed", 500.0, 15.0)
            result = restored.update("sensor.nilm_kettle_energy_consumed", 100.0, 30.0)

        self.assertEqual(out_of_order.total_kwh, expected)
        self.assertAlmostEqual(result.total_kwh, expected * 2.0)


if __name__ == "__main__":
    unittest.main()
