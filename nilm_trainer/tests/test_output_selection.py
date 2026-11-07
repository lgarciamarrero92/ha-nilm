"""Regression tests for the shared NumPy trainer and its head contract."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from app.losses import resolve_losses
from app.model_registry import discover_server_bundles
from app.shuffle import epoch_orders
from app.trainer import loss_and_gradient, train_ref_embedding


LOSSES = resolve_losses({
    "onoff": {"name": "binary_crossentropy"},
    "power": {"name": "mse"},
})


class IndependentHead:
    embedding_dim = 2

    def forward(self, ref, query):
        values = ref + query
        probability = 1 / (1 + np.exp(-values[:, 1]))
        # Power first on purpose. The trainer must use names to select losses.
        return {"power": values[:, 0], "onoff": probability}, probability

    def backward(self, probability, output_gradients):
        power = output_gradients["power"]
        onoff = output_gradients["onoff"] * probability * (1 - probability)
        return np.array([[power.sum(), onoff.sum()]])


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "bundles"
        cls.bundle = next(bundle for bundle in discover_server_bundles(str(root))
                          if bundle.bundle_id == "online_v1")

    def test_seeded_batch_order_matches_tensorflow(self):
        # Golden orders from TensorFlow 2.15.1 with reshuffle_each_iteration=True.
        self.assertEqual([order.tolist() for order in epoch_orders(7, 0, 2)],
                         [[1, 6, 3, 0, 4, 2, 5], [4, 2, 0, 1, 3, 5, 6]])
        buffered = [sha256(order.astype("<i8").tobytes()).hexdigest()
                    for order in epoch_orders(10012, 0, 2)]
        self.assertEqual(buffered, [
            "0d57c3c7ac3efd2b1e5ba36ac0a9211796d631c5844c8a0de1346b0e18d8e8f7",
            "5281dc4ad5b838f527a5db9b0a349838300e701a0596db2382ca4c84eb224e6f",
        ])

    def test_classification_updates_its_own_reference_component(self):
        initial = np.random.RandomState(0).normal(0, 0.05, 2).astype(np.float32)
        embedding, _, _ = train_ref_embedding(
            query_embeddings=[[0.0, 0.0]] * 4, targets_on=[1] * 4,
            targets_power=[float(initial[0])] * 4, weak_mains=None,
            settings={"power_mean": 0.0, "power_std": 1.0}, load_head=IndependentHead,
            losses=LOSSES, epochs=1, batch_size=4, seed=0, lr=0.01,
        )
        self.assertAlmostEqual(embedding[0], initial[0])
        self.assertGreater(embedding[1], initial[1] + 0.005)

    def test_bundles_share_a_head_with_different_weight_dimensions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for dim, hidden, power_bias in [(128, (128, 128), 0.25), (256, (256, 192), 0.75)]:
                bundle_dir = root / f"test_{dim}"
                bundle_dir.mkdir()
                (bundle_dir / "bundle_manifest.json").write_text(json.dumps({
                    "bundle_id": bundle_dir.name,
                    "head_implementation": "refquery_dense_v1",
                }), encoding="utf-8")
                shapes = {
                    "shared_fc_1": (4 * dim, hidden[0]),
                    "shared_fc_2": hidden,
                    "onoff_classification": (hidden[1], 1),
                    "delta_power_z": (hidden[1], 1),
                    "power_z": (1, 1),
                }
                weights = {}
                for name, (inputs, outputs) in shapes.items():
                    weights[f"{name}_kernel"] = np.zeros((inputs, outputs), np.float32)
                    weights[f"{name}_bias"] = np.zeros(outputs, np.float32)
                weights["power_z_bias"][0] = power_bias
                np.savez(bundle_dir / "head.npz", **weights)

            bundles = {bundle.bundle_id: bundle for bundle in discover_server_bundles(directory)}
            first = bundles["test_128"].load_head()
            second = bundles["test_256"].load_head()
            self.assertIs(type(first), type(second))
            for head, dim, power_bias in [(first, 128, 0.25), (second, 256, 0.75)]:
                self.assertEqual(head.embedding_dim, dim)
                predictions, cache = head.forward(np.zeros((1, dim)), np.zeros((3, dim)))
                np.testing.assert_array_equal(predictions["power"], [power_bias] * 3)
                gradient = head.backward(cache, {"onoff": np.ones(3), "power": np.ones(3)})
                self.assertEqual(gradient.shape, (1, dim))

            weights["shared_fc_2_kernel"] = np.zeros((128, 192), np.float32)
            np.savez(root / "test_256" / "head.npz", **weights)
            with self.assertRaisesRegex(ValueError, "shared_fc_2 requires 256 inputs"):
                bundles["test_256"].load_head()

    def test_reference_gradients(self):
        head = self.bundle.load_head()
        rng = np.random.default_rng(42)
        queries = rng.normal(0, 0.1, (19, head.embedding_dim))
        rows = rng.permutation(len(queries))[::2]
        query = head.prepare_queries(queries)[rows]
        ref = rng.normal(0, 0.05, (1, head.embedding_dim))
        cached_predictions, _ = head.forward(ref, query)
        direct_predictions, _ = head.forward(ref, queries[rows])
        for name in cached_predictions:
            np.testing.assert_allclose(cached_predictions[name], direct_predictions[name])
        direction = rng.normal(size=ref.shape)
        direction /= np.linalg.norm(direction)
        onoff = np.arange(len(rows)) % 2
        power = rng.normal(size=len(rows))
        huber = {**LOSSES, **resolve_losses({"power": {"name": "huber", "delta": 0.7}})}
        cases = [
            (None, LOSSES, (1.0, 1.0)),
            (power, LOSSES, (1.0, 1.0)),
            (power, LOSSES, (1.3, 0.7)),
            (power, huber, (1.0, 1.0)),
        ]
        for target, losses, weights in cases:
            with self.subTest(weights=weights, regression=target is not None, huber=losses is huber):
                def evaluate(embedding):
                    return loss_and_gradient(head, embedding, query, onoff, target, losses, *weights)

                eps = 1e-6
                numerical = (evaluate(ref + eps * direction)[0] -
                             evaluate(ref - eps * direction)[0]) / (2 * eps)
                analytic = np.sum(evaluate(ref)[-1] * direction)
                np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-6)

    def test_sensor_and_interval_training(self):
        rng = np.random.default_rng(42)
        queries = rng.normal(0, 0.1, (11, 128)).tolist()
        labels = [0, 1] * 5 + [0]
        for mode, power, weak, expected_stages in [
            ("power", [100 * y for y in labels], None, {"power"}),
            ("weak_onoff", None, [200 * y for y in labels], {"weak_stage1", "weak_stage2"}),
        ]:
            with self.subTest(mode=mode):
                progress = []
                embedding, stats, params = train_ref_embedding(
                    query_embeddings=queries, targets_on=labels, targets_power=power,
                    weak_mains=weak, settings={"power_mean": 30.0, "power_std": 180.0},
                    load_head=self.bundle.load_head, losses=LOSSES, epochs=2, batch_size=4,
                    on_progress=progress.append,
                )
                self.assertEqual(len(embedding), 128)
                self.assertTrue(np.isfinite(embedding).all())
                self.assertTrue(np.isfinite(stats["final_loss"]))
                self.assertEqual(stats["fine_tune_target"], mode)
                self.assertIn("onoff_threshold", params)
                self.assertEqual({event["stage"] for event in progress}, expected_stages)


if __name__ == "__main__":
    unittest.main()
