"""Regression tests for the actual trainer output roles and classification loss."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from app.trainer import _infer_output_indices, train_ref_embedding


def head_names(*names):
    return SimpleNamespace(outputs=[SimpleNamespace(name=name) for name in names])


class OutputSelectionTests(unittest.TestCase):
    def test_regression_does_not_match_on(self):
        self.assertEqual(_infer_output_indices(head_names(
            'power_regression_reshaped/Reshape:0', 'onoff_classification_reshaped/Reshape:0')), (0, 1))

    def test_reversed_named_outputs(self):
        self.assertEqual(_infer_output_indices(head_names('onoff/Sigmoid:0', 'power_regression/Reshape:0')), (1, 0))

    def test_standalone_on_name(self):
        self.assertEqual(_infer_output_indices(head_names('on/Identity:0', 'power/Identity:0')), (1, 0))

    def test_named_regression_infers_other_output(self):
        self.assertEqual(_infer_output_indices(head_names('Identity:0', 'power/Identity:0')), (1, 0))

    def test_legacy_unnamed_heads(self):
        self.assertEqual(_infer_output_indices(head_names('Identity:0', 'Identity_1:0')), (0, 1))
        self.assertEqual(_infer_output_indices(head_names('Identity:0')), (None, 0))

    def test_single_classification(self):
        self.assertEqual(_infer_output_indices(head_names('onoff/Sigmoid:0')), (None, 0))

    def test_invalid_or_ambiguous_heads_fail(self):
        for names in [(), ('power:0',), ('a','b','c'),
                      ('power_classification:0','Identity:0'), ('onoff:0','classification:0')]:
            with self.subTest(names=names), self.assertRaises(RuntimeError):
                _infer_output_indices(head_names(*names))

    def test_real_online_bundle_output_roles(self):
        root = Path('/app/bundles')
        if not root.is_dir():
            root = Path(__file__).resolve().parents[1] / 'bundles'
        for bundle in ['online_v1', 'online_v2']:
            with self.subTest(bundle=bundle):
                head = tf.keras.models.load_model(str(root / bundle / 'head.h5'), compile=False)
                self.assertEqual(_infer_output_indices(head), (0, 1))

    def test_classification_loss_updates_its_own_reference_component(self):
        # Independent branches make the original bug observable: the ON component
        # receives no gradient if classification loss incorrectly uses power.
        ref = tf.keras.Input((2,), name='ref_embedding')
        query = tf.keras.Input((2,), name='query_embedding')
        combined = tf.keras.layers.Add()([ref, query])
        power = tf.keras.layers.Dense(1, use_bias=False, name='power_regression',
            kernel_initializer=tf.keras.initializers.Constant([[1.], [0.]]))(combined)
        on = tf.keras.layers.Dense(1, use_bias=False, activation='sigmoid', name='onoff_classification',
            kernel_initializer=tf.keras.initializers.Constant([[0.], [1.]]))(combined)
        head = tf.keras.Model([ref, query], [power, on])
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'head.h5')
            head.save(path, include_optimizer=False)
            embedding, stats, _ = train_ref_embedding(query_embeddings=[[0.,0.]]*4,
                targets_on=[1]*4, targets_power=[0.]*4, weak_mains=None,
                settings={'power_mean':0., 'power_std':1.}, head_model_path=path,
                epochs=1, batch_size=4, seed=0, lr=.01)
        initial = np.random.RandomState(0).normal(0., .05, 2)
        self.assertGreater(embedding[1], initial[1] + .005)
        self.assertTrue(np.isfinite(stats['final_cls_loss']))


if __name__ == '__main__':
    unittest.main()
