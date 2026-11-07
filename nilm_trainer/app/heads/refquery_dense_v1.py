"""Frozen RefQuery dense head and its gradient with respect to the reference."""

import numpy as np


class _QueryBatch:
    def __init__(self, query, fixed_projection):
        self.query = query
        self.fixed_projection = fixed_projection

    def __getitem__(self, rows):
        return _QueryBatch(self.query[rows], self.fixed_projection[rows])


class Head:
    def __init__(self, weights_path):
        with np.load(weights_path, allow_pickle=False) as weights:
            first_kernel, self.first_bias = self._layer(weights, "shared_fc_1")
            feature_width, first_width = first_kernel.shape
            if feature_width % 4:
                raise ValueError("shared_fc_1 must have four input blocks of equal width.")
            self.shared2 = self._layer(weights, "shared_fc_2", input_dim=first_width)
            second_width = self.shared2[0].shape[1]
            self.onoff = self._layer(weights, "onoff_classification", input_dim=second_width, output_dim=1)
            self.delta_power = self._layer(weights, "delta_power_z", input_dim=second_width, output_dim=1)
            self.power = self._layer(weights, "power_z", input_dim=1, output_dim=1)
        self.ref_kernel, self.query_kernel, self.square_kernel, product_kernel = np.split(first_kernel, 4)
        self.cross_kernel = product_kernel - 2.0 * self.square_kernel
        self.embedding_dim = self.ref_kernel.shape[0]

    @staticmethod
    def _layer(weights, name, *, input_dim=None, output_dim=None):
        try:
            kernel = weights[f"{name}_kernel"]
            bias = weights[f"{name}_bias"]
        except KeyError as exc:
            raise ValueError(f"Missing weights for {name}.") from exc
        if kernel.ndim != 2 or 0 in kernel.shape or bias.shape != (kernel.shape[1],):
            raise ValueError(f"Invalid kernel or bias shape for {name}.")
        if input_dim is not None and kernel.shape[0] != input_dim:
            raise ValueError(f"{name} requires {input_dim} inputs.")
        if output_dim is not None and kernel.shape[1] != output_dim:
            raise ValueError(f"{name} requires {output_dim} outputs.")
        return kernel, bias

    @staticmethod
    def _dense(values, layer):
        kernel, bias = layer
        result = values @ kernel
        result += bias
        return result

    def prepare_queries(self, query):
        """Cache the first-layer contribution that depends only on each query."""
        width = self.query_kernel.shape[1]
        fixed = np.empty((len(query), width), dtype=np.result_type(query, self.query_kernel))
        for start in range(0, len(query), 4096):
            rows = slice(start, start + 4096)
            batch = query[rows]
            fixed[rows] = batch @ self.query_kernel
            fixed[rows] += np.square(batch) @ self.square_kernel
        fixed += self.first_bias
        return _QueryBatch(query, fixed)

    def forward(self, ref, query):
        batch = query if isinstance(query, _QueryBatch) else self.prepare_queries(query)
        query = batch.query
        # Expand (ref - query)^2 to separate the fixed and shared contributions.
        ref_projection = ref @ self.ref_kernel + np.square(ref) @ self.square_kernel
        hidden1 = (ref * query) @ self.cross_kernel
        hidden1 += batch.fixed_projection
        hidden1 += ref_projection
        np.maximum(hidden1, 0.0, out=hidden1)
        hidden2 = self._dense(hidden1, self.shared2)
        np.maximum(hidden2, 0.0, out=hidden2)
        logits = self._dense(hidden2, self.onoff)
        with np.errstate(over="ignore"):
            probability = 1.0 / (1.0 + np.exp(-logits))
        delta = self._dense(hidden2, self.delta_power)
        np.maximum(delta, 0.0, out=delta)
        power = self._dense(probability * delta, self.power)
        predictions = {"onoff": probability[:, 0], "power": power[:, 0]}
        return predictions, (ref.copy(), query, hidden1, hidden2, probability, delta)

    def backward(self, cache, output_gradients):
        ref, query, hidden1, hidden2, probability, delta = cache
        probability_grad = output_gradients["onoff"][:, None]
        power_grad = output_gradients.get("power")
        if power_grad is not None:
            gated_grad = power_grad[:, None] @ self.power[0].T
            probability_grad = probability_grad + gated_grad * delta

        logit_grad = probability_grad * probability * (1.0 - probability)
        hidden2_grad = logit_grad @ self.onoff[0].T
        if power_grad is not None:
            delta_grad = gated_grad * probability * (delta > 0)
            hidden2_grad += delta_grad @ self.delta_power[0].T

        hidden2_grad *= hidden2 > 0
        first_grad = hidden2_grad @ self.shared2[0].T
        first_grad *= hidden1 > 0
        # Every row shares the reference-only term, so reduce before projecting it.
        shared_grad = first_grad.sum(axis=0, keepdims=True)
        cross_grad = first_grad @ self.cross_kernel.T
        cross_grad *= query
        return (
            shared_grad @ self.ref_kernel.T
            + 2.0 * ref * (shared_grad @ self.square_kernel.T)
            + cross_grad.sum(axis=0, keepdims=True)
        )
