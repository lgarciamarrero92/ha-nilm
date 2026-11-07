from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from app.shuffle import epoch_orders


def loss_and_gradient(head, ref, query, onoff, power, losses, cls_weight=1.0, reg_weight=1.0):
    predictions, cache = head.forward(ref, query)
    cls_loss, cls_grad = losses["onoff"](onoff, predictions["onoff"])
    output_gradients = {"onoff": cls_weight * cls_grad}
    reg_loss = 0.0
    if power is not None:
        if "power" not in predictions:
            raise RuntimeError("Head model does not expose a regression output for power-based fine-tuning.")
        reg_loss, reg_grad = losses["power"](power, predictions["power"])
        output_gradients["power"] = reg_weight * reg_grad
    gradient = head.backward(cache, output_gradients)
    return cls_weight * cls_loss + reg_weight * reg_loss, cls_loss, reg_loss, gradient


class Adam:
    def __init__(self, ref, lr):
        self.lr = np.float32(lr)
        self.momentum = np.zeros_like(ref)
        self.velocity = np.zeros_like(ref)
        self.step = 0

    def update(self, ref, gradient):
        self.step += 1
        self.momentum += (gradient - self.momentum) * 0.1
        self.velocity += (gradient**2 - self.velocity) * 0.001
        beta1_power = np.float32(0.9) ** np.float32(self.step)
        beta2_power = np.float32(0.999) ** np.float32(self.step)
        rate = self.lr * np.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
        ref -= rate * self.momentum / (np.sqrt(self.velocity) + 1e-7)


def _to_probability(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    if float(np.min(x)) < -0.05 or float(np.max(x)) > 1.05:
        x = 1.0 / (1.0 + np.exp(-x))
    return np.clip(x, 0.0, 1.0)


def _normalize_power_targets(targets_power: list[float], settings: Dict[str, Any]) -> np.ndarray:
    return _normalize_power_array(np.asarray(targets_power, dtype=np.float32), settings)


def _normalize_power_array(y_power: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    power_mean = float(settings["power_mean"])
    power_std = float(settings["power_std"])
    if abs(power_std) < 1e-12:
        raise ValueError("settings.power_std must be non-zero.")
    y_power = np.asarray(y_power, dtype=np.float32).reshape(-1)
    return ((y_power - power_mean) / power_std).astype(np.float32)


def _denorm_power(v_norm: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    power_mean = float(settings["power_mean"])
    power_std = float(settings["power_std"])
    return v_norm.astype(np.float32) * power_std + power_mean


def _best_prob_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob = _to_probability(y_prob)

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5, 0.0

    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= thr).astype(np.int32)
        f1 = _binary_f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def _binary_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return 0.0

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    denom = (2 * tp) + fp + fn
    if denom <= 0:
        return 0.0
    return float((2.0 * tp) / float(denom))


def _best_power_threshold(y_true_w: np.ndarray, y_pred_w: np.ndarray, gt_on_threshold: float = 5.0) -> Tuple[float, float]:
    eps = 1e-12
    y_true_w = np.clip(np.asarray(y_true_w, dtype=np.float32).reshape(-1), 0.0, None)
    y_pred_w = np.clip(np.asarray(y_pred_w, dtype=np.float32).reshape(-1), 0.0, None)

    true_on = y_true_w >= float(gt_on_threshold)
    if int(true_on.sum()) == 0:
        fallback = float(max(gt_on_threshold, float(np.max(y_pred_w)) + 1.0 if y_pred_w.size else gt_on_threshold))
        return fallback, 0.0

    cand = np.unique(np.quantile(y_pred_w, np.linspace(0.0, 1.0, 101)))
    cand = cand[cand >= float(gt_on_threshold)]
    if cand.size == 0:
        cand = np.array([float(gt_on_threshold)], dtype=np.float32)
    elif not np.any(np.isclose(cand, float(gt_on_threshold))):
        cand = np.unique(np.append(cand, float(gt_on_threshold)))

    best_t = float(cand[0])
    best_f1 = -1.0
    for t in cand:
        pred_on = y_pred_w >= float(t)
        tp = np.sum(pred_on & true_on)
        fp = np.sum(pred_on & ~true_on)
        fn = np.sum(~pred_on & true_on)
        f1 = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


def _quantile_threshold_from_scores(y_scores: np.ndarray, quantile: float) -> float:
    y_scores = np.asarray(y_scores, dtype=np.float32).reshape(-1)
    if y_scores.size == 0:
        return 1.0
    quantile = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(np.clip(y_scores, 0.0, 1.0), quantile))


def _build_weak_pseudo_target(
    q_embs: np.ndarray,
    weak_mains: np.ndarray,
    onoff: np.ndarray,
    cls_conf: np.ndarray,
    conf_threshold: float,
    *,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a weak pseudo-power target from ON samples and confident donor points.
    OFF samples remain 0, ON samples receive a donor mixture clipped by local weak mains.
    """
    q_embs = np.asarray(q_embs, dtype=np.float32)
    weak_mains = np.asarray(weak_mains, dtype=np.float32).reshape(-1)
    onoff = (np.asarray(onoff, dtype=np.float32).reshape(-1) > 0.5).astype(np.uint8)
    cls_conf = np.clip(np.asarray(cls_conf, dtype=np.float32).reshape(-1), 0.0, 1.0)

    if q_embs.ndim != 2:
        raise ValueError("q_embs must be a 2D array")
    if q_embs.shape[0] != weak_mains.shape[0] or q_embs.shape[0] != onoff.shape[0]:
        raise ValueError("q_embs, weak_mains, and onoff must have the same length")

    n = weak_mains.shape[0]
    target = np.zeros(n, dtype=np.float32)
    donor_idx = np.array([], dtype=np.int32)

    on_mask = onoff == 1
    vals = np.maximum(weak_mains, 0.0).astype(np.float32)
    on_idx = np.flatnonzero(on_mask & (vals > 0.0))
    if on_idx.size == 0:
        return target.reshape(-1, 1).astype(np.float32), donor_idx

    conf_threshold = float(np.clip(conf_threshold, 1e-4, 1.0))
    eligible_idx = on_idx[cls_conf[on_idx] >= conf_threshold]
    if eligible_idx.size == 0:
        eligible_idx = on_idx[np.argsort(cls_conf[on_idx])[::-1][:1]]

    donor_idx = eligible_idx.astype(np.int32)
    donor_q = q_embs[donor_idx]
    donor_val = vals[donor_idx]

    sim = np.matmul(q_embs[on_idx], donor_q.T).astype(np.float32)
    temp = max(float(temperature), 1e-4)
    sim = sim / temp
    sim = sim - np.max(sim, axis=1, keepdims=True)
    attn = np.exp(sim).astype(np.float32)
    attn_sum = np.sum(attn, axis=1, keepdims=True)

    fallback_mask = attn_sum[:, 0] <= 1e-8
    if np.any(fallback_mask):
        attn[fallback_mask, :] = 1.0
        attn_sum = np.sum(attn, axis=1, keepdims=True)

    attn = attn / np.maximum(attn_sum, 1e-8)
    pseudo_on = np.sum(attn * donor_val.reshape(1, -1), axis=1)
    pseudo_on = np.minimum(np.maximum(pseudo_on.astype(np.float32), 0.0), vals[on_idx])
    target[on_idx] = pseudo_on

    return target.reshape(-1, 1).astype(np.float32), donor_idx


def train_ref_embedding(
    *,
    query_embeddings: list[list[float]],
    targets_on: list[int],
    targets_power: Optional[list[float]],
    weak_mains: Optional[list[float]],
    settings: Dict[str, Any],
    load_head: Callable,
    losses,
    epochs: int = 200,
    lr: float = 0.01,
    batch_size: int = 1024,
    seed: int = 0,
    patience: int = 5,
    min_delta: float = 1e-4,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_every_s: float = 1.0,
) -> Tuple[list[float], Dict[str, Any], Dict[str, float]]:
    rng = np.random.RandomState(seed)

    Q_np = np.asarray(query_embeddings, dtype=np.float32)
    if Q_np.ndim != 2 or Q_np.size == 0:
        raise ValueError("query_embeddings must be a non-empty 2D array-like structure.")

    y_on_np = np.asarray(targets_on, dtype=np.float32).reshape(-1)
    if y_on_np.shape[0] != Q_np.shape[0]:
        raise ValueError("targets_on and query_embeddings must have the same length.")

    y_power_norm_np: Optional[np.ndarray] = None
    weak_mains_np: Optional[np.ndarray] = None
    fine_tune_target = "onoff"

    if weak_mains is not None:
        if len(weak_mains) != len(targets_on):
            raise ValueError("weak_mains and targets_on must have the same length.")
        weak_mains_np = np.clip(np.asarray(weak_mains, dtype=np.float32).reshape(-1), 0.0, None)
        fine_tune_target = "weak_onoff"
    elif targets_power is not None:
        if len(targets_power) != len(targets_on):
            raise ValueError("targets_power and targets_on must have the same length.")
        y_power_norm_np = _normalize_power_targets(targets_power, settings)
        fine_tune_target = "power"

    N = int(Q_np.shape[0])
    D = int(Q_np.shape[1])

    head = load_head()
    if D != head.embedding_dim:
        raise ValueError(f"Expected {head.embedding_dim} embedding values, received {D}.")
    # A head can cache its fixed query work as long as the result supports row indexing.
    prepare_queries = getattr(head, "prepare_queries", None)
    head_queries = prepare_queries(Q_np) if prepare_queries is not None else Q_np

    weak_reg_loss_weight = float(settings.get("weak_reg_loss_weight", 1.0))
    onoff_loss_weight = float(settings.get("onoff_loss_weight", 1.0))

    def _evaluate_current_ref(ref_value: np.ndarray) -> Dict[str, np.ndarray]:
        outputs = {}
        for start in range(0, N, batch_size):
            predictions, _ = head.forward(ref_value, head_queries[start:start + batch_size])
            for name, values in predictions.items():
                outputs.setdefault(name, []).append(values)
        return {name: np.concatenate(batches) for name, batches in outputs.items()}

    def _run_stage(
        *,
        stage_name: str,
        init_ref_value: np.ndarray,
        mode: str,
        aux_target: Optional[np.ndarray] = None,
        stage_epochs: int = epochs,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
        ref_emb = init_ref_value.reshape(1, D).copy()
        optimizer = Adam(ref_emb, lr)
        steps_per_epoch = (N + batch_size - 1) // batch_size
        cls_weight = onoff_loss_weight if mode == "weak_onoff" else 1.0
        reg_weight = weak_reg_loss_weight if mode == "weak_onoff" else 1.0

        total_losses: list[float] = []
        cls_losses: list[float] = []
        reg_losses: list[float] = []
        best_total = float("inf")
        best_epoch = -1
        best_ref_emb = None
        last_emit = 0.0

        for epoch, indices in enumerate(epoch_orders(N, seed, stage_epochs)):
            total_sum = 0.0
            cls_sum = 0.0
            reg_sum = 0.0
            step = 0

            for start in range(0, N, batch_size):
                step += 1
                batch = indices[start:start + batch_size]
                power = aux_target[batch] if aux_target is not None else None
                total_v, cls_v, reg_v, gradient = loss_and_gradient(
                    head, ref_emb, head_queries[batch], y_on_np[batch], power,
                    losses, cls_weight, reg_weight,
                )
                optimizer.update(ref_emb, gradient)
                total_sum += total_v
                cls_sum += cls_v
                reg_sum += reg_v

                if on_progress:
                    now = time.monotonic()
                    if (now - last_emit) >= progress_every_s:
                        last_emit = now
                        avg_total = total_sum / max(1, step)
                        avg_cls = cls_sum / max(1, step)
                        avg_reg = reg_sum / max(1, step)
                        on_progress(
                            {
                                "phase": "running",
                                "stage": stage_name,
                                "epoch": epoch + 1,
                                "total_epochs": stage_epochs,
                                "step": step,
                                "total_steps": steps_per_epoch,
                                "pct": (step / max(1, steps_per_epoch)) * 100.0,
                                "loss": avg_total,
                                "min_loss": float(min(total_losses) if total_losses else avg_total),
                                "cls_loss": avg_cls,
                                "reg_loss": avg_reg if mode != "onoff" else None,
                                "fine_tune_target": fine_tune_target,
                            }
                        )

            epoch_total = total_sum / max(1, step)
            epoch_cls = cls_sum / max(1, step)
            epoch_reg = reg_sum / max(1, step)
            total_losses.append(float(epoch_total))
            cls_losses.append(float(epoch_cls))
            reg_losses.append(float(epoch_reg))

            if on_progress:
                on_progress(
                    {
                        "phase": "running",
                        "stage": stage_name,
                        "epoch": epoch + 1,
                        "total_epochs": stage_epochs,
                        "step": steps_per_epoch,
                        "total_steps": steps_per_epoch,
                        "pct": 100.0,
                        "loss": float(epoch_total),
                        "min_loss": float(min(total_losses)),
                        "cls_loss": float(epoch_cls),
                        "reg_loss": float(epoch_reg) if mode != "onoff" else None,
                        "fine_tune_target": fine_tune_target,
                    }
                )

            if epoch_total + min_delta < best_total:
                best_total = epoch_total
                best_epoch = epoch
                best_ref_emb = ref_emb.copy()
            elif patience > 0 and (epoch - best_epoch) >= patience:
                break

        final_ref_emb = (best_ref_emb if best_ref_emb is not None else ref_emb).reshape(-1).astype(np.float32)
        outputs = _evaluate_current_ref(final_ref_emb)
        stats = {
            "epochs_requested": int(stage_epochs),
            "epochs_ran": int(len(total_losses)),
            "final_loss": float(total_losses[-1]),
            "min_loss": float(min(total_losses)),
            "final_cls_loss": float(cls_losses[-1]),
            "min_cls_loss": float(min(cls_losses)),
            "final_reg_loss": float(reg_losses[-1]) if mode != "onoff" else None,
            "min_reg_loss": float(min(reg_losses)) if mode != "onoff" else None,
            "n_samples": int(N),
            "embedding_dim": int(D),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "early_stopped": bool(len(total_losses) < stage_epochs),
            "steps_per_epoch": int(steps_per_epoch),
            "fine_tune_target": fine_tune_target,
            "stage": stage_name,
        }
        return final_ref_emb, stats, outputs

    if fine_tune_target == "weak_onoff":
        stage1_init = rng.normal(loc=0.0, scale=0.05, size=(D,)).astype(np.float32)
        stage1_emb, stage1_stats, stage1_outputs = _run_stage(
            stage_name="weak_stage1",
            init_ref_value=stage1_init,
            mode="onoff",
            stage_epochs=epochs,
        )
        cls_scores = _to_probability(np.asarray(stage1_outputs["onoff"], dtype=np.float32).reshape(-1))
        conf_threshold = _quantile_threshold_from_scores(
            cls_scores,
            float(settings.get("weak_conf_quantile", 0.95)),
        )
        weak_mains_target, donor_idx = _build_weak_pseudo_target(
            Q_np,
            weak_mains_np if weak_mains_np is not None else np.zeros((N,), dtype=np.float32),
            y_on_np,
            cls_scores,
            conf_threshold,
            temperature=float(settings.get("weak_memory_temperature", 0.10)),
        )
        weak_mains_norm_np = _normalize_power_array(weak_mains_target.reshape(-1), settings)
        stage2_emb, stage2_stats, stage2_outputs = _run_stage(
            stage_name="weak_stage2",
            init_ref_value=stage1_emb,
            mode="weak_onoff",
            aux_target=weak_mains_norm_np,
            stage_epochs=epochs,
        )
        final_ref_emb = stage2_emb
        final_stats = {
            **stage2_stats,
            "stage1_final_loss": stage1_stats.get("final_loss"),
            "stage1_min_loss": stage1_stats.get("min_loss"),
            "weak_conf_threshold": float(conf_threshold),
            "weak_donor_count": int(len(donor_idx)),
        }
        final_outputs = stage2_outputs
    elif fine_tune_target == "power":
        init_emb = rng.normal(loc=0.0, scale=0.05, size=(D,)).astype(np.float32)
        final_ref_emb, final_stats, final_outputs = _run_stage(
            stage_name="power",
            init_ref_value=init_emb,
            mode="power",
            aux_target=y_power_norm_np,
            stage_epochs=epochs,
        )
    else:
        init_emb = rng.normal(loc=0.0, scale=0.05, size=(D,)).astype(np.float32)
        final_ref_emb, final_stats, final_outputs = _run_stage(
            stage_name="onoff",
            init_ref_value=init_emb,
            mode="onoff",
            stage_epochs=epochs,
        )

    onoff_prob = _to_probability(np.asarray(final_outputs["onoff"], dtype=np.float32).reshape(-1))
    onoff_threshold, onoff_f1 = _best_prob_threshold(np.asarray(targets_on, dtype=np.int32), onoff_prob)

    appliance_params: Dict[str, float] = {
        "onoff_threshold": float(onoff_threshold),
    }

    power_f1 = None
    if fine_tune_target == "power":
        pred_power_norm = np.asarray(final_outputs["power"], dtype=np.float32).reshape(-1)
        pred_power_w = _denorm_power(pred_power_norm, settings)
        power_threshold, power_f1 = _best_power_threshold(np.asarray(targets_power, dtype=np.float32), pred_power_w)
        appliance_params["power_threshold"] = float(power_threshold)
    elif fine_tune_target == "weak_onoff":
        appliance_params["weak_conf_threshold"] = float(final_stats.get("weak_conf_threshold", 1.0))

    stats = {
        **final_stats,
        "onoff_threshold": float(onoff_threshold),
        "onoff_f1": float(onoff_f1),
        "power_threshold": float(appliance_params.get("power_threshold", 0.0)),
        "power_f1": None if power_f1 is None else float(power_f1),
    }

    return final_ref_emb.reshape(-1).tolist(), stats, appliance_params
