"""
Loss functions for the W-net.

WeightedFocalTverskyLoss is the corrected version: the per-pixel weight
multiplies each of the TP, FP and FN terms. (The earlier version multiplied
the masks and predictions by the weights BEFORE computing 1 - y, which could
make the FP/FN terms negative.)
"""

import keras


class WeightedFocalTverskyLoss:
    """
    Focal Tversky loss with optional per-pixel weights.

    alpha weights false positives, beta false negatives (beta > alpha favours
    recall). gamma > 1 focuses training on hard examples: the loss is
    (1 - Tversky index) ** (1 / gamma).

    Call as loss(y_true, y_pred, weight_map). weight_map must broadcast against
    y_true, e.g. [B, H, W, 1] against [B, H, W, 1].
    """

    def __init__(
        self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6, max_weight=50.0
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.max_weight = max_weight

    def __call__(self, y_true, y_pred, weight_map=None):
        # convert_to_tensor puts everything on the backend's device (the GPU
        # under torch), so CPU and GPU inputs can be mixed safely.
        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.clip(
            keras.ops.convert_to_tensor(y_pred), 1e-7, 1.0 - 1e-7
        )
        if weight_map is None:
            w = 1.0
        else:
            w = keras.ops.clip(
                keras.ops.convert_to_tensor(weight_map), 0.0, self.max_weight
            )

        tp = keras.ops.sum(w * y_true * y_pred)
        fn = keras.ops.sum(w * y_true * (1.0 - y_pred))
        fp = keras.ops.sum(w * (1.0 - y_true) * y_pred)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        # Keep (1 - tversky) away from exactly 0 so the power has a finite gradient
        return keras.ops.clip(1.0 - tversky, 1e-7, 1.0) ** (1.0 / self.gamma)
