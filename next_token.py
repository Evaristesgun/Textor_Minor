"""Train and check a NumPy next-token baseline on short Markov chains."""
import argparse
import json
import numpy as np


def log_probs(Z):
    """November 16 rowwise log-Softmax, with classes on the last axis."""
    centered = Z - Z.max(axis=-1, keepdims=True)
    return centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))


def shift_tokens(tokens):
    """Integer IDs (B,L) -> input and target arrays, both (B,L-1)."""
    if tokens.ndim != 2 or len(tokens) == 0 or tokens.shape[1] < 2:
        raise ValueError("Each nonempty batch needs at least two time steps.")
    return tokens[:, :-1], tokens[:, 1:]


def position_nll(logits, targets):
    """Finite logits (B,T,V), legal IDs (B,T) -> scalar mean NLL."""
    log_p = log_probs(logits)
    batch_size, time_steps = targets.shape
    total_nll = 0.0
    for i in range(batch_size):
        for t in range(time_steps):
            total_nll -= log_p[i, t, targets[i, t]]
    return total_nll / (batch_size * time_steps)


def loss_grad(X, y, W, b, lam):
    """Prior interface. Mean loss with an L2 penalty on W only."""
    n = X.shape[0]
    log_p = log_probs(X @ W + b)
    P = np.exp(log_p)
    T = np.zeros_like(P)
    rows = np.arange(n)
    T[rows, y] = 1.0

    J = -log_p[rows, y].mean() + 0.5 * lam * np.sum(W ** 2)
    residual = P - T
    dW = (X.T @ residual) / n + lam * W
    db = residual.sum(axis=0) / n
    return J, dW, db


def choose_fit(fits, validation):
    """Keys are penalty values. Return (key, fitted W, fitted b)."""
    best = next(iter(fits))
    for lam in fits:
        if validation[lam] < validation[best] or (
            validation[lam] == validation[best] and lam > best
        ):
            best = lam
    W, b = fits[best]
    return best, W, b


def draw_sequences(n, seed):
    """Independent length-five chains, uniform start, vocabulary size three."""
    rng = np.random.default_rng(seed)
    trans = np.array([[.08, .68, .24], [.12, .16, .72], [.67, .25, .08]])
    tokens = np.empty((n, 5), dtype=int)
    tokens[:, 0] = rng.integers(3, size=n)
    for t in range(1, tokens.shape[1]):
        cutoffs = np.cumsum(trans[tokens[:, t-1]], axis=1)[:, :-1]
        tokens[:, t] = (rng.random((n, 1)) >= cutoffs).sum(axis=1)
    return tokens


def fit_softmax(X, y, lam):
    """Full-batch gradient descent, 100 steps, shared zero initialization."""
    W, b = np.zeros((X.shape[1], 3)), np.zeros(3)
    for _ in range(100):
        _, dW, db = loss_grad(X, y, W, b, lam)
        W -= .5 * dW
        b -= .5 * db
    return W, b


def token_checks():
    tokens = np.array([[0, 2, 1, 0], [2, 1, 0, 2]])
    inp, target = shift_tokens(tokens)
    np.testing.assert_array_equal(inp, [[0, 2, 1], [2, 1, 0]])
    np.testing.assert_array_equal(target, [[2, 1, 0], [1, 0, 2]])
    W = np.log([[1., 1., 2.], [1., 2., 1.], [1., 2., 3.]])
    hand = np.log(24.) / 3
    loss = position_nll(W[inp[:1]], target[:1])
    np.testing.assert_allclose(loss, hand, rtol=0, atol=1e-12)
    lp = log_probs(W[inp[:1]])
    np.testing.assert_allclose(np.exp(lp[0, np.arange(3), target[0]]),
                               [.5, 1/3, .25], rtol=0, atol=1e-14)
    two_logits = np.log(np.array([[[1., 3., 4.]], [[1., 4., 3.]]]))
    two_y = np.array([[0], [1]])
    np.testing.assert_allclose(position_nll(two_logits, two_y), np.log(16.)/2,
                               rtol=0, atol=1e-12)
    small_in, small_y = shift_tokens(np.array([[1, 2]]))
    np.testing.assert_array_equal(small_in, [[1]])
    np.testing.assert_array_equal(small_y, [[2]])
    try:
        shift_tokens(np.array([[0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("A single token has no target.")
    extreme = position_nll(np.array([[[1000., 0., -1000.]]]), np.array([[2]]))
    np.testing.assert_allclose(extreme, 2000., rtol=0, atol=1e-12)
    return {"hand_nll": loss, "extreme_nll": extreme}


def gradient_checks():
    rng = np.random.default_rng(20261123)
    X, y = rng.normal(size=(5, 2)), np.array([2, 0, 2, 1, 2])
    W, b = rng.normal(size=(2, 3)), rng.normal(size=3)
    lam, h = .07, 1e-6
    loss, dW, db = loss_grad(X, y, W, b, lam)
    # A second loss implementation for the numerical oracle.
    def objective():
        scores = X @ W + b
        nll = np.logaddexp.reduce(scores, axis=1) - scores[np.arange(5), y]
        return nll.mean() + lam * np.sum(W**2) / 2

    np.testing.assert_allclose(loss, objective(), rtol=0, atol=1e-12)
    errors = {}
    for name, param, grad in (("W", W, dW), ("b", b, db)):
        assert grad.shape == param.shape
        diffs = []
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + h
            plus = objective()
            param[idx] = old - h
            minus = objective()
            param[idx] = old
            fd = (plus - minus) / (2*h)
            diffs.append(abs(fd-grad[idx]) / max(1., abs(fd), abs(grad[idx])))
        errors[name] = float(np.max(diffs))
        assert errors[name] < 1e-7, (name, errors[name])
    edge = loss_grad(np.ones((1, 1)), np.array([2]),
                     np.array([[1000., 0., -1000.]]), np.zeros(3), 0.)
    np.testing.assert_allclose(edge[0], 2000., rtol=0, atol=1e-12)
    np.testing.assert_array_equal(edge[1], [[1., 0., -1.]])
    np.testing.assert_array_equal(edge[2], [1., 0., -1.])
    order = np.array([2, 0, 1])  # Entry j is the old label of new column j.
    new_y = np.argsort(order)[y]
    new_W, new_b = W[:, order], b[order]
    np.testing.assert_allclose(log_probs(X @ new_W + new_b),
                               log_probs(X @ W + b)[:, order], rtol=0, atol=1e-12)
    perm = loss_grad(X, new_y, new_W, new_b, lam)
    dup = loss_grad(np.tile(X, (3, 1)), np.tile(y, 3), W, b, lam)
    for actual, expected in ((perm, (loss, dW[:, order], db[order])),
                             (dup, (loss, dW, db))):
        for value, oracle in zip(actual, expected):
            np.testing.assert_allclose(value, oracle, rtol=0, atol=1e-12)
    return errors


def experiment():
    train_x, train_y = shift_tokens(draw_sequences(20, 20261124))
    val_x, val_y = shift_tokens(draw_sequences(10, 20261125))
    X = np.eye(3)[train_x.ravel()]
    fits, validation = {}, {}
    for lam in (0., .1):
        W, b = fit_softmax(X, train_y.ravel(), lam)
        fits[lam] = (W, b)
        validation[lam] = position_nll(W[val_x] + b, val_y)
    # Exercise the selection rule with a tie and with either candidate winning.
    for scores, key in [({0.: 1., .1: 1.}, .1), ({0.: 0., .1: 1.}, 0.),
                        ({0.: 1., .1: 0.}, .1)]:
        k, w, bias = choose_fit(fits, scores)
        assert k == key
        np.testing.assert_array_equal(w, fits[key][0])
        np.testing.assert_array_equal(bias, fits[key][1])
    lam, W, b = choose_fit(fits, validation)
    test_x, test_y = shift_tokens(draw_sequences(16, 20261126))
    test_logits = W[test_x] + b
    # Rebuild test logits from the selected training return and one-hot features.
    fit_W, fit_b = fits[lam]
    direct = np.eye(3)[test_x] @ fit_W + fit_b
    np.testing.assert_allclose(test_logits, direct, rtol=0, atol=1e-12)
    return {"validation_nll": validation, "selected_lambda": lam,
            "test_positions": int(test_y.size),
            "test_nll": position_nll(test_logits, test_y)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=["tokens", "checks", "all"],
                        default="all")
    stage = parser.parse_args().stage
    report = {"tokens": token_checks()}
    if stage != "tokens":
        report["gradient_error"] = gradient_checks()
    if stage == "all":
        report["fit"] = experiment()
    report["status"] = "PASS"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
