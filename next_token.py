#!/usr/bin/env python
# coding: utf-8

# Complete `shift_tokens` and `position_nll` in the [November 23 starter](https://drive.google.com/file/d/1rUwo_KIhmXK-QHX4BGjH5GtW8lQfvCom/view?usp=drivesdk) (setup in P7).
# 
# - **Input:** integer IDs of shape $(B, L)$, with $B \ge 1$ and $L \ge 2$. Slice within each row.
# - **Shapes:** let $T = L - 1$. The logits have shape $(B, T, V)$, and the valid target IDs have shape $(B, T)$.
# - **Indexing:**
#   - sequences: $0 \le s < B$
#   - positions: $0 \le t < T$
#   - vocabulary entries: $0 \le k < V$
# - **Loss:** gather along the vocabulary axis, then average over all $BT$ positions.
# 
# Run the `tokens` stage.

# In[43]:


import numpy as np


def logsumexp(z): # NumPy replacement for scipy.special.logsumexp
    m = np.max(z)
    return m + np.log(np.sum(np.exp(z - m)))

rng = np.random.default_rng(20261123)

A = np.array([
    [0.08, 0.68, 0.24],
    [0.12, 0.16, 0.72],
    [0.67, 0.25, 0.08]
], dtype=np.float64)


def generate_sequences(B, seed):
    rng = np.random.default_rng(seed)
    S = np.zeros((B, 5), dtype=np.int64)
    S[:, 0] = rng.integers(0, 3, size=B)
    for t in range(1, 5):
        cutoffs = np.cumsum(A[S[:, t-1]], axis=1)[:, :-1]
        S[:, t] = (rng.random((B, 1)) >= cutoffs).sum(axis=1)
    return S

S_train = generate_sequences(20, 20261124)
S_val = generate_sequences(10, 20261125)
S_test = generate_sequences(16, 20261126)

def prepare_data(S):
    S1, S2 = S[:, :-1], S[:, 1:]
    X = np.eye(3, dtype=np.float64)[S1].reshape(-1, 3)
    y = S2.reshape(-1)
    return X, y

X_train, y_train = prepare_data(S_train)
X_val, y_val = prepare_data(S_val)
X_test, y_test = prepare_data(S_test)


# In[44]:


def shift_tokens(X):
    if X.ndim != 2 or len(X) == 0 or X.shape[1] < 2:
        raise ValueError("Each nonempty batch needs at least two time steps.")
    return X[:, :-1], X[:, 1:] # X2[i][j] is the next element of X1[i][j]. shape (B, T)

def position_nll(Z, X): # Z is the logits, (B, T, V); X is the target, (B, T).
    b, _ = X.shape
    _, t, _ = Z.shape # logits
    l = 0
    for i in range(b):
        for j in range(t):
            l += logsumexp(Z[i][j]) - Z[i][j][X[i][j]] # the logits tensor Z already includes the mapping Z_{ijk} = A_{X_{1,ij},k}
    return l/(b * t)

#----------- Test ------------

if __name__ == "__main__":
    X = np.array([[0, 2, 1, 0]])
    
    M = np.array([
        [1, 1, 2],
        [1, 2, 1],
        [1, 2, 3]
    ], dtype=float)
    
    logM = np.log(M)
    
    W = np.array([
        [logM[0], logM[2], logM[1]]
    ])
    
    X1, X2 = shift_tokens(X)
    
    print(np.array_equal(X1, [[0, 2, 1]]))
    print(np.array_equal(X2, [[2, 1, 0]]))
    
    expected = np.log(24) / 3
    
    print(np.isclose(position_nll(W, X2), expected))


# Implement `loss_grad`. For the centered difference use float64, $h=10^{-6}$, five dense examples, $d=2$, $K=3$, seed 20261123 and $\lambda=.07$ (the checker constructs all arrays). Run `python next_token.py checks`. Report the largest scaled error for each parameter block. Inspect the permutation and duplication assertions. If one fails, name the quantity and axis you would inspect first. This part also completes November 16, 7(4).

# In[46]:


def log_prob(Z):
    n, K = Z.shape
    P = np.zeros((n, K))
    for i in range(n):
        mi = np.max(Z[i])
        for j in range(K):
            P[i][j] = Z[i][j] - mi - logsumexp(Z[i] - mi)
    return P

def loss_grad(X, y, W, b, lam):
    n = X.shape[0]
    d, K = W.shape
    J, dWJ, dbJ = 0, np.zeros((d, K)), np.zeros(n)
    Z = X @ W + b
    P = log_prob(Z)
    T = np.zeros((n, K))      
    for i in range(n):
        J += -P[i][y[i]]
        for j in range(K):
            P[i][j] = np.exp(P[i][j])
            if y[i] == j:
                T[i][j] = 1
    dWJ = (X.T @ (P - T))/n + lam * W
    dbJ = np.sum(P-T, axis=0)/n
    J = J/n + 0.5 * lam * np.sum(W ** 2)
    return J, dWJ, dbJ

#-------- Test ----------
if __name__ == "__main__":
    rng = np.random.default_rng(20261123)
    X, y = rng.normal(size=(5, 2)), np.array([2, 0, 2, 1, 2])
    W, b = rng.normal(size=(2, 3)), rng.normal(size=3)
    lam = 0.07
    h = 1e-6
    
    J, dWJ, dbJ = loss_grad(X, y, W, b, lam)
    
    dWJ_num = np.zeros_like(W)
    dbJ_num = np.zeros_like(b)
    
    # Numerical gradient with respect to W
    for i in range(W.shape[0]):
        for k in range(W.shape[1]):
            W_plus = W.copy()
            W_minus = W.copy()
            W_plus[i, k] += h
            W_minus[i, k] -= h
            J_plus = loss_grad(X, y, W_plus, b, lam)[0]
            J_minus = loss_grad(X, y, W_minus, b, lam)[0]
            dWJ_num[i, k] = (J_plus - J_minus) / (2 * h)
    
    # Numerical gradient with respect to b
    for k in range(len(b)):
        b_plus = b.copy()
        b_minus = b.copy()
        b_plus[k] += h
        b_minus[k] -= h
        J_plus = loss_grad(X, y, W, b_plus, lam)[0]
        J_minus = loss_grad(X, y, W, b_minus, lam)[0]
        dbJ_num[k] = (J_plus - J_minus) / (2 * h)
    
    # Normalized gradient errors
    err_W = np.abs(dWJ - dWJ_num) / np.maximum(
        1, np.maximum(np.abs(dWJ), np.abs(dWJ_num))
    )
    
    err_b = np.abs(dbJ - dbJ_num) / np.maximum(
        1, np.maximum(np.abs(dbJ), np.abs(dbJ_num))
    )
    
    print(np.max(err_W))
    print(np.max(err_b))

    # Permutation and duplication assertions
    order = np.array([2, 0, 1]) # Entry j is the old label of new column j.
    new_y = np.argsort(order)[y]
    perm = loss_grad(X, new_y, W[:, order], b[order], lam)
    dup = loss_grad(np.tile(X, (3, 1)), np.tile(y, 3), W, b, lam)
    for actual, expected in ((perm, (J, dWJ[:, order], dbJ[order])),
                             (dup, (J, dWJ, dbJ))):
        for value, oracle in zip(actual, expected):
            np.testing.assert_allclose(value, oracle, rtol=0, atol=1e-12)


# Implement `choose_fit(fits, validation)`. A penalty key indexes a returned $(W,\boldsymbol b)$ pair and its validation NLL. Minimize validation NLL, choosing the larger penalty on an exact tie. Return the key and that fitted pair. Each candidate starts at zero and takes 100 full-batch steps of size $.5$. Compare $\lambda\in\{0,.1\}$. Split seeds are 20261124, 20261125 and 20261126 respectively.

# In[48]:


def full_gradD(X, y, W, b, lam): # incorporated from 11-16-26
    n = X.shape[0]
    d, K = W.shape
    Wf, bf = W, b
    eta = 0.5 # the steps of size 0.5
    for i in range(100):
        J, dWJ, dbJ = loss_grad(X, y, Wf, bf, lam)
        Wf = Wf - eta * dWJ
        bf = bf - eta * dbJ
        # Parameter centering
        Wf = Wf - Wf.mean(axis=1, keepdims=True)
        bf = bf - bf.mean()
    return lam, (Wf, bf)

def choose_fit(fits, val): # validation, choosing the hyperparameter. Both are dicts keyed by lambda.
    keys = list(fits)
    j = keys[0]
    for i in keys[1:]:
        if val[i] < val[j] or (val[i] == val[j] and i > j): # comparing the regularization term first.
            j = i
    W, b = fits[j]
    return j, W, b

#-----------Test------------
if __name__ == "__main__":
    fits = {}
    val = {}
    
    d = X_train.shape[1]
    K = A.shape[0]
    
    print("Lambda    Train NLL    Validation NLL")
    
    for lam in [0.0, 0.1]:
        W = np.zeros((d, K))
        b = np.zeros(K)
        fit = full_gradD(X_train, y_train, W, b, lam)
        _, (Wf, bf) = fit
        
        Z_train = Wf[S_train[:, :-1]] + bf # We include the information of the initial mapping Z_{ijk} = A_{S_{1,ij},k} here
        Z_val = Wf[S_val[:, :-1]] + bf
        train_nll = position_nll(Z_train, S_train[:, 1:])
        val_nll = position_nll(Z_val, S_val[:, 1:])

        fits[lam] = (Wf, bf)
        val[lam] = val_nll
    
        print(f"{lam:<9.1f} {train_nll:<12.8f} {val_nll:.8f}")
    
        assert np.isfinite(val_nll)
        assert Wf.shape == (d, K)
        assert bf.shape == (K,)
    
        np.testing.assert_allclose(
            Wf.mean(axis=1), 0, atol=1e-12
        )
        np.testing.assert_allclose(
            bf.mean(), 0, atol=1e-12
        )
    
    # Model selection
    lam_best, W_best, b_best = choose_fit(fits, val)
    # Test set evaluation
    Z_test = W_best[S_test[:, :-1]] + b_best
    test_nll = position_nll(Z_test, S_test[:, 1:])
    
    test_error = np.mean(
        np.argmax(X_test @ W_best + b_best, axis=1) != y_test
    )
    print(f"\nSelected lambda = {lam_best}")
    print(f"Test NLL = {test_nll:.8f}")
    print(f"Test error = {test_error:.8f}")
    
    # Verify model selection and exact tie
    assert val[lam_best] == min(val.values())
    assert choose_fit(fits, {0.0: 1.0, 0.1: 1.0})[0] == 0.1
    assert choose_fit(dict(reversed(fits.items())), {0.1: 1.0, 0.0: 1.0})[0] == 0.1

