# Problem (baseline_calcs): Compute the variance of the policy gradient estimator (5 points)

Let $\pi_\theta$ define a policy over the binary action space $\mathcal{A} = \{0, 1\}$ with $\pi_\theta(A = 1) = p = \sigma(\theta)$, where $\sigma$ represents the sigmoid function $\sigma(\theta) = \frac{1}{1 + e^{-\theta}}$. Our binary reward function will assign reward 1 to action $A = 1$ and reward 0 otherwise, or $r(A) = \mathbf{1}\{A = 1\}$.

**(a)** Let the policy gradient estimator be given by

$$
\frac{1}{n}\sum_{i=1}^{n} r(A_i)\,\nabla_\theta \log \pi_\theta(A_i) \tag{18}
$$

given $n$ samples $A_i \overset{\mathrm{iid}}{\sim} \pi_\theta$. What is the variance of this estimator?

**Deliverable:** An expression in terms of $n$ and $p$, accompanied by a derivation.

**Answer** (a):

$$
\color{#4CAF50}{\operatorname{Var}(\hat g) = \frac{p(1-p)^3}{n}.}
$$

**(b)** Let the baseline-adjusted policy gradient estimator be given by

$$
\frac{1}{n}\sum_{i=1}^{n} \bigl(r(A_i) - b\bigr)\,\nabla_\theta \log \pi_\theta(A_i) \tag{19}
$$

given $n$ samples $A_i \overset{\mathrm{iid}}{\sim} \pi_\theta$. What is the variance of this estimator?

**Deliverable:** An expression in terms of $n$, $b$, and $p$, accompanied by a derivation and some discussion.

**Answer** (b):

$$
\color{#4CAF50}{\operatorname{Var}(\hat g_b) = \frac{p(1-p)}{n}\bigl[b-(1-p)\bigr]^2.}
$$

**(c)** What is the resulting variance if we substitute the “population mean” baseline $b = p$? Compare this variance to that of the unadjusted policy gradient estimator: is it always lower, always higher, or sometimes higher or lower depending on $p$?

**Answer** (c):

$$
\color{#4CAF50}{
\begin{aligned}
\operatorname{Var}(\hat g_p) &= \frac{p(1-p)}{n}(1-2p)^2, \\
\operatorname{Var}(\hat g_p)-\operatorname{Var}(\hat g) &= \frac{p^2(1-p)}{n}(3p-2).
\end{aligned}
}
$$

$$
\color{#4CAF50}{
\operatorname{Var}(\hat g_p)
\begin{cases}
< \operatorname{Var}(\hat g), & 0<p<\frac{2}{3}, \\
= \operatorname{Var}(\hat g), & p=\frac{2}{3}, \\
> \operatorname{Var}(\hat g), & \frac{2}{3}<p<1.
\end{cases}
}
$$
