# Tensor parallel calculations

## (a)

$$
\begin{aligned}
dy^{(i)} &= dy, \\[4pt]
dz^{(i)} &= dy^{(i)}(W_3^{(i)})^\top, \\[4pt]
dW_3^{(i)} &= (z^{(i)})^\top dy^{(i)}, \\[4pt]
dx_1^{(i)} &= dz^{(i)} \odot f'(x_1^{(i)}) \odot x_2^{(i)}, \\[4pt]
dx_2^{(i)} &= dz^{(i)} \odot f(x_1^{(i)}), \\[4pt]
dW_1^{(i)} &= x^\top dx_1^{(i)}, \\[4pt]
dW_2^{(i)} &= x^\top dx_2^{(i)}.
\end{aligned}
$$

$$
dx = \operatorname{all-reduce}_{\mathrm{SUM}}
\left(
\left\{
dx_1^{(i)}(W_1^{(i)})^\top
+
dx_2^{(i)}(W_2^{(i)})^\top
\right\}_{i=0}^{N_{\mathrm{TP}}-1}
\right).
$$

其中，$\odot$ 表示逐元素乘法。

## (b)

前向：

$$
F_{\mathrm{fwd}} = \frac{6 B D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}
$$

反向：

$$
F_{\mathrm{bwd}} = \frac{12 B D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}
$$

## (c)

前向：

$$
T_{\mathrm{comm,fwd}}
= \frac{4(N_{\mathrm{TP}}-1)BD}{N_{\mathrm{TP}}W}
$$

反向：

$$
T_{\mathrm{comm,bwd}}
= \frac{4(N_{\mathrm{TP}}-1)BD}{N_{\mathrm{TP}}W}
$$

## (d)

前向：

$$
N_{\mathrm{TP}} \le 1 + \frac{3D_{\mathrm{FF}}W}{2C}
$$

反向：

$$
N_{\mathrm{TP}} \le 1 + \frac{3D_{\mathrm{FF}}W}{C}
$$
