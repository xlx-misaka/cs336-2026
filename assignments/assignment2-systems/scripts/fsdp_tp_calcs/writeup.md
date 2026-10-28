# 2D parallelism calculations

## (a)

$$
F_{\mathrm{fwd}}
= \frac{6 B D D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}N_{\mathrm{TP}}}
$$

## (b)

$$
T_{\mathrm{comm}}
= \max\left(
\frac{6(N_{\mathrm{FSDP}}-1)D D_{\mathrm{FF}}}
{N_{\mathrm{FSDP}}N_{\mathrm{TP}}W},
\frac{4(N_{\mathrm{TP}}-1)BD}
{N_{\mathrm{FSDP}}N_{\mathrm{TP}}W}
\right)
$$

## (c)

$$
N \le
\left(1+\frac{BW}{C}\right)
\left(1+\frac{3D_{\mathrm{FF}}W}{2C}\right)
$$

## (d)

令

$$
H = \frac{3BD_{\mathrm{FF}}W}{C} + 3D_{\mathrm{FF}} + 2B.
$$

总通信时间不超过计算时间的条件可整理为：

$$
3D_{\mathrm{FF}}N_{\mathrm{FSDP}}
+ 2BN_{\mathrm{TP}} \le H.
$$

要求两种并行度均至少为 $1$，最大设备数满足：

$$
N \le N_{\max} =
\begin{cases}
1+\dfrac{3D_{\mathrm{FF}}W}{2C},
& H<6D_{\mathrm{FF}}, \\[8pt]
1+\dfrac{BW}{C},
& H<4B, \\[8pt]
\dfrac{H^2}{24BD_{\mathrm{FF}}},
& H\ge\max(6D_{\mathrm{FF}},4B).
\end{cases}
$$

第一种情况取 $N_{\mathrm{FSDP}}=1$，第二种情况取
$N_{\mathrm{TP}}=1$。第三种情况的最优并行度为：

$$
N_{\mathrm{FSDP}}^* = \frac{H}{6D_{\mathrm{FF}}},
\qquad
N_{\mathrm{TP}}^* = \frac{H}{4B}.
$$

内点最优时，两个加权项相等：

$$
3D_{\mathrm{FF}}N_{\mathrm{FSDP}}
= 2BN_{\mathrm{TP}} = \frac{H}{2}.
$$

因此乘积为 $H^2/(24BD_{\mathrm{FF}})$；若某个内点并行度小于
$1$，最优配置则落在该轴取 $1$ 的边界上。
