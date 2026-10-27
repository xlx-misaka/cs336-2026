# Fully sharded data parallel calculations

## (a)

前向：

$$
F_{\mathrm{fwd}}
= \frac{6 B D D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}
$$

反向：

$$
F_{\mathrm{bwd}}
= \frac{12 B D D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}
$$

## (b)

前向：

$$
T_{\mathrm{comm,fwd}}
= \frac{6(N_{\mathrm{FSDP}}-1)D D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}W}
$$

反向：

$$
T_{\mathrm{comm,bwd}}
= \frac{12(N_{\mathrm{FSDP}}-1)D D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}W}
$$

## (c)

前向和反向均满足：

$$
N_{\mathrm{FSDP}} \le 1 + \frac{BW}{C}
$$
