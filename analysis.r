library("TOSTER")
dapter_f1_mean <- 0.6877919136731481
adapter_f1_std <- 0.012064411522318822

baseline_mean <- 0.7221526639
baseline_std <- 0.009772657484

n1 <- 10
n2 <- 10

alpha <- 0.05

base <- (baseline_std^2 + adapter_f1_std^2 ) / 2
s_pooled <- sqrt(base)
correction_for_samples_less_than_50 <- (n1 - 3) / (n1 - 2.25) * sqrt((n1-2) / n1)

d <- (baseline_mean - dapter_f1_mean) / s_pooled * correction_for_samples_less_than_50
d <- abs(d)
print(d)

tsum_TOST(m1=dapter_f1_mean, m2=baseline_mean, sd1=adapter_f1_std, sd2=baseline_std, n1=n1, n2=n2, low_eqbound=-d, high_eqbound=d, alpha = alpha)
