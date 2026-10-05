data {
  int<lower=1> N;
  int<lower=1> R;
  int<lower=1> J;
  array[R] int<lower=1, upper=N> item_idx;
  array[R] int<lower=1, upper=J> view_idx;
  vector[R] y;
  vector<lower=0>[R] se;
  int<lower=1, upper=J> ref_view;
  int<lower=0, upper=1> fit_bias;
  real<lower=0> bias_prior_scale;
}
parameters {
  vector[N] theta;
  real mu0;
  real<lower=0> sigma_theta;
  vector[J - 1] log_a_nonref;
  vector<lower=0>[J] tau;
  vector[J - 1] b_nonref;
}
transformed parameters {
  vector[J] a;
  vector[J] b;
  int pos;
  pos = 1;
  for (j in 1:J) {
    if (j == ref_view) {
      a[j] = 1.0;
      b[j] = 0.0;
    } else {
      a[j] = exp(log_a_nonref[pos]);
      b[j] = fit_bias ? b_nonref[pos] : 0.0;
      pos += 1;
    }
  }
}
model {
  mu0 ~ normal(0, 2);
  sigma_theta ~ exponential(1);
  theta ~ normal(mu0, sigma_theta);
  log_a_nonref ~ normal(0, 1);
  tau ~ exponential(5);
  b_nonref ~ normal(0, bias_prior_scale);
  for (r in 1:R) {
    y[r] ~ normal(b[view_idx[r]] + a[view_idx[r]] * theta[item_idx[r]],
                  sqrt(square(se[r]) + square(tau[view_idx[r]])));
  }
}
generated quantities {
  vector[R] log_lik;
  for (r in 1:R) {
    log_lik[r] = normal_lpdf(y[r] | b[view_idx[r]] + a[view_idx[r]] * theta[item_idx[r]],
                             sqrt(square(se[r]) + square(tau[view_idx[r]])));
  }
}
