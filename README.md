# axon

Small Bayesian neural networks for [Laplace](https://github.com/mlatinov/laplace), built to sit *inside* ordinary statistical models.

A network in `axon` is just a function plus its weights. It returns numbers on an unconstrained (link) scale, and your model decides what they mean:

- the linear predictor of a regression or classifier;
- a correction to a mechanistic parameter (for example `log CL = log θ + f(covariates)`);
- a missing term in an ODE right-hand side;
- a log hazard, or any other unknown function.

`axon` never touches your likelihood, so it works in any Stan model. Weights get priors stated in **function space**: the templates set how large and how wiggly the learned function can be, independently of how many inputs or hidden units you use. All weights are sampled in non-centred form.

## Install

```bash
laplace add axon --git https://github.com/mlatinov/axon --tag 0.1.0 --subdir laplace
```

```stan
library {
  import axon
}
```

`axon` needs `laplace >= 0.2`, because it uses `pub`, templates and functions as arguments. It has no dependencies.

## What's inside

Shapes follow one convention everywhere:
- `X` is `N x P` (one row per unit of evaluation);
- `W` is `H x P` (output x input), `b` and `v` have length `H`;
- `act` is an activation passed by name.

### Activations

Each has four overloads: `real`, `vector`, `row_vector`, `matrix`. Batch networks bind the `matrix` overload, and `*_vec` networks bind the `vector` one.

| Function | φ(z) | Notes |
|---|---|---|
| `act_identity` | `z` | turns a network into a linear model; useful for tests |
| `act_tanh` | `tanh(z)` | **default**: smooth, bounded, extrapolates as a constant |
| `act_logistic` | `1 / (1 + exp(-z))` | output in (0, 1); natural for gates |
| `act_softplus` | `log1p_exp(z)` | smooth ReLU; extrapolates linearly |
| `act_relu` | `fmax(z, 0)` | provided for comparison; the kink is hard on HMC |
| `act_silu` | `z * inv_logit(z)` | smooth, slightly non-monotone |

### Primitives (ordinary functions)

| Function | Returns |
|---|---|
| `affine(X, W, b)`, `affine(X, W)`, `affine(x, W, b)`, `affine(x, W)` | Pre-activations `X W' + 1 b'` (N x H), or `W x + b` |
| `linear_out(A, v, c)`, `linear_out(a, v, c)` | Output layer `A v + c`: one value per row, or a single value |
| `linear_out(A, V, c)`, `linear_out(a, V, c)` | `K` outputs per row from a shared hidden layer |

### Layers and networks (higher-order: take `act`)

| Function | Returns |
|---|---|
| `dense(X, W, b, act)` | Hidden layer `act(X W' + 1 b')`, N x H |
| `dense_vec(x, W, b, act)` | Hidden layer for a single input vector |
| `dense_res(A, W, b, eps, act)` | Residual layer `A + eps * act(A W' + 1 b')` (W is H x H) |
| `dense_stack(A, W[], b[], act)` | `L = size(W)` equal-width layers (L may be 0) |
| `dense_res_stack(A, W[], b[], eps, act)` | `L` residual layers: the stable way to go deep |
| `mlp1(X, W, b, v, c, act)` | One-hidden-layer network, one value per row: the workhorse |
| `mlp1_vec(x, W, b, v, c, act)` | Same at a single input (ODE right-hand sides, loops) |
| `mlp2(X, W1, b1, W2, b2, v, c, act1, act2)` | Two hidden layers, an activation per layer |
| `mlp2_vec(...)` | Same at a single input |
| `mlp1_multi(X, W, b, V, c, act)` | One shared hidden layer, `K` outputs (N x K) |
| `mlp1_tanh(X, W, b, v, c)`, `mlp1_vec_tanh(...)` | tanh networks as *ordinary* functions (callable from templates and from your own higher-order functions) |

### Parameter templates

One `@use` line declares a layer's weights, their non-centred raw versions, and their priors. Pass the declared names (`<name>_W`, `<name>_b`, `<name>_v`, ...) to the networks.

| Template | Declares | Prior |
|---|---|---|
| `dense_params(name, H, P, s_w, s_b)` | `name_W` (H x P), `name_b` | `W ~ N(0, s_w/√P)`, `b ~ N(0, s_b)` |
| `dense_params_ard(name, H, P, s_b, rel_prior)` | + `name_rel` (length P) | `W[k,p] = rel[p]/√P · z`, one learned relevance per input |
| `dense_params_ordered(name, H, P, s_w, s_b)` | ordered `name_b` | removes the H! unit relabellings |
| `stack_params(name, L, H, s_w, s_b)` | `name_W[L]`, `name_b[L]` | `W_l ~ N(0, s_w/√H)` |
| `out_params(name, H, omega)` | `name_v` | `v ~ N(0, omega/√H)` |
| `out_params_tau(name, H, tau_prior)` | `name_v`, `name_tau` | `v = tau/√H · z`, learned function size **(default when embedding)** |
| `out_params_pos(name, H, omega)` | `name_v > 0` | removes sign-flip copies (tanh only) |
| `out_params_multi(name, K, H, omega)` | `name_V` (K x H) | `V ~ N(0, omega/√H)` |
| `bias_param(name, s_c)`, `bias_vec_param(name, K, s_c)` | `name_c` | `c ~ N(0, s_c)` |
| `mlp1_tanh_block(f, X, N, P, H, tau_prior)` | the whole network, as `vector[N] f` | quick start |

Arguments ending in `_prior` take a whole distribution, e.g. `normal(0, 0.3)` or `exponential(1)`. Because the scale is declared positive, `normal(0, s)` acts as a half-normal.

## Choosing the prior scales

The scales are chosen so their meaning does not change with the number of inputs `P` or hidden units `H`. **Inputs must be standardised** (mean 0, SD 1) for this to hold.

| Argument | Default | Meaning |
|---|---|---|
| `s_w` | 1 | Spread of pre-activations caused by the inputs. Larger values give steeper transitions and wigglier functions. |
| `s_b` | 1 | Bias spread. Keep it near `s_w` so the units' transitions fall inside the data. |
| `omega` / `tau` | depends | Size of the learned function on the link scale. On a log scale, `0.05` means changes of a few percent, `0.3` up to about ×1.5, and `1` several-fold. For the whole mean of a standardised outcome, use `1`. |

With tanh, `omega` is an upper bound on the prior SD of the network output. With `out_params_tau`, the posterior of `tau` tells you how much the data actually need the network: `tau` near 0 means the covariates add nothing beyond the rest of the model.

## Rules to remember

These follow from how Laplace 0.2 compiles functions as arguments and templates.

1. **Pass activations as bare names**, e.g. `axon::act_tanh` or your own function. Built-ins like `tanh` cannot be passed; wrap them first.
2. **Don't nest network calls.** `axon::dense_stack(axon::dense(X, ...), ...)` does not compile; assign the inner layer to a variable first. Nesting inside an *ordinary* call (`linear_out`, `mean`, ...) is fine.
3. **Inside your own higher-order function**, you cannot call `axon` networks. Apply your `act` to `axon::affine(...)` yourself, or call `axon::mlp1_tanh`.
4. **To use a network where Stan expects a function** (ODE solvers, `reduce_sum`), wrap it in an ordinary function that names its activation.
5. **Templates only declare weights.** You write the forward pass. Use one prefix per layer, or share a prefix between templates whose suffixes differ (`dense_params(cov, ...)` + `out_params_tau(cov, ...)` is fine).
6. **Pass `c = 0` when the model already has an intercept** for the same quantity, and centre the network output (`f - mean(f)`).
7. **Judge convergence on network outputs, not weights.** Weights have many symmetric copies, so a high R-hat on a weight is expected. Monitor R-hat and ESS on predictions at fixed inputs.

## Examples

All examples compile with `laplace build --validate`.

### 1. Quick start: a semi-parametric regression

```stan
library {
  import axon
}

@use axon::mlp1_tanh_block(f, X, N, P, H, normal(0, 1));

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> H;
  matrix[N, P] X;          // standardised inputs
  vector[N] y;             // standardised outcome
}
parameters {
  real alpha;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 0.5);
  sigma ~ exponential(1);
  y ~ normal(alpha + f, sigma);
}
```

The template declares the weights, their priors, and `vector[N] f` in `transformed parameters`, so `f` is ready to use in `model`.

### 2. A Bayesian MLP classifier

```stan
library {
  import axon
}

@use axon::dense_params(L1, H, P, 1.0, 1.0);
@use axon::out_params(L1, H, 1.0);
@use axon::bias_param(L1, 1.0);

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> H;
  matrix[N, P] X;                        // standardised inputs
  array[N] int<lower=0, upper=1> y;      // binary outcome
}
transformed parameters {
  vector[N] eta = axon::mlp1(X, L1_W, L1_b, L1_v, L1_c, axon::act_tanh);
}
model {
  y ~ bernoulli_logit(eta);
}
```

### 3. A covariate network inside a pharmacokinetic model

The one-compartment model `C(t) = D/V · exp(-CL/V · t)` stays mechanistic. The network learns how clearance depends on covariates beyond allometric weight scaling. It is evaluated once per subject (`J` rows) and mapped to observations with `CL[id]`.

```stan
library {
  import axon
}

@use axon::dense_params(cov, H, P, 1.0, 1.0);
@use axon::out_params_tau(cov, H, normal(0, 0.3));

data {
  int<lower=1> J;                       // subjects
  int<lower=1> N;                       // concentration records
  int<lower=1> P;                       // covariates fed to the network
  int<lower=1> H;                       // hidden units
  matrix[J, P] X;                       // subject covariates, standardised
  vector<lower=0>[J] WT;                // body weight, kg
  vector<lower=0>[J] dose;              // mg
  array[N] int<lower=1, upper=J> id;
  vector<lower=0>[N] time;              // h
  vector<lower=0>[N] conc;              // mg/L
}
parameters {
  real log_theta_CL;
  real log_theta_V;
  vector[J] eta_raw;
  real<lower=0> omega_CL;
  real<lower=0> sigma;
}
transformed parameters {
  vector[J] f_raw = axon::mlp1(X, cov_W, cov_b, cov_v, 0, axon::act_tanh);
  vector[J] f = f_raw - mean(f_raw);    // centre: log_theta_CL = average patient
  vector[J] CL = exp(log_theta_CL + 0.75 * log(WT / 70) + f + omega_CL * eta_raw);
  vector[J] V  = exp(log_theta_V + log(WT / 70));
}
model {
  vector[N] mu = log(dose[id] ./ V[id]) - (CL[id] ./ V[id]) .* time;
  log_theta_CL ~ normal(log(5), 0.5);
  log_theta_V ~ normal(log(50), 0.5);
  eta_raw ~ std_normal();
  omega_CL ~ exponential(2);
  sigma ~ exponential(2);
  conc ~ lognormal(mu, sigma);
}
```

`exp(f)` is the factor by which a patient's clearance differs from the typical patient of the same weight. `cov_tau` is the learned size of that covariate effect.

### 4. A custom deep network as your own function

```stan
library {
  import axon
}

@use axon::dense_params(in, H, P, 1.0, 1.0);
@use axon::stack_params(hid, L, H, 0.5, 0.5);
@use axon::out_params(out, H, 1.0);
@use axon::bias_param(out, 1.0);

functions {
  vector my_net(matrix X, matrix W0, vector b0, array[] matrix W, array[] vector b,
                vector v, real c) {
    matrix[rows(X), rows(W0)] h0 = axon::dense(X, W0, b0, axon::act_softplus);
    matrix[rows(X), rows(W0)] h  = axon::dense_res_stack(h0, W, b, 0.5, axon::act_tanh);
    return axon::linear_out(h, v, c);
  }
}
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> H;
  int<lower=0> L;                        // residual layers (0 is allowed)
  matrix[N, P] X;
  array[N] int<lower=0, upper=1> y;
}
transformed parameters {
  vector[N] eta = my_net(X, in_W, in_b, hid_W, hid_b, out_v, out_c);
}
model {
  y ~ bernoulli_logit(eta);
}
```

`my_net` takes no function argument, so it is an ordinary function. That is what allows it to call `axon` networks with concrete activations. Depth `L` is data, so one model file covers every depth.

### 5. Your own activation and your own generic network

```stan
library {
  import axon
}

@use axon::dense_params(L1, H, P, 1.0, 1.0);
@use axon::out_params(L1, H, 1.0);

functions {
  // your own activation: provide the overloads you will bind
  matrix act_gauss(matrix z) { return exp(-square(z)); }
  vector act_gauss(vector z) { return exp(-square(z)); }

  // your own generic network: apply act to axon::affine yourself
  vector my_generic(matrix X, matrix W, vector b, vector v, func(matrix) -> matrix act) {
    return axon::linear_out(act(axon::affine(X, W, b)), v, 0);
  }
}
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> H;
  matrix[N, P] X;
  vector[N] y;
}
parameters {
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] f1 = axon::mlp1(X, L1_W, L1_b, L1_v, 0, act_gauss);
  vector[N] f2 = my_generic(X, L1_W, L1_b, L1_v, axon::act_silu);
}
model {
  sigma ~ exponential(1);
  y ~ normal(f1 + f2, sigma);
}
```

A generic function may *call* `act` but not pass it on to `axon::dense`. That is why `affine` is public.

### 6. A network inside an ODE (a universal differential equation)

Logistic growth with a learned extra term in the per-capita growth rate. Multiplying by `N[1]` guarantees that an extinct population stays extinct. The tight prior on `g_tau` keeps `r` and `K` interpretable.

```stan
library {
  import axon
}

@use axon::dense_params(g, H, 1, 1.0, 1.0);
@use axon::out_params_tau(g, H, normal(0, 0.1));

functions {
  vector rhs(real t, vector N, real r, real K, real m, real s,
             matrix W, vector b, vector v) {
    vector[1] u = [ (log(N[1]) - m) / s ]';
    real g = axon::mlp1_vec(u, W, b, v, 0, axon::act_tanh);
    return [ N[1] * (r * (1 - N[1] / K) + g) ]';
  }
}
data {
  int<lower=1> T;
  int<lower=1> H;
  array[T] real ts;
  vector<lower=0>[T] y;
}
transformed data {
  real m = mean(log(y));
  real s = sd(log(y));
}
parameters {
  real<lower=0> r;
  real<lower=0> K;
  real<lower=0> N0;
  real<lower=0> sigma;
}
model {
  array[T] vector[1] Nt = ode_rk45(rhs, [N0]', 0, ts, r, K, m, s, g_W, g_b, g_v);
  r ~ lognormal(0, 0.5);
  K ~ lognormal(log(1e9), 1);
  N0 ~ lognormal(log(1e6), 1);
  sigma ~ exponential(1);
  for (t in 1:T) y[t] ~ lognormal(log(Nt[t][1]), sigma);
}
```

### 7. Predictions and function-space diagnostics

```stan
library {
  import axon
}

@use axon::dense_params(L1, H, P, 1.0, 1.0);
@use axon::out_params(L1, H, 1.0);

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> H;
  matrix[N, P] X;
  vector[N] y;
  int<lower=1> M;
  matrix[M, P] X_new;       // new inputs or a fixed probe set, standardised like X
}
parameters {
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] f = axon::mlp1(X, L1_W, L1_b, L1_v, 0, axon::act_tanh);
}
model {
  sigma ~ exponential(1);
  y ~ normal(f, sigma);
}
generated quantities {
  vector[M] f_new = axon::mlp1(X_new, L1_W, L1_b, L1_v, 0, axon::act_tanh);
  array[M] real y_new = normal_rng(f_new, sigma);
}
```

Use a small fixed probe set (quantiles of each covariate, typical profiles) as `X_new`, and check R-hat and ESS on `f_new` rather than on `L1_W`. Standardise `X_new` with the **training** means and SDs.

### 8. Parallel likelihood with `reduce_sum`

```stan
library {
  import axon
}

@use axon::dense_params(L1, H, P, 1.0, 1.0);
@use axon::out_params(L1, H, 1.0);

functions {
  real partial_ll(array[] int y_slice, int start, int end, matrix X,
                  matrix W, vector b, vector v) {
    vector[end - start + 1] eta = axon::mlp1(X[start:end], W, b, v, 0, axon::act_tanh);
    return bernoulli_logit_lpmf(y_slice | eta);
  }
}
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> H;
  matrix[N, P] X;
  array[N] int<lower=0, upper=1> y;
}
model {
  target += reduce_sum(partial_ll, y, 64, X, L1_W, L1_b, L1_v);
}
```

### 9. Several outputs and automatic relevance determination

```stan
library {
  import axon
}

@use axon::dense_params_ard(cov, H, P, 1.0, exponential(1));
@use axon::out_params_multi(cov, 2, H, 0.3);

data {
  int<lower=1> J;
  int<lower=1> P;
  int<lower=1> H;
  matrix[J, P] X;
}
transformed parameters {
  // column 1 corrects log CL, column 2 corrects log V; cov_rel = relevance of each covariate
  matrix[J, 2] F = axon::mlp1_multi(X, cov_W, cov_b, cov_V, rep_vector(0, 2), axon::act_tanh);
}
```

The two outputs share hidden features, so one covariate pattern can affect both parameters. A posterior `cov_rel[p]` near 0 means covariate `p` is irrelevant.

## Preparing data

Until the helper module ships, do these steps yourself:

- **Standardise** continuous inputs: `(x - mean) / sd`, column by column, in `transformed data` or before passing data. Keep the training means and SDs to transform new inputs. Log-transform quantities spanning orders of magnitude first.
- **Categorical inputs**: append indicator (one-hot) columns to `X`; keep all `J` columns.
- **Centre** an embedded network when the model has an intercept: `f - mean(f)`.
- **Inputs must be complete**: impute missing values, or model them as parameters and build `X` in `transformed parameters`.
- **One row per unit at which the function varies**: for example one row per subject, not per measurement, then index back with `f[id]`.

## Roadmap

| Version | Adds |
|---|---|
| next | **Helpers**: `col_means`, `col_sds`, `standardize`, `one_hot`, `center`, `fan_in_scale`, `weights_rng` (prior predictive draws), `pd_design` (partial dependence), `shift_design` (marginal effects) |
| 0.2 | Structured networks: monotone and partially monotone networks, anchored networks (`f(x0) = 0`), neural additive models, embeddings for many-level categories |
| 0.3 | Recurrent: effect-compartment recurrence, Elman RNN, GRU, ragged sequences |
| 0.4 | Attention and pooling: segment softmax, attention pooling, dot-product and self-attention, kernel smoother, time features |

The full specification (every function's signature, equation and meaning) is in the library plan document.