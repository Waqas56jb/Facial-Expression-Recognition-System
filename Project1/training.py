import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    # i still use sigmoid here because for this assignment i want to stay close to classical perceptron idea
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """derivative when a already holds sigmoid output for the final setup"""
    # i plug this into backward pass so training code for the final model stays consistent with theory section
    return a * (1.0 - a)


def softmax(z):
    """compute class probabilities from logits in this final network"""
    # i rely on this for the last layer so i can directly use cross entropy and get clear class scores
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def modified_input(a):
    """apply a plus a squared mapping before every dense layer of final architecture"""
    # i wanted at least one visible twist in the input so i keep this here and explain it in my report
    return a + np.square(a)


def modified_input_derivative(a):
    """gradient expression for my modified input step"""
    # i keep this around so when i debug gradients for the final run i can quickly check this part alone
    return 1.0 + 2.0 * a


def batch_norm_forward(x, gamma, beta, running_mean, running_var, eps=1e-5, training=True):
    # for the final model i use batch norm mainly to keep learning steady across long training runs
    if training:
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True) + eps
        running_mean[:] = 0.9 * running_mean + 0.1 * mean
        running_var[:] = 0.9 * running_var + 0.1 * var
    else:
        mean = running_mean
        var = running_var + eps
    x_norm = (x - mean) / np.sqrt(var)
    out = gamma * x_norm + beta
    return out, {"x": x, "mean": mean, "var": var, "x_norm": (x - mean) / np.sqrt(var), "gamma": gamma}


def batch_norm_backward(dout, cache):
    x, mean, var, x_norm, gamma = cache["x"], cache["mean"], cache["var"], cache["x_norm"], cache["gamma"]
    N = x.shape[0]
    ivar = 1.0 / np.sqrt(var)
    dx_hat = dout * gamma
    dgamma = np.sum(dout * x_norm, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    dx = (1.0 / N) * ivar * (N * dx_hat - np.sum(dx_hat, axis=0, keepdims=True)
          - x_norm * np.sum(dx_hat * x_norm, axis=0, keepdims=True))
    return dx, dgamma, dbeta


def dropout_forward(a, keep_prob, training=True):
    # here i drop some activations so this final network does not overfit when i train for many epochs
    if not training or keep_prob >= 1.0:
        return a, None
    mask = (np.random.rand(*a.shape) < keep_prob) / keep_prob
    return a * mask, mask


def dropout_backward(da, mask):
    if mask is None:
        return da
    return da * mask


class ModifiedPerceptron:
    """multi layer perceptron that i finally chose to present as my main solution"""

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.5, seed=42):
        # in this constructor i lock in sizes and initial parameters for the final architecture i am submitting
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.output_size = output_size
        self.dropout = dropout
        self.training = True

        dims = [input_size] + self.hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        self.gamma_bn = []
        self.beta_bn = []
        self.running_mean = []
        self.running_var = []

        for i in range(len(dims) - 1):
            # Xavier/Glorot init for sigmoid: scale = sqrt(2/(fan_in+fan_out))
            fan_in, fan_out = dims[i], dims[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(dims[i], dims[i + 1]) * scale)
            self.biases.append(np.zeros((1, dims[i + 1])))
            if i < len(dims) - 2:
                self.gamma_bn.append(np.ones((1, dims[i + 1])))
                self.beta_bn.append(np.zeros((1, dims[i + 1])))
                self.running_mean.append(np.zeros((1, dims[i + 1])))
                self.running_var.append(np.ones((1, dims[i + 1])))

        self.num_layers = len(self.weights)
        self._cache = {}
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
        self.velocity_gamma = [np.zeros_like(g) for g in self.gamma_bn]
        self.velocity_beta = [np.zeros_like(b) for b in self.beta_bn]

    def forward(self, X):
        """Run input through network and get probabilities."""
        # i run input through each layer applying modified input batch norm sigmoid and dropout until final softmax
        self._cache = {}
        a = X
        self._cache["a0"] = a.copy()

        for i in range(self.num_layers - 1):
            a_mod = modified_input(a)
            self._cache[f"a_mod{i}"] = a_mod
            z = a_mod @ self.weights[i] + self.biases[i]
            z_bn, bn_cache = batch_norm_forward(
                z, self.gamma_bn[i], self.beta_bn[i],
                self.running_mean[i], self.running_var[i],
                training=self.training,
            )
            self._cache[f"bn{i}"] = bn_cache
            a_sigmoid = sigmoid(z_bn)
            self._cache[f"a_sigmoid{i}"] = a_sigmoid.copy()
            a, drop_mask = dropout_forward(a_sigmoid, self.dropout, self.training)
            self._cache[f"drop{i}"] = drop_mask
            self._cache[f"a{i+1}"] = a.copy()

        a_mod = modified_input(a)
        self._cache[f"a_mod{self.num_layers-1}"] = a_mod
        z_out = a_mod @ self.weights[-1] + self.biases[-1]
        probs = softmax(z_out)
        self._cache["probs"] = probs
        return probs

    def backward(self, X, y_true_onehot, probs, sample_weights=None):
        """Backprop step to get gradients."""
        # i compute gradients for all weights biases batch norm and dropout so optimizer can update them
        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        grads_gamma = [None] * (self.num_layers - 1)
        grads_beta = [None] * (self.num_layers - 1)

        dz = probs - y_true_onehot
        if sample_weights is not None:
            dz = dz * sample_weights[:, np.newaxis]
        grads_w[-1] = self._cache[f"a_mod{self.num_layers-1}"].T @ dz
        grads_b[-1] = np.sum(dz, axis=0, keepdims=True)
        da = dz @ self.weights[-1].T
        da = da * modified_input_derivative(self._cache[f"a{self.num_layers-1}"])

        for i in range(self.num_layers - 2, -1, -1):
            da = dropout_backward(da, self._cache[f"drop{i}"])
            da = da * sigmoid_derivative(self._cache[f"a_sigmoid{i}"])
            da, dgamma, dbeta = batch_norm_backward(da, self._cache[f"bn{i}"])
            grads_gamma[i] = dgamma
            grads_beta[i] = dbeta
            # da is dL/d(z_i); use it for weight gradients, then propagate to previous layer
            grads_w[i] = self._cache[f"a_mod{i}"].T @ da
            grads_b[i] = np.sum(da, axis=0, keepdims=True)
            da = da @ self.weights[i].T  # dL/d(a_mod_i), shape (batch, prev_layer_size)
            da = da * modified_input_derivative(self._cache[f"a{i}"])  # dL/d(a_i)

        return grads_w, grads_b, grads_gamma, grads_beta

    def sgd_step(self, X_batch, y_batch, learning_rate, momentum=0.9, weight_decay=1e-4, clip_norm=1.0, sample_weights=None):
        """Do one SGD update on a batch."""
        # i do one optimization step here including gradient clipping weight decay and momentum
        y_onehot = np.zeros((X_batch.shape[0], self.output_size))
        y_onehot[np.arange(X_batch.shape[0]), y_batch] = 1.0

        self.training = True
        probs = self.forward(X_batch)
        grads_w, grads_b, grads_gamma, grads_beta = self.backward(X_batch, y_onehot, probs, sample_weights)

        all_grads = grads_w + grads_b + grads_gamma + grads_beta
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_grads))
        if clip_norm > 0 and total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-8)
            grads_w = [g * scale for g in grads_w]
            grads_b = [g * scale for g in grads_b]
            grads_gamma = [g * scale for g in grads_gamma]
            grads_beta = [g * scale for g in grads_beta]

        for i in range(self.num_layers):
            self.velocity_w[i] = momentum * self.velocity_w[i] + grads_w[i] + weight_decay * self.weights[i]
            self.velocity_b[i] = momentum * self.velocity_b[i] + grads_b[i]
            self.weights[i] -= learning_rate * self.velocity_w[i]
            self.biases[i] -= learning_rate * self.velocity_b[i]
        for i in range(self.num_layers - 1):
            self.velocity_gamma[i] = momentum * self.velocity_gamma[i] + grads_gamma[i]
            self.velocity_beta[i] = momentum * self.velocity_beta[i] + grads_beta[i]
            self.gamma_bn[i] -= learning_rate * self.velocity_gamma[i]
            self.beta_bn[i] -= learning_rate * self.velocity_beta[i]

    def predict(self, X):
        self.training = False
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        self.training = False
        return self.forward(X)

    def get_params_dict(self):
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
            "gamma_bn": [g.copy() for g in self.gamma_bn],
            "beta_bn": [b.copy() for b in self.beta_bn],
            "running_mean": [m.copy() for m in self.running_mean],
            "running_var": [v.copy() for v in self.running_var],
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "dropout": self.dropout,
        }

    def set_params_dict(self, params):
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]
        self.gamma_bn = [g.copy() for g in params["gamma_bn"]]
        self.beta_bn = [b.copy() for b in params["beta_bn"]]
        self.running_mean = [m.copy() for m in params["running_mean"]]
        self.running_var = [v.copy() for v in params["running_var"]]
        self.input_size = params["input_size"]
        self.hidden_sizes = params["hidden_sizes"]
        self.output_size = params["output_size"]
        self.dropout = params.get("dropout", 0.5)
        self.num_layers = len(self.weights)
        # Reset momentum velocity (not saved; for inference not needed)
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
        self.velocity_gamma = [np.zeros_like(g) for g in self.gamma_bn]
        self.velocity_beta = [np.zeros_like(b) for b in self.beta_bn]
