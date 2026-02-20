#include <cmath>
#include <cstring>
#include <vector>

static inline float sigmoid_safe(float z) {
  if (z > 35.0f) return 1.0f;
  if (z < -35.0f) return 0.0f;
  return 1.0f / (1.0f + std::exp(-z));
}

extern "C" {

#if defined(_WIN32)
__declspec(dllexport)
#endif
void train_lr(const float* X, const int* y, int n_samples, int n_features,
              int epochs, float lr, float l2, float* out_w, float* out_b) {
  if (n_samples <= 0 || n_features <= 0) return;

  std::vector<float> w(static_cast<size_t>(n_features), 0.0f);
  float b = 0.0f;

  for (int ep = 0; ep < epochs; ++ep) {
    std::vector<float> grad_w(static_cast<size_t>(n_features), 0.0f);
    float grad_b = 0.0f;

    for (int i = 0; i < n_samples; ++i) {
      const float* xi = X + static_cast<size_t>(i) * n_features;
      float z = b;
      for (int j = 0; j < n_features; ++j) {
        z += w[j] * xi[j];
      }
      const float p = sigmoid_safe(z);
      const float diff = p - static_cast<float>(y[i]);

      for (int j = 0; j < n_features; ++j) {
        grad_w[j] += diff * xi[j];
      }
      grad_b += diff;
    }

    for (int j = 0; j < n_features; ++j) {
      grad_w[j] = grad_w[j] / n_samples + l2 * w[j];
      w[j] -= lr * grad_w[j];
    }
    grad_b = grad_b / n_samples;
    b -= lr * grad_b;
  }

  std::memcpy(out_w, w.data(), sizeof(float) * static_cast<size_t>(n_features));
  *out_b = b;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
void predict_lr_batch(const float* w, float b, const float* X, int n_samples,
                      int n_features, float* out_probs) {
  if (n_samples <= 0 || n_features <= 0) return;

  for (int i = 0; i < n_samples; ++i) {
    const float* xi = X + static_cast<size_t>(i) * n_features;
    float z = b;
    for (int j = 0; j < n_features; ++j) {
      z += w[j] * xi[j];
    }
    out_probs[i] = sigmoid_safe(z);
  }
}

}  // extern "C"
