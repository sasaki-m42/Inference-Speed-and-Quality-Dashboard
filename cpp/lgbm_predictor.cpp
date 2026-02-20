#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using BoosterHandle = void*;

using BoosterCreateFromModelfileFn = int (*)(const char*, int*, BoosterHandle*);
using BoosterFreeFn = int (*)(BoosterHandle);
// Keep this variadic to tolerate LightGBM minor ABI differences across builds.
using BoosterPredictForMatFn = int (*)(...);
using GetLastErrorFn = const char* (*)();

struct LightGBMApi {
  BoosterCreateFromModelfileFn booster_create_from_modelfile = nullptr;
  BoosterFreeFn booster_free = nullptr;
  BoosterPredictForMatFn booster_predict_for_mat = nullptr;
  GetLastErrorFn get_last_error = nullptr;
};

#if defined(_WIN32)
using LibHandle = HMODULE;
#else
using LibHandle = void*;
#endif

struct PredictorContext {
  LibHandle lib_handle = nullptr;
  LightGBMApi api;
  BoosterHandle booster = nullptr;
  int num_threads = 1;
};

thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }

#if defined(_WIN32)
LibHandle open_library(const char* path) {
  return LoadLibraryA(path);
}

void* load_symbol(LibHandle lib, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(lib, name));
}

void close_library(LibHandle lib) {
  if (lib != nullptr) {
    FreeLibrary(lib);
  }
}

const char* default_lgbm_lib_name() { return "lib_lightgbm.dll"; }
#else
LibHandle open_library(const char* path) {
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

void* load_symbol(LibHandle lib, const char* name) {
  return dlsym(lib, name);
}

void close_library(LibHandle lib) {
  if (lib != nullptr) {
    dlclose(lib);
  }
}

#if defined(__APPLE__)
const char* default_lgbm_lib_name() { return "lib_lightgbm.dylib"; }
#else
const char* default_lgbm_lib_name() { return "lib_lightgbm.so"; }
#endif
#endif

bool load_lightgbm_api(LibHandle lib, LightGBMApi* api) {
  api->booster_create_from_modelfile = reinterpret_cast<BoosterCreateFromModelfileFn>(
      load_symbol(lib, "LGBM_BoosterCreateFromModelfile"));
  api->booster_free = reinterpret_cast<BoosterFreeFn>(load_symbol(lib, "LGBM_BoosterFree"));
  api->booster_predict_for_mat = reinterpret_cast<BoosterPredictForMatFn>(
      load_symbol(lib, "LGBM_BoosterPredictForMat"));
  api->get_last_error = reinterpret_cast<GetLastErrorFn>(load_symbol(lib, "LGBM_GetLastError"));

  return api->booster_create_from_modelfile != nullptr && api->booster_free != nullptr &&
         api->booster_predict_for_mat != nullptr && api->get_last_error != nullptr;
}

std::string api_error(const PredictorContext* ctx, const std::string& prefix) {
  if (ctx != nullptr && ctx->api.get_last_error != nullptr) {
    const char* err = ctx->api.get_last_error();
    if (err != nullptr) {
      return prefix + ": " + std::string(err);
    }
  }
  return prefix;
}

}  // namespace

extern "C" {

#if defined(_WIN32)
__declspec(dllexport)
#endif
int lgbm_predictor_init(const char* lgbm_lib_path, const char* model_path, int num_threads,
                        void** out_handle) {
  if (model_path == nullptr || out_handle == nullptr) {
    set_error("model_path and out_handle must not be null");
    return 1;
  }

  const char* lib_path = lgbm_lib_path;
  if (lib_path == nullptr || std::strlen(lib_path) == 0) {
    lib_path = default_lgbm_lib_name();
  }

  auto ctx = std::make_unique<PredictorContext>();
  ctx->num_threads = std::max(1, num_threads);

  ctx->lib_handle = open_library(lib_path);
  if (ctx->lib_handle == nullptr) {
    set_error(std::string("Failed to load LightGBM native library: ") + lib_path);
    return 2;
  }

  if (!load_lightgbm_api(ctx->lib_handle, &ctx->api)) {
    set_error("Failed to resolve LightGBM C API symbols");
    close_library(ctx->lib_handle);
    return 3;
  }

  int num_iterations = 0;
  int rc = ctx->api.booster_create_from_modelfile(model_path, &num_iterations, &ctx->booster);
  if (rc != 0 || ctx->booster == nullptr) {
    set_error(api_error(ctx.get(), "LGBM_BoosterCreateFromModelfile failed"));
    close_library(ctx->lib_handle);
    return 4;
  }

  *out_handle = ctx.release();
  return 0;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
int lgbm_predictor_predict(void* handle, const float* X, int n_samples, int n_features, int predict_type,
                           double* out_scores) {
  if (handle == nullptr || X == nullptr || out_scores == nullptr) {
    set_error("handle, X, and out_scores must not be null");
    return 1;
  }
  if (n_samples <= 0 || n_features <= 0) {
    set_error("n_samples and n_features must be > 0");
    return 2;
  }

  auto* ctx = reinterpret_cast<PredictorContext*>(handle);
  std::string params = "num_threads=" + std::to_string(std::max(1, ctx->num_threads));

  int64_t out_len = 0;
  int rc = ctx->api.booster_predict_for_mat(
      ctx->booster,
      X,
      0,  // C_API_DTYPE_FLOAT32
      static_cast<int32_t>(n_samples),
      static_cast<int32_t>(n_features),
      1,             // row major
      predict_type,  // C_API_PREDICT_NORMAL=0
      0,
      -1,
      params.c_str(),
      &out_len,
      out_scores);

  if (rc != 0) {
    set_error(api_error(ctx, "LGBM_BoosterPredictForMat failed"));
    return 3;
  }

  if (out_len < n_samples) {
    set_error("LightGBM output length is smaller than requested sample count");
    return 4;
  }

  return 0;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
void lgbm_predictor_free(void* handle) {
  if (handle == nullptr) {
    return;
  }
  auto* ctx = reinterpret_cast<PredictorContext*>(handle);
  if (ctx->booster != nullptr && ctx->api.booster_free != nullptr) {
    ctx->api.booster_free(ctx->booster);
    ctx->booster = nullptr;
  }
  close_library(ctx->lib_handle);
  ctx->lib_handle = nullptr;
  delete ctx;
}

#if defined(_WIN32)
__declspec(dllexport)
#endif
const char* lgbm_predictor_last_error() {
  return g_last_error.c_str();
}

}  // extern "C"
