// TTM solar model on ESP32-C6 (TFLite Micro, float32).
//
// Sibling of main.cpp (SmallTCN). Only one is compiled at a time --
// switch via main/CMakeLists.txt. Both share model_data.cc/.h.
//
// Two inputs:
//   past_values: float32, (1, 1, 52)  -- TF order
//   freq_token:  int64,   (1,)        -- constant 3
// One output:
//   prediction_outputs: float32, (1, 16, 1) in z-scored, capacity-normalized
//   space. Undo with:  y_real = y_norm * kNormStd + kNormMean

#include <cstdio>
#include <cmath>

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "test_vector_ttm.h"

namespace {
constexpr int kTensorArenaSize = 240 * 1024;  // greedy planner
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

extern "C" void app_main(void) {
  printf("\n=== TTM solar model on ESP32-C6 (TFLite Micro, float32) ===\n\n");

  const tflite::Model* model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("ERROR: Model schema version %lu != supported %d\n",
           model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  printf("Model loaded OK (schema %lu, %u bytes)\n",
         model->version(), g_model_data_len);

  static tflite::MicroMutableOpResolver<20> resolver;
  resolver.AddAbs();
  resolver.AddAdd();
  resolver.AddCast();
  resolver.AddConcatenation();
  resolver.AddDiv();
  resolver.AddExp();
  resolver.AddFullyConnected();
  resolver.AddGather();
  resolver.AddLess();
  resolver.AddMean();
  resolver.AddMul();
  resolver.AddReshape();
  resolver.AddRsqrt();
  resolver.AddSelect();
  resolver.AddSign();
  resolver.AddSoftmax();
  resolver.AddSqrt();
  resolver.AddSub();
  resolver.AddSum();
  resolver.AddTranspose();

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    /*resource_variables=*/nullptr, /*profiler=*/nullptr,
    /*preserve_all_tensors=*/false);  // greedy planner -- linear needs ~2.9 MB
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed. Bump kTensorArenaSize "
           "(currently %d bytes).\n", kTensorArenaSize);
    return;
  }
  printf("Tensors allocated. Arena used: %d / %d bytes\n",
         (int)interpreter->arena_used_bytes(), kTensorArenaSize);

  TfLiteTensor* past_values = interpreter->input(0);
  TfLiteTensor* freq_token  = interpreter->input(1);
  TfLiteTensor* output      = interpreter->output(0);

  printf("Input[0] past_values: type=%d dims=%d (",
         past_values->type, past_values->dims->size);
  for (int i = 0; i < past_values->dims->size; ++i)
    printf("%d%s", past_values->dims->data[i],
           i + 1 < past_values->dims->size ? "," : "");
  printf(")\n");

  printf("Input[1] freq_token : type=%d dims=%d (",
         freq_token->type, freq_token->dims->size);
  for (int i = 0; i < freq_token->dims->size; ++i)
    printf("%d%s", freq_token->dims->data[i],
           i + 1 < freq_token->dims->size ? "," : "");
  printf(")\n");

  printf("Output  prediction : type=%d dims=%d (",
         output->type, output->dims->size);
  for (int i = 0; i < output->dims->size; ++i)
    printf("%d%s", output->dims->data[i],
           i + 1 < output->dims->size ? "," : "");
  printf(")\n\n");

  if (past_values->type != kTfLiteFloat32) {
    printf("ERROR: past_values is not float32\n"); return;
  }
  if (freq_token->type != kTfLiteInt64) {
    printf("ERROR: freq_token is not int64\n"); return;
  }

  for (int i = 0; i < kTtmContextLength; ++i)
    past_values->data.f[i] = kTtmInput[i];
  freq_token->data.i64[0] = kTtmFreqToken;

  constexpr int kNumRuns = 20;
  int64_t total_us = 0;
  for (int run = 0; run < kNumRuns; ++run) {
    int64_t t0 = esp_timer_get_time();
    TfLiteStatus st = interpreter->Invoke();
    int64_t t1 = esp_timer_get_time();
    if (st != kTfLiteOk) {
      printf("ERROR: Invoke() failed on run %d\n", run);
      return;
    }
    total_us += (t1 - t0);
  }
  double avg_ms = (total_us / (double)kNumRuns) / 1000.0;
  printf("--- Timing ---\n");
  printf("Average inference: %.3f ms over %d runs (%.2f Hz)\n",
         avg_ms, kNumRuns, 1000.0 / avg_ms);

  printf("\n--- Output check ---\n");
  const float* y = output->data.f;
  float max_diff = 0.0f;
  for (int i = 0; i < kTtmForecastLength; ++i) {
    float diff = fabsf(y[i] - kTtmExpectedOutput[i]);
    if (diff > max_diff) max_diff = diff;
    printf("  [%2d] device=%+.6f  onnx=%+.6f  diff=%.6f\n",
           i, y[i], kTtmExpectedOutput[i], diff);
  }
  printf("\nMax abs diff (device vs ONNX): %.6f\n", max_diff);
  printf("Match within 1e-4: %s\n", max_diff < 1e-4f ? "YES" : "NO");

  printf("\n--- De-normalized (undo z-score, then capacity) ---\n");
  for (int i = 0; i < kTtmForecastLength; ++i) {
    float y_real = y[i] * kNormStd + kNormMean;
    float y_wh   = y_real * kSiteCapacityKwp;
    printf("  [%2d] y_real=%.4f  y_Wh=%.4f\n", i, y_real, y_wh);
  }

  printf("\n=== Done ===\n");

  while (true) { vTaskDelay(pdMS_TO_TICKS(10000)); }
}
