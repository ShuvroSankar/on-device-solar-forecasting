// SIGN kernel for TFLite Micro.
// out[i] = (in[i] > 0) - (in[i] < 0)   (returns -1, 0, or +1)
// Float32 only; TTM's Erf decomposition emits float32 SIGN ops.
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace sign {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  output->type = input->type;

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const float* in = tflite::micro::GetTensorData<float>(input);
  float* out = tflite::micro::GetTensorData<float>(output);
  const int n = tflite::micro::GetTensorShape(input).FlatSize();
  for (int i = 0; i < n; ++i) {
    out[i] = static_cast<float>((in[i] > 0.0f) - (in[i] < 0.0f));
  }
  return kTfLiteOk;
}

}  // namespace sign
}  // namespace micro
}  // namespace ops

// NOTE: Register_* goes in namespace tflite, not tflite::ops::micro --
// that's what the resolver header expects when it declares
// `TFLMRegistration Register_SIGN();` inside `namespace tflite`.
TFLMRegistration Register_SIGN() {
  return tflite::micro::RegisterOp(nullptr,
                                   tflite::ops::micro::sign::Prepare,
                                   tflite::ops::micro::sign::Eval);
}

}  // namespace tflite

// ---------------------------------------------------------------------------
// Flatbuffer option parser for SIGN -- SIGN has no options, so this is a
// no-op that just returns success. Required because this version of the
// resolver's AddBuiltin requires a non-null parser argument.
// ---------------------------------------------------------------------------
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

TfLiteStatus ParseSign(const Operator*, ErrorReporter*,
                       BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

}  // namespace tflite
