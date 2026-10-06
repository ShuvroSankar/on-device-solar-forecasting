// Register_SELECT() wrapper -- the upstream select.cc only exposes
// Register_SELECT_V2(), but the underlying Prepare/Eval already handle
// the V1 (non-broadcasting, same-shape) case correctly via
// `requires_broadcast = false`. This wrapper re-exposes the same
// registration under the SELECT (v1) opcode so graphs that emit the
// older opcode (like TTM's Erf decomposition) can register it.
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

// Provided by select.cc.
TFLMRegistration Register_SELECT_V2();

TFLMRegistration Register_SELECT() {
  return Register_SELECT_V2();
}

}  // namespace tflite
