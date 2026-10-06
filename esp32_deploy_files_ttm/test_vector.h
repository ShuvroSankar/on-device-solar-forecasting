#pragma once
#include <cstdint>

// Real preprocessed window pulled from the test split.
// TTM is deployed as FLOAT32 (see README: full int8, int16x8,
// and dynamic-range quantization were all tried and ruled out).
// No quantization: values below are real float32 model inputs/outputs.

constexpr int kTestContextLength = 52;
constexpr int kTestForecastLength = 16;
constexpr int64_t kFreqToken = 3;

// past_values, TF-order layout (1, 1, 52) -- NOT (1, 52, 1).
// onnx2tf transposed this during ONNX -> TF conversion; confirm
// your input-preparation code on-device matches this exact layout.
const float kTestInput[] = {
    2.37099743f, 2.60015535f, 3.64676476f, 3.54627061f, 3.84162855f, 3.48999929f, 3.71824861f, 3.40180540f, 3.33621192f, 3.74968290f, 3.42109013f, 3.67444658f, 3.73539615f, 3.68055511f, 3.16425014f, 2.86577916f, 2.91813016f, 2.90951419f, 3.40318561f, 3.13221073f, 3.01051331f, 3.44779563f, 3.32436514f, 3.15676236f, 3.11147881f, 2.65802598f, 2.23718476f, 2.57227230f, 2.44959903f, 2.86490440f, 3.19671082f, 2.54571819f, 1.84370482f, 1.90053153f, 1.69560504f, 1.47087252f, 1.52631938f, 1.50584030f, 1.61717153f, 1.76116538f, 2.08698130f, 2.16092181f, 2.18892312f, 2.21626782f, 2.15215445f, 2.15392184f, 1.73779178f, 1.66179824f, 1.66326225f, 1.09713328f, 1.25305605f, 1.37478697f
};

// Expected output from running this exact input through the
// verified float32 .tflite model on host -- compare on-device
// output against this (float32, no dequantization needed).
const float kExpectedOutput[] = {
    1.33186364f, 1.30620635f, 1.28770971f, 1.25837064f, 1.22651136f, 1.20136118f, 1.16707361f, 1.14191747f, 1.11006308f, 1.09412265f, 1.07585013f, 1.06800914f, 1.03158200f, 1.00795054f, 0.99844480f, 0.98204529f
};
