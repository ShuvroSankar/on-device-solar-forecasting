#pragma once
#include <cstdint>

// Real preprocessed+quantized window pulled from the test split.
// Input quantization: scale=0.020258275792002678, zero_point=-79
// Output quantization: scale=0.015330422669649124, zero_point=-88

constexpr int kTestInputContextLength = 36;
constexpr int kTestInputChannels = 5;
constexpr int kTestOutputLength = 18;
constexpr float kInputScale = 0.020258275792002678f;
constexpr int kInputZeroPoint = -79;
constexpr float kOutputScale = 0.015330422669649124f;
constexpr int kOutputZeroPoint = -88;

const int8_t kTestInput[] = {
    -107, -91, -31, -38, -107, -107, -90, -31, -38, -107, -107, -89, -31, -38, -107, -107, -88, -30, -38, -107, -107, -87, -30, -38, -107, -107, -85, -30, -38, -107, -107, -84, -30, -38, -107, -107, -83, -30, -38, -107, -107, -82, -30, -38, -107, -107, -81, -30, -38, -107, -107, -80, -30, -38, -107, -107, -79, -30, -39, -107, -107, -78, -30, -39, -107, -107, -77, -30, -39, -107, -107, -76, -30, -39, -107, -107, -75, -30, -39, -107, -107, -74, -30, -39, -107, -107, -73, -30, -39, -107, -107, -71, -30, -39, -107, -107, -70, -30, -39, -107, -107, -69, -31, -39, -107, -107, -68, -31, -39, -107, -107, -67, -31, -39, -107, -107, -66, -31, -39, -107, -107, -65, -32, -39, -107, -107, -64, -32, -39, -107, -107, -63, -32, -39, -107, -107, -62, -33, -39, -107, -107, -61, -33, -39, -107, -107, -60, -33, -39, -107, -107, -59, -34, -39, -107, -107, -58, -34, -39, -107, -107, -57, -35, -39, -107, -107, -56, -35, -39, -107, -107, -55, -36, -39, -107, -107, -54, -36, -39, -107
};

// Expected output from running this exact input through the
// verified .tflite model on host -- compare on-device output against this.
const int8_t kExpectedOutput[] = {
    -126, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -125, -126, -126, -125, -126, -125
};
