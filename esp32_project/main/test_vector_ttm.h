#pragma once
#include <cstdint>

// Window 90 -- highest variance in the calibration set.
// Input range -0.0598..4.3852 std=1.7589
constexpr int kTtmContextLength   = 52;
constexpr int kTtmForecastLength  = 16;
constexpr int64_t kTtmFreqToken   = 3;

constexpr float kNormMean = 9.020441f;
constexpr float kNormStd  = 15.846984f;

constexpr float kSiteCapacityKwp = 1.0f;

const float kTtmInput[] = {
    0.849300f, -0.059782f, -0.011094f, 0.049074f, 0.075043f, 0.250369f, 0.329711f, 0.378474f, 1.642145f, 0.182793f, -0.050448f, 0.176738f, -0.023182f, 0.320508f, 0.057437f, 0.468778f, 0.260107f, 0.085727f, 0.503601f, 0.436313f, 0.742155f, 0.789922f, 0.912618f, 0.646649f, 1.081049f, 0.886248f, 0.411748f, 1.110288f, 2.303805f, 3.057135f, 3.117184f, 3.301093f, 3.582584f, 4.355501f, 3.500928f, 2.968815f, 4.035567f, 4.385186f, 4.185223f, 4.133402f, 4.133832f, 4.177802f, 4.125805f, 4.109625f, 4.128077f, 4.217052f, 4.121059f, 4.107328f, 4.151046f, 4.069441f, 4.053185f, 4.033396f
};

const float kTtmExpectedOutput[] = {
    3.828199f, 3.715088f, 3.639961f, 3.599819f, 3.542212f, 3.484352f, 3.452294f, 3.401974f, 3.336715f, 3.292979f, 3.236177f, 3.156691f, 3.136054f, 3.079819f, 3.041953f, 2.967344f
};
