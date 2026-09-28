/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-09-26T20:49:54+0600
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[63]) {
   { .size_bytes = 180, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 5}}, .scale = {1, (const float[1]){0.020258275792002678}}, .zeropoint = {1, (const int16_t[1]){-79}}, .name = "serving_default_input0_output" },
   { .size_bytes = 200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 40, 5}}, .scale = {1, (const float[1]){0.020258275792002678}}, .zeropoint = {1, (const int16_t[1]){-79}}, .name = "pad_0_output" },
   { .size_bytes = 180, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 5}}, .scale = {1, (const float[1]){0.020258275792002678}}, .zeropoint = {1, (const int16_t[1]){-79}}, .name = "slice_9_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.059947796165943146}}, .zeropoint = {1, (const int16_t[1]){55}}, .name = "conv2d_18_output" },
   { .size_bytes = 180, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 5}}, .scale = {1, (const float[1]){0.020258275792002678}}, .zeropoint = {1, (const int16_t[1]){-79}}, .name = "slice_7_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.0321701355278492}}, .zeropoint = {1, (const int16_t[1]){2}}, .name = "conv2d_16_output" },
   { .size_bytes = 180, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 5}}, .scale = {1, (const float[1]){0.020258275792002678}}, .zeropoint = {1, (const int16_t[1]){-79}}, .name = "slice_5_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.024298058822751045}}, .zeropoint = {1, (const int16_t[1]){6}}, .name = "conv2d_14_output" },
   { .size_bytes = 180, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 5}}, .scale = {1, (const float[1]){0.020258275792002678}}, .zeropoint = {1, (const int16_t[1]){-79}}, .name = "slice_3_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.021649004891514778}}, .zeropoint = {1, (const int16_t[1]){-31}}, .name = "conv2d_12_output" },
   { .size_bytes = 180, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 5}}, .scale = {1, (const float[1]){0.020258275792002678}}, .zeropoint = {1, (const int16_t[1]){-79}}, .name = "slice_1_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.029205355793237686}}, .zeropoint = {1, (const int16_t[1]){11}}, .name = "conv2d_11_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.04217273369431496}}, .zeropoint = {1, (const int16_t[1]){38}}, .name = "eltwise_13_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.043119899928569794}}, .zeropoint = {1, (const int16_t[1]){35}}, .name = "eltwise_15_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.04443957656621933}}, .zeropoint = {1, (const int16_t[1]){23}}, .name = "eltwise_17_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.018089503049850464}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_19_output" },
   { .size_bytes = 1408, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 44, 32}}, .scale = {1, (const float[1]){0.018089503049850464}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_21_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.018089503049850464}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_30_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.10019893199205399}}, .zeropoint = {1, (const int16_t[1]){69}}, .name = "conv2d_39_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.018089503049850464}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_28_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.050918929278850555}}, .zeropoint = {1, (const int16_t[1]){70}}, .name = "conv2d_37_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.018089503049850464}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_26_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.06250172108411789}}, .zeropoint = {1, (const int16_t[1]){75}}, .name = "conv2d_35_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.018089503049850464}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_24_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.08613935112953186}}, .zeropoint = {1, (const int16_t[1]){84}}, .name = "conv2d_33_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.018089503049850464}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_22_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.11588045209646225}}, .zeropoint = {1, (const int16_t[1]){88}}, .name = "conv2d_32_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.16412462294101715}}, .zeropoint = {1, (const int16_t[1]){87}}, .name = "eltwise_34_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.1870257556438446}}, .zeropoint = {1, (const int16_t[1]){88}}, .name = "eltwise_36_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.1870257556438446}}, .zeropoint = {1, (const int16_t[1]){88}}, .name = "eltwise_38_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 32}}, .scale = {1, (const float[1]){0.02584196999669075}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_40_output" },
   { .size_bytes = 1664, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 52, 32}}, .scale = {1, (const float[1]){0.02584196999669075}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_42_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.02584196999669075}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_51_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.14971131086349487}}, .zeropoint = {1, (const int16_t[1]){64}}, .name = "conv2d_60_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.02584196999669075}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_49_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.09137392789125443}}, .zeropoint = {1, (const int16_t[1]){87}}, .name = "conv2d_58_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.02584196999669075}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_47_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.06863502413034439}}, .zeropoint = {1, (const int16_t[1]){59}}, .name = "conv2d_56_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.02584196999669075}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_45_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.05569436401128769}}, .zeropoint = {1, (const int16_t[1]){37}}, .name = "conv2d_54_output" },
   { .size_bytes = 1152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 32}}, .scale = {1, (const float[1]){0.02584196999669075}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_43_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.09451217204332352}}, .zeropoint = {1, (const int16_t[1]){47}}, .name = "conv2d_53_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.11401693522930145}}, .zeropoint = {1, (const int16_t[1]){58}}, .name = "eltwise_55_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.13104695081710815}}, .zeropoint = {1, (const int16_t[1]){67}}, .name = "eltwise_57_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.15027357637882233}}, .zeropoint = {1, (const int16_t[1]){42}}, .name = "eltwise_59_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.05029941350221634}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_61_output" },
   { .size_bytes = 4352, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 68, 64}}, .scale = {1, (const float[1]){0.05029941350221634}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_63_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 64}}, .scale = {1, (const float[1]){0.05029941350221634}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_72_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.19084693491458893}}, .zeropoint = {1, (const int16_t[1]){39}}, .name = "conv2d_81_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 64}}, .scale = {1, (const float[1]){0.05029941350221634}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_70_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.1060132309794426}}, .zeropoint = {1, (const int16_t[1]){75}}, .name = "conv2d_79_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 64}}, .scale = {1, (const float[1]){0.05029941350221634}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_68_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.12077406048774719}}, .zeropoint = {1, (const int16_t[1]){91}}, .name = "conv2d_77_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 64}}, .scale = {1, (const float[1]){0.05029941350221634}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_66_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.14425751566886902}}, .zeropoint = {1, (const int16_t[1]){79}}, .name = "conv2d_75_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 36, 64}}, .scale = {1, (const float[1]){0.05029941350221634}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_64_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.1846686601638794}}, .zeropoint = {1, (const int16_t[1]){89}}, .name = "conv2d_74_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.29135704040527344}}, .zeropoint = {1, (const int16_t[1]){101}}, .name = "eltwise_76_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.30816176533699036}}, .zeropoint = {1, (const int16_t[1]){102}}, .name = "eltwise_78_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.3592980206012726}}, .zeropoint = {1, (const int16_t[1]){69}}, .name = "eltwise_80_output" },
   { .size_bytes = 2304, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 36, 64}}, .scale = {1, (const float[1]){0.08129635453224182}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_82_output" },
   { .size_bytes = 64, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 64}}, .scale = {1, (const float[1]){0.08129635453224182}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "slice_84_gather_0_output" },
   { .size_bytes = 18, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 18}}, .scale = {1, (const float[1]){0.015330422669649124}}, .zeropoint = {1, (const int16_t[1]){-88}}, .name = "gemm_85_output" }
  },
  .nodes = (const stai_node_details[62]){
    {.id = 0, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* pad_0 */
    {.id = 9, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* slice_9 */
    {.id = 18, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_18 */
    {.id = 7, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* slice_7 */
    {.id = 16, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* conv2d_16 */
    {.id = 5, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* slice_5 */
    {.id = 14, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* conv2d_14 */
    {.id = 3, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* slice_3 */
    {.id = 12, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* conv2d_12 */
    {.id = 1, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* slice_1 */
    {.id = 11, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* conv2d_11 */
    {.id = 13, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){11, 9}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* eltwise_13 */
    {.id = 15, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){12, 7}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* eltwise_15 */
    {.id = 17, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){13, 5}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* eltwise_17 */
    {.id = 19, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){14, 3}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* eltwise_19 */
    {.id = 21, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* pad_21 */
    {.id = 30, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* slice_30 */
    {.id = 39, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* conv2d_39 */
    {.id = 28, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* slice_28 */
    {.id = 37, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* conv2d_37 */
    {.id = 26, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* slice_26 */
    {.id = 35, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* conv2d_35 */
    {.id = 24, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* slice_24 */
    {.id = 33, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){23}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* conv2d_33 */
    {.id = 22, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* slice_22 */
    {.id = 32, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* conv2d_32 */
    {.id = 34, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){26, 24}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* eltwise_34 */
    {.id = 36, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){27, 22}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* eltwise_36 */
    {.id = 38, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){28, 20}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* eltwise_38 */
    {.id = 40, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){29, 18}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* eltwise_40 */
    {.id = 42, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* pad_42 */
    {.id = 51, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* slice_51 */
    {.id = 60, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* conv2d_60 */
    {.id = 49, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* slice_49 */
    {.id = 58, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* conv2d_58 */
    {.id = 47, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){36}} }, /* slice_47 */
    {.id = 56, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){37}} }, /* conv2d_56 */
    {.id = 45, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){38}} }, /* slice_45 */
    {.id = 54, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){38}}, .output_tensors = {1, (const int32_t[1]){39}} }, /* conv2d_54 */
    {.id = 43, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){40}} }, /* slice_43 */
    {.id = 53, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){40}}, .output_tensors = {1, (const int32_t[1]){41}} }, /* conv2d_53 */
    {.id = 55, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){41, 39}}, .output_tensors = {1, (const int32_t[1]){42}} }, /* eltwise_55 */
    {.id = 57, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){42, 37}}, .output_tensors = {1, (const int32_t[1]){43}} }, /* eltwise_57 */
    {.id = 59, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){43, 35}}, .output_tensors = {1, (const int32_t[1]){44}} }, /* eltwise_59 */
    {.id = 61, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){44, 33}}, .output_tensors = {1, (const int32_t[1]){45}} }, /* eltwise_61 */
    {.id = 63, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){45}}, .output_tensors = {1, (const int32_t[1]){46}} }, /* pad_63 */
    {.id = 72, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){47}} }, /* slice_72 */
    {.id = 81, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){47}}, .output_tensors = {1, (const int32_t[1]){48}} }, /* conv2d_81 */
    {.id = 70, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){49}} }, /* slice_70 */
    {.id = 79, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){49}}, .output_tensors = {1, (const int32_t[1]){50}} }, /* conv2d_79 */
    {.id = 68, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){51}} }, /* slice_68 */
    {.id = 77, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){51}}, .output_tensors = {1, (const int32_t[1]){52}} }, /* conv2d_77 */
    {.id = 66, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){53}} }, /* slice_66 */
    {.id = 75, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){53}}, .output_tensors = {1, (const int32_t[1]){54}} }, /* conv2d_75 */
    {.id = 64, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){55}} }, /* slice_64 */
    {.id = 74, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){55}}, .output_tensors = {1, (const int32_t[1]){56}} }, /* conv2d_74 */
    {.id = 76, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){56, 54}}, .output_tensors = {1, (const int32_t[1]){57}} }, /* eltwise_76 */
    {.id = 78, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){57, 52}}, .output_tensors = {1, (const int32_t[1]){58}} }, /* eltwise_78 */
    {.id = 80, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){58, 50}}, .output_tensors = {1, (const int32_t[1]){59}} }, /* eltwise_80 */
    {.id = 82, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){59, 48}}, .output_tensors = {1, (const int32_t[1]){60}} }, /* eltwise_82 */
    {.id = 84, .type = AI_LAYER_GATHER_TYPE, .input_tensors = {1, (const int32_t[1]){60}}, .output_tensors = {1, (const int32_t[1]){61}} }, /* slice_84_gather_0 */
    {.id = 85, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){61}}, .output_tensors = {1, (const int32_t[1]){62}} } /* gemm_85 */
  },
  .n_nodes = 62
};
#endif

