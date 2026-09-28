/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-09-26T20:49:54+0600
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
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

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "ai_lite_inspect.h"

#include "lite_operators.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x69a374bd3eb44505460e89eb867de41d"
#define _STAI_NETWORK_DATETIME            "2026-09-26T20:49:54+0600"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 1),
  .tool_version = STAI_INIT_VERSION(4, 0, 1),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 36, 5),
      STAI_DECLARE_ARRAY(float, 1, 0.020258275792002678f),
      STAI_DECLARE_ARRAY(int16_t, 1, -79)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 18),
      STAI_DECLARE_ARRAY(float, 1, 0.015330422669649124f),
      STAI_DECLARE_ARRAY(int16_t, 1, -88)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 16768),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 41708),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pad_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020258275792002678f),
    AI_PACK_INTQ_ZP(-79)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_9_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020258275792002678f),
    AI_PACK_INTQ_ZP(-79)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020258275792002678f),
    AI_PACK_INTQ_ZP(-79)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020258275792002678f),
    AI_PACK_INTQ_ZP(-79)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_3_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020258275792002678f),
    AI_PACK_INTQ_ZP(-79)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_1_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020258275792002678f),
    AI_PACK_INTQ_ZP(-79)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_12_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021649004891514778f),
    AI_PACK_INTQ_ZP(-31)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029205355793237686f),
    AI_PACK_INTQ_ZP(11)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_13_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04217273369431496f),
    AI_PACK_INTQ_ZP(38)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_14_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.024298058822751045f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_15_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.043119899928569794f),
    AI_PACK_INTQ_ZP(35)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_16_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0321701355278492f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_17_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04443957656621933f),
    AI_PACK_INTQ_ZP(23)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_18_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.059947796165943146f),
    AI_PACK_INTQ_ZP(55)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_19_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018089503049850464f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pad_21_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018089503049850464f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_30_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018089503049850464f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_28_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018089503049850464f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_26_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018089503049850464f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_24_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018089503049850464f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_22_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018089503049850464f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_33_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08613935112953186f),
    AI_PACK_INTQ_ZP(84)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_32_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11588045209646225f),
    AI_PACK_INTQ_ZP(88)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_34_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.16412462294101715f),
    AI_PACK_INTQ_ZP(87)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_35_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06250172108411789f),
    AI_PACK_INTQ_ZP(75)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_36_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1870257556438446f),
    AI_PACK_INTQ_ZP(88)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_37_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.050918929278850555f),
    AI_PACK_INTQ_ZP(70)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_38_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1870257556438446f),
    AI_PACK_INTQ_ZP(88)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_39_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10019893199205399f),
    AI_PACK_INTQ_ZP(69)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_40_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02584196999669075f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pad_42_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02584196999669075f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_51_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02584196999669075f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_49_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02584196999669075f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_47_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02584196999669075f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_45_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02584196999669075f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_43_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02584196999669075f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_54_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05569436401128769f),
    AI_PACK_INTQ_ZP(37)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_53_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09451217204332352f),
    AI_PACK_INTQ_ZP(47)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_55_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11401693522930145f),
    AI_PACK_INTQ_ZP(58)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_56_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06863502413034439f),
    AI_PACK_INTQ_ZP(59)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_57_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13104695081710815f),
    AI_PACK_INTQ_ZP(67)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_58_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09137392789125443f),
    AI_PACK_INTQ_ZP(87)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_59_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15027357637882233f),
    AI_PACK_INTQ_ZP(42)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_60_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14971131086349487f),
    AI_PACK_INTQ_ZP(64)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_61_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05029941350221634f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pad_63_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05029941350221634f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_72_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05029941350221634f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_70_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05029941350221634f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_68_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05029941350221634f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_66_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05029941350221634f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_64_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05029941350221634f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_75_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14425751566886902f),
    AI_PACK_INTQ_ZP(79)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_74_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1846686601638794f),
    AI_PACK_INTQ_ZP(89)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_76_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.29135704040527344f),
    AI_PACK_INTQ_ZP(101)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_77_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12077406048774719f),
    AI_PACK_INTQ_ZP(91)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_78_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.30816176533699036f),
    AI_PACK_INTQ_ZP(102)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_79_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1060132309794426f),
    AI_PACK_INTQ_ZP(75)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_80_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.3592980206012726f),
    AI_PACK_INTQ_ZP(69)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_81_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19084693491458893f),
    AI_PACK_INTQ_ZP(39)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_82_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08129635453224182f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_84_gather_0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08129635453224182f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_85_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015330422669649124f),
    AI_PACK_INTQ_ZP(-88)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_85_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 18,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0014587340410798788f, 0.001442906679585576f, 0.0013340138830244541f, 0.0012979652965441346f, 0.001624617725610733f, 0.0015973328845575452f, 0.001421899301931262f, 0.0013797698775306344f, 0.0013943007215857506f, 0.0011846947018057108f, 0.001068683573976159f, 0.0010646070586517453f, 0.001700808061286807f, 0.0018655165331438184f, 0.0017885108245536685f, 0.0016854899004101753f, 0.0015730595914646983f, 0.0014594895765185356f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  pad_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 200, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  slice_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 180, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  slice_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 180, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  slice_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 180, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  slice_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 180, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  slice_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 180, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_19_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  pad_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1408, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  slice_30_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  slice_28_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  slice_26_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  slice_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  slice_22_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_33_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_34_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_36_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_37_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_38_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_40_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  pad_42_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1664, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  slice_51_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  slice_49_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  slice_47_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  slice_45_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  slice_43_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_55_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_56_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_57_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_58_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_59_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_60_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_61_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  pad_63_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4352, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  slice_72_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  slice_70_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  slice_68_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  slice_66_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  slice_64_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_75_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_74_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_76_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_77_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_78_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_79_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_80_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_81_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_82_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  slice_84_gather_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  slice_84_gather_0_placeholder_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  gemm_85_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 18, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  gemm_85_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  gemm_85_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 18, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  gemm_85_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 154, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  pad_0_output, AI_STATIC,
  104, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 1, 40), AI_STRIDE_INIT(4, 1, 1, 5, 5),
  1, &pad_0_output_array, &pad_0_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  slice_9_output, AI_STATIC,
  149, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 1, 36), AI_STRIDE_INIT(4, 1, 1, 5, 5),
  1, &slice_9_output_array, &slice_9_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  slice_7_output, AI_STATIC,
  145, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 1, 36), AI_STRIDE_INIT(4, 1, 1, 5, 5),
  1, &slice_7_output_array, &slice_7_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  slice_5_output, AI_STATIC,
  133, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 1, 36), AI_STRIDE_INIT(4, 1, 1, 5, 5),
  1, &slice_5_output_array, &slice_5_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  slice_3_output, AI_STATIC,
  121, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 1, 36), AI_STRIDE_INIT(4, 1, 1, 5, 5),
  1, &slice_3_output_array, &slice_3_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  slice_1_output, AI_STATIC,
  109, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 1, 36), AI_STRIDE_INIT(4, 1, 1, 5, 5),
  1, &slice_1_output_array, &slice_1_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_11_output_array, &conv2d_11_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_12_output_array, &conv2d_12_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_13_output, AI_STATIC,
  80, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &eltwise_13_output_array, &eltwise_13_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_14_output_array, &conv2d_14_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_15_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &eltwise_15_output_array, &eltwise_15_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_16_output_array, &conv2d_16_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_17_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &eltwise_17_output_array, &eltwise_17_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_18_output_array, &conv2d_18_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_19_output, AI_STATIC,
  83, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &eltwise_19_output_array, &eltwise_19_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  pad_21_output, AI_STATIC,
  105, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 44), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &pad_21_output_array, &pad_21_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  slice_30_output, AI_STATIC,
  119, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_30_output_array, &slice_30_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  slice_28_output, AI_STATIC,
  117, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_28_output_array, &slice_28_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  slice_26_output, AI_STATIC,
  115, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_26_output_array, &slice_26_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  slice_24_output, AI_STATIC,
  113, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_24_output_array, &slice_24_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  slice_22_output, AI_STATIC,
  111, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_22_output_array, &slice_22_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_32_output_array, &conv2d_32_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_33_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_33_output_array, &conv2d_33_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_34_output, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &eltwise_34_output_array, &eltwise_34_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_35_output_array, &conv2d_35_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_36_output, AI_STATIC,
  86, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &eltwise_36_output_array, &eltwise_36_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_37_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_37_output_array, &conv2d_37_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_38_output, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &eltwise_38_output_array, &eltwise_38_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &conv2d_39_output_array, &conv2d_39_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_40_output, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 36, 1), AI_STRIDE_INIT(4, 1, 1, 32, 1152),
  1, &eltwise_40_output_array, &eltwise_40_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  pad_42_output, AI_STATIC,
  106, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 52), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &pad_42_output_array, &pad_42_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  slice_51_output, AI_STATIC,
  131, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_51_output_array, &slice_51_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  slice_49_output, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_49_output_array, &slice_49_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  slice_47_output, AI_STATIC,
  127, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_47_output_array, &slice_47_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  slice_45_output, AI_STATIC,
  125, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_45_output_array, &slice_45_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  slice_43_output, AI_STATIC,
  123, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 36), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &slice_43_output_array, &slice_43_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_53_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_53_output_array, &conv2d_53_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_54_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_54_output_array, &conv2d_54_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_55_output, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &eltwise_55_output_array, &eltwise_55_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_56_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_56_output_array, &conv2d_56_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_57_output, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &eltwise_57_output_array, &eltwise_57_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_58_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_58_output_array, &conv2d_58_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_59_output, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &eltwise_59_output_array, &eltwise_59_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_60_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_60_output_array, &conv2d_60_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_61_output, AI_STATIC,
  93, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &eltwise_61_output_array, &eltwise_61_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  pad_63_output, AI_STATIC,
  107, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 68), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &pad_63_output_array, &pad_63_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  slice_72_output, AI_STATIC,
  143, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 36), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &slice_72_output_array, &slice_72_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  slice_70_output, AI_STATIC,
  141, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 36), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &slice_70_output_array, &slice_70_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  slice_68_output, AI_STATIC,
  139, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 36), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &slice_68_output_array, &slice_68_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  slice_66_output, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 36), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &slice_66_output_array, &slice_66_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  slice_64_output, AI_STATIC,
  135, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 36), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &slice_64_output_array, &slice_64_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_74_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_74_output_array, &conv2d_74_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_75_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_75_output_array, &conv2d_75_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_76_output, AI_STATIC,
  95, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &eltwise_76_output_array, &eltwise_76_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_77_output, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_77_output_array, &conv2d_77_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_78_output, AI_STATIC,
  96, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &eltwise_78_output_array, &eltwise_78_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_79_output, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_79_output_array, &conv2d_79_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_80_output, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &eltwise_80_output_array, &eltwise_80_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_output, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &conv2d_81_output_array, &conv2d_81_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_82_output, AI_STATIC,
  98, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 36, 1), AI_STRIDE_INIT(4, 1, 1, 64, 2304),
  1, &eltwise_82_output_array, &eltwise_82_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_82_output0, AI_STATIC,
  99, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 36), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &eltwise_82_output_array, &eltwise_82_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  slice_84_gather_0_output, AI_STATIC,
  147, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &slice_84_gather_0_output_array, &slice_84_gather_0_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  slice_84_gather_0_placeholder, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &slice_84_gather_0_placeholder_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  gemm_85_bias, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 1, 18, 1, 1), AI_STRIDE_INIT(4, 4, 4, 72, 72),
  1, &gemm_85_bias_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  gemm_85_output, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 18, 1, 1), AI_STRIDE_INIT(4, 1, 1, 18, 18),
  1, &gemm_85_output_array, &gemm_85_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  gemm_85_scratch0, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 154, 1, 1), AI_STRIDE_INIT(4, 2, 2, 308, 308),
  1, &gemm_85_scratch0_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  gemm_85_weights, AI_STATIC,
  103, 0x1,
  AI_SHAPE_INIT(4, 64, 18, 1, 1), AI_STRIDE_INIT(4, 1, 64, 1152, 1152),
  1, &gemm_85_weights_array, &gemm_85_weights_array_intq)



AI_STATIC_CONST ai_u8 slice_9_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_9_axes, AI_ARRAY_FORMAT_U8,
    slice_9_axes_data, slice_9_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_9_starts_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_9_starts, AI_ARRAY_FORMAT_S16,
    slice_9_starts_data, slice_9_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_9_ends_data[] = { 40 };
AI_ARRAY_OBJ_DECLARE(
    slice_9_ends, AI_ARRAY_FORMAT_S16,
    slice_9_ends_data, slice_9_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_9_layer, 9,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_9_chain,
  NULL, &slice_9_layer, AI_STATIC, 
  .axes = &slice_9_axes, 
  .starts = &slice_9_starts, 
  .ends = &slice_9_ends, 
)


AI_STATIC_CONST ai_u8 slice_7_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_axes, AI_ARRAY_FORMAT_U8,
    slice_7_axes_data, slice_7_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_7_starts_data[] = { 3 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_starts, AI_ARRAY_FORMAT_S16,
    slice_7_starts_data, slice_7_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_7_ends_data[] = { 39 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_ends, AI_ARRAY_FORMAT_S16,
    slice_7_ends_data, slice_7_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_7_layer, 7,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_7_chain,
  NULL, &slice_7_layer, AI_STATIC, 
  .axes = &slice_7_axes, 
  .starts = &slice_7_starts, 
  .ends = &slice_7_ends, 
)


AI_STATIC_CONST ai_u8 slice_5_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_axes, AI_ARRAY_FORMAT_U8,
    slice_5_axes_data, slice_5_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_5_starts_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_starts, AI_ARRAY_FORMAT_S16,
    slice_5_starts_data, slice_5_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_5_ends_data[] = { 38 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_ends, AI_ARRAY_FORMAT_S16,
    slice_5_ends_data, slice_5_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_5_layer, 5,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_5_chain,
  NULL, &slice_5_layer, AI_STATIC, 
  .axes = &slice_5_axes, 
  .starts = &slice_5_starts, 
  .ends = &slice_5_ends, 
)


AI_STATIC_CONST ai_u8 slice_3_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_axes, AI_ARRAY_FORMAT_U8,
    slice_3_axes_data, slice_3_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_3_starts_data[] = { 1 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_starts, AI_ARRAY_FORMAT_S16,
    slice_3_starts_data, slice_3_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_3_ends_data[] = { 37 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_ends, AI_ARRAY_FORMAT_S16,
    slice_3_ends_data, slice_3_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_3_layer, 3,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_3_chain,
  NULL, &slice_3_layer, AI_STATIC, 
  .axes = &slice_3_axes, 
  .starts = &slice_3_starts, 
  .ends = &slice_3_ends, 
)


AI_STATIC_CONST ai_u8 slice_1_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_axes, AI_ARRAY_FORMAT_U8,
    slice_1_axes_data, slice_1_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_1_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_starts, AI_ARRAY_FORMAT_S16,
    slice_1_starts_data, slice_1_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_1_ends_data[] = { 36 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_ends, AI_ARRAY_FORMAT_S16,
    slice_1_ends_data, slice_1_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_1_layer, 1,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_1_chain,
  NULL, &slice_1_layer, AI_STATIC, 
  .axes = &slice_1_axes, 
  .starts = &slice_1_starts, 
  .ends = &slice_1_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_11_output, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_13_layer, 13,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_13_chain,
  NULL, &eltwise_13_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_13_output, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_15_layer, 15,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_15_chain,
  NULL, &eltwise_15_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_15_output, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_17_layer, 17,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_17_chain,
  NULL, &eltwise_17_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_17_output, &conv2d_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_19_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_19_layer, 19,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_19_chain,
  NULL, &eltwise_19_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_30_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_30_axes, AI_ARRAY_FORMAT_U8,
    slice_30_axes_data, slice_30_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_30_starts_data[] = { 8 };
AI_ARRAY_OBJ_DECLARE(
    slice_30_starts, AI_ARRAY_FORMAT_S16,
    slice_30_starts_data, slice_30_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_30_ends_data[] = { 44 };
AI_ARRAY_OBJ_DECLARE(
    slice_30_ends, AI_ARRAY_FORMAT_S16,
    slice_30_ends_data, slice_30_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_30_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_30_layer, 30,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_30_chain,
  NULL, &slice_30_layer, AI_STATIC, 
  .axes = &slice_30_axes, 
  .starts = &slice_30_starts, 
  .ends = &slice_30_ends, 
)


AI_STATIC_CONST ai_u8 slice_28_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_28_axes, AI_ARRAY_FORMAT_U8,
    slice_28_axes_data, slice_28_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_28_starts_data[] = { 6 };
AI_ARRAY_OBJ_DECLARE(
    slice_28_starts, AI_ARRAY_FORMAT_S16,
    slice_28_starts_data, slice_28_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_28_ends_data[] = { 42 };
AI_ARRAY_OBJ_DECLARE(
    slice_28_ends, AI_ARRAY_FORMAT_S16,
    slice_28_ends_data, slice_28_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_28_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_28_layer, 28,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_28_chain,
  NULL, &slice_28_layer, AI_STATIC, 
  .axes = &slice_28_axes, 
  .starts = &slice_28_starts, 
  .ends = &slice_28_ends, 
)


AI_STATIC_CONST ai_u8 slice_26_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_26_axes, AI_ARRAY_FORMAT_U8,
    slice_26_axes_data, slice_26_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_26_starts_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_26_starts, AI_ARRAY_FORMAT_S16,
    slice_26_starts_data, slice_26_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_26_ends_data[] = { 40 };
AI_ARRAY_OBJ_DECLARE(
    slice_26_ends, AI_ARRAY_FORMAT_S16,
    slice_26_ends_data, slice_26_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_26_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_26_layer, 26,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_26_chain,
  NULL, &slice_26_layer, AI_STATIC, 
  .axes = &slice_26_axes, 
  .starts = &slice_26_starts, 
  .ends = &slice_26_ends, 
)


AI_STATIC_CONST ai_u8 slice_24_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_24_axes, AI_ARRAY_FORMAT_U8,
    slice_24_axes_data, slice_24_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_24_starts_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_24_starts, AI_ARRAY_FORMAT_S16,
    slice_24_starts_data, slice_24_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_24_ends_data[] = { 38 };
AI_ARRAY_OBJ_DECLARE(
    slice_24_ends, AI_ARRAY_FORMAT_S16,
    slice_24_ends_data, slice_24_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_24_layer, 24,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_24_chain,
  NULL, &slice_24_layer, AI_STATIC, 
  .axes = &slice_24_axes, 
  .starts = &slice_24_starts, 
  .ends = &slice_24_ends, 
)


AI_STATIC_CONST ai_u8 slice_22_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_22_axes, AI_ARRAY_FORMAT_U8,
    slice_22_axes_data, slice_22_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_22_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_22_starts, AI_ARRAY_FORMAT_S16,
    slice_22_starts_data, slice_22_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_22_ends_data[] = { 36 };
AI_ARRAY_OBJ_DECLARE(
    slice_22_ends, AI_ARRAY_FORMAT_S16,
    slice_22_ends_data, slice_22_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_22_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_22_layer, 22,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_22_chain,
  NULL, &slice_22_layer, AI_STATIC, 
  .axes = &slice_22_axes, 
  .starts = &slice_22_starts, 
  .ends = &slice_22_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_32_output, &conv2d_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_34_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_34_layer, 34,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_34_chain,
  NULL, &eltwise_34_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_34_output, &conv2d_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_36_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_36_layer, 36,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_36_chain,
  NULL, &eltwise_36_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_38_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_36_output, &conv2d_37_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_38_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_38_layer, 38,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_38_chain,
  NULL, &eltwise_38_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_38_output, &conv2d_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_40_layer, 40,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_40_chain,
  NULL, &eltwise_40_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_51_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_51_axes, AI_ARRAY_FORMAT_U8,
    slice_51_axes_data, slice_51_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_51_starts_data[] = { 16 };
AI_ARRAY_OBJ_DECLARE(
    slice_51_starts, AI_ARRAY_FORMAT_S16,
    slice_51_starts_data, slice_51_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_51_ends_data[] = { 52 };
AI_ARRAY_OBJ_DECLARE(
    slice_51_ends, AI_ARRAY_FORMAT_S16,
    slice_51_ends_data, slice_51_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_51_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_51_layer, 51,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_51_chain,
  NULL, &slice_51_layer, AI_STATIC, 
  .axes = &slice_51_axes, 
  .starts = &slice_51_starts, 
  .ends = &slice_51_ends, 
)


AI_STATIC_CONST ai_u8 slice_49_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_49_axes, AI_ARRAY_FORMAT_U8,
    slice_49_axes_data, slice_49_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_49_starts_data[] = { 12 };
AI_ARRAY_OBJ_DECLARE(
    slice_49_starts, AI_ARRAY_FORMAT_S16,
    slice_49_starts_data, slice_49_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_49_ends_data[] = { 48 };
AI_ARRAY_OBJ_DECLARE(
    slice_49_ends, AI_ARRAY_FORMAT_S16,
    slice_49_ends_data, slice_49_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_49_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_49_layer, 49,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_49_chain,
  NULL, &slice_49_layer, AI_STATIC, 
  .axes = &slice_49_axes, 
  .starts = &slice_49_starts, 
  .ends = &slice_49_ends, 
)


AI_STATIC_CONST ai_u8 slice_47_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_47_axes, AI_ARRAY_FORMAT_U8,
    slice_47_axes_data, slice_47_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_47_starts_data[] = { 8 };
AI_ARRAY_OBJ_DECLARE(
    slice_47_starts, AI_ARRAY_FORMAT_S16,
    slice_47_starts_data, slice_47_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_47_ends_data[] = { 44 };
AI_ARRAY_OBJ_DECLARE(
    slice_47_ends, AI_ARRAY_FORMAT_S16,
    slice_47_ends_data, slice_47_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_47_layer, 47,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_47_chain,
  NULL, &slice_47_layer, AI_STATIC, 
  .axes = &slice_47_axes, 
  .starts = &slice_47_starts, 
  .ends = &slice_47_ends, 
)


AI_STATIC_CONST ai_u8 slice_45_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_45_axes, AI_ARRAY_FORMAT_U8,
    slice_45_axes_data, slice_45_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_45_starts_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_45_starts, AI_ARRAY_FORMAT_S16,
    slice_45_starts_data, slice_45_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_45_ends_data[] = { 40 };
AI_ARRAY_OBJ_DECLARE(
    slice_45_ends, AI_ARRAY_FORMAT_S16,
    slice_45_ends_data, slice_45_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_45_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_45_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_45_layer, 45,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_45_chain,
  NULL, &slice_45_layer, AI_STATIC, 
  .axes = &slice_45_axes, 
  .starts = &slice_45_starts, 
  .ends = &slice_45_ends, 
)


AI_STATIC_CONST ai_u8 slice_43_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_43_axes, AI_ARRAY_FORMAT_U8,
    slice_43_axes_data, slice_43_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_43_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_43_starts, AI_ARRAY_FORMAT_S16,
    slice_43_starts_data, slice_43_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_43_ends_data[] = { 36 };
AI_ARRAY_OBJ_DECLARE(
    slice_43_ends, AI_ARRAY_FORMAT_S16,
    slice_43_ends_data, slice_43_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_43_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_43_layer, 43,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_43_chain,
  NULL, &slice_43_layer, AI_STATIC, 
  .axes = &slice_43_axes, 
  .starts = &slice_43_starts, 
  .ends = &slice_43_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_55_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_53_output, &conv2d_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_55_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_55_layer, 55,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_55_chain,
  NULL, &eltwise_55_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_57_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_55_output, &conv2d_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_57_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_57_layer, 57,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_57_chain,
  NULL, &eltwise_57_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_59_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_57_output, &conv2d_58_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_59_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_59_layer, 59,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_59_chain,
  NULL, &eltwise_59_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_61_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_59_output, &conv2d_60_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_61_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_61_layer, 61,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_61_chain,
  NULL, &eltwise_61_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_72_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_72_axes, AI_ARRAY_FORMAT_U8,
    slice_72_axes_data, slice_72_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_72_starts_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_72_starts, AI_ARRAY_FORMAT_S16,
    slice_72_starts_data, slice_72_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_72_ends_data[] = { 68 };
AI_ARRAY_OBJ_DECLARE(
    slice_72_ends, AI_ARRAY_FORMAT_S16,
    slice_72_ends_data, slice_72_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_72_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_72_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_72_layer, 72,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_72_chain,
  NULL, &slice_72_layer, AI_STATIC, 
  .axes = &slice_72_axes, 
  .starts = &slice_72_starts, 
  .ends = &slice_72_ends, 
)


AI_STATIC_CONST ai_u8 slice_70_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_70_axes, AI_ARRAY_FORMAT_U8,
    slice_70_axes_data, slice_70_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_70_starts_data[] = { 24 };
AI_ARRAY_OBJ_DECLARE(
    slice_70_starts, AI_ARRAY_FORMAT_S16,
    slice_70_starts_data, slice_70_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_70_ends_data[] = { 60 };
AI_ARRAY_OBJ_DECLARE(
    slice_70_ends, AI_ARRAY_FORMAT_S16,
    slice_70_ends_data, slice_70_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_70_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_70_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_70_layer, 70,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_70_chain,
  NULL, &slice_70_layer, AI_STATIC, 
  .axes = &slice_70_axes, 
  .starts = &slice_70_starts, 
  .ends = &slice_70_ends, 
)


AI_STATIC_CONST ai_u8 slice_68_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_68_axes, AI_ARRAY_FORMAT_U8,
    slice_68_axes_data, slice_68_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_68_starts_data[] = { 16 };
AI_ARRAY_OBJ_DECLARE(
    slice_68_starts, AI_ARRAY_FORMAT_S16,
    slice_68_starts_data, slice_68_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_68_ends_data[] = { 52 };
AI_ARRAY_OBJ_DECLARE(
    slice_68_ends, AI_ARRAY_FORMAT_S16,
    slice_68_ends_data, slice_68_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_68_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_68_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_68_layer, 68,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_68_chain,
  NULL, &slice_68_layer, AI_STATIC, 
  .axes = &slice_68_axes, 
  .starts = &slice_68_starts, 
  .ends = &slice_68_ends, 
)


AI_STATIC_CONST ai_u8 slice_66_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_66_axes, AI_ARRAY_FORMAT_U8,
    slice_66_axes_data, slice_66_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_66_starts_data[] = { 8 };
AI_ARRAY_OBJ_DECLARE(
    slice_66_starts, AI_ARRAY_FORMAT_S16,
    slice_66_starts_data, slice_66_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_66_ends_data[] = { 44 };
AI_ARRAY_OBJ_DECLARE(
    slice_66_ends, AI_ARRAY_FORMAT_S16,
    slice_66_ends_data, slice_66_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_66_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_66_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_66_layer, 66,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_66_chain,
  NULL, &slice_66_layer, AI_STATIC, 
  .axes = &slice_66_axes, 
  .starts = &slice_66_starts, 
  .ends = &slice_66_ends, 
)


AI_STATIC_CONST ai_u8 slice_64_axes_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_64_axes, AI_ARRAY_FORMAT_U8,
    slice_64_axes_data, slice_64_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_64_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_64_starts, AI_ARRAY_FORMAT_S16,
    slice_64_starts_data, slice_64_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_64_ends_data[] = { 36 };
AI_ARRAY_OBJ_DECLARE(
    slice_64_ends, AI_ARRAY_FORMAT_S16,
    slice_64_ends_data, slice_64_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_64_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_64_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_64_layer, 64,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_64_chain,
  NULL, &slice_64_layer, AI_STATIC, 
  .axes = &slice_64_axes, 
  .starts = &slice_64_starts, 
  .ends = &slice_64_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_76_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_74_output, &conv2d_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_76_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_76_layer, 76,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_76_chain,
  NULL, &eltwise_76_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_78_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_76_output, &conv2d_77_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_78_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_78_layer, 78,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_78_chain,
  NULL, &eltwise_78_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_80_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_78_output, &conv2d_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_80_layer, 80,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_80_chain,
  NULL, &eltwise_80_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_80_output, &conv2d_81_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_82_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_82_layer, 82,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_82_chain,
  NULL, &eltwise_82_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_84_gather_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_82_output0, &slice_84_gather_0_placeholder),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_84_gather_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_84_gather_0_layer, 84,
  GATHER_TYPE, 0x0, NULL,
  gather, forward_gather,
  &slice_84_gather_0_chain,
  NULL, &slice_84_gather_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_85_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_84_gather_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_85_weights, &gemm_85_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_85_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_85_layer, 85,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_85_chain,
  NULL, &gemm_85_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_slice_slice_9(_stai_network_context* net_ctx)
{
  pad_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 952);
  pad_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 952);
  slice_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  slice_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  _STAI_NETWORK_EVENT_NODE_START_CB(9, 1, { pad_0_output.data->data});
  forward_slice(&slice_9_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(9, 1, { slice_9_output.data->data});
}
void forward_lite_slice_slice_7(_stai_network_context* net_ctx)
{
  pad_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 952);
  pad_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 952);
  slice_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  slice_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  _STAI_NETWORK_EVENT_NODE_START_CB(7, 1, { pad_0_output.data->data});
  forward_slice(&slice_7_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(7, 1, { slice_7_output.data->data});
}
void forward_lite_slice_slice_5(_stai_network_context* net_ctx)
{
  pad_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 952);
  pad_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 952);
  slice_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  slice_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 1, { pad_0_output.data->data});
  forward_slice(&slice_5_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, { slice_5_output.data->data});
}
void forward_lite_slice_slice_3(_stai_network_context* net_ctx)
{
  pad_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 952);
  pad_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 952);
  slice_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  slice_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  _STAI_NETWORK_EVENT_NODE_START_CB(3, 1, { pad_0_output.data->data});
  forward_slice(&slice_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(3, 1, { slice_3_output.data->data});
}
void forward_lite_slice_slice_1(_stai_network_context* net_ctx)
{
  pad_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 952);
  pad_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 952);
  slice_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  slice_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  _STAI_NETWORK_EVENT_NODE_START_CB(1, 1, { pad_0_output.data->data});
  forward_slice(&slice_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(1, 1, { slice_1_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_13(_stai_network_context* net_ctx)
{
  conv2d_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 5128);
  conv2d_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5128);
  eltwise_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 2, { conv2d_11_output.data->data,conv2d_12_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_13_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, { eltwise_13_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_15(_stai_network_context* net_ctx)
{
  eltwise_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 3976);
  conv2d_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3976);
  eltwise_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 5128);
  eltwise_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5128);
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 2, { eltwise_13_output.data->data,conv2d_14_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_15_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, { eltwise_15_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_17(_stai_network_context* net_ctx)
{
  eltwise_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 5128);
  eltwise_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5128);
  conv2d_16_output_array.data = AI_PTR(net_ctx->_activations[0] + 2824);
  conv2d_16_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2824);
  eltwise_17_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_17_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(17, 2, { eltwise_15_output.data->data,conv2d_16_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_17_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(17, 1, { eltwise_17_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_19(_stai_network_context* net_ctx)
{
  eltwise_17_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_17_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_18_output_array.data = AI_PTR(net_ctx->_activations[0] + 1672);
  conv2d_18_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1672);
  eltwise_19_output_array.data = AI_PTR(net_ctx->_activations[0] + 2824);
  eltwise_19_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2824);
  _STAI_NETWORK_EVENT_NODE_START_CB(19, 2, { eltwise_17_output.data->data,conv2d_18_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_19_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(19, 1, { eltwise_19_output.data->data});
}
void forward_lite_slice_slice_30(_stai_network_context* net_ctx)
{
  pad_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_30_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  slice_30_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_NETWORK_EVENT_NODE_START_CB(30, 1, { pad_21_output.data->data});
  forward_slice(&slice_30_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(30, 1, { slice_30_output.data->data});
}
void forward_lite_slice_slice_28(_stai_network_context* net_ctx)
{
  pad_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_28_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  slice_28_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_NETWORK_EVENT_NODE_START_CB(28, 1, { pad_21_output.data->data});
  forward_slice(&slice_28_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(28, 1, { slice_28_output.data->data});
}
void forward_lite_slice_slice_26(_stai_network_context* net_ctx)
{
  pad_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_26_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  slice_26_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, { pad_21_output.data->data});
  forward_slice(&slice_26_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, { slice_26_output.data->data});
}
void forward_lite_slice_slice_24(_stai_network_context* net_ctx)
{
  pad_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_24_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  slice_24_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_NETWORK_EVENT_NODE_START_CB(24, 1, { pad_21_output.data->data});
  forward_slice(&slice_24_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(24, 1, { slice_24_output.data->data});
}
void forward_lite_slice_slice_22(_stai_network_context* net_ctx)
{
  pad_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_22_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  slice_22_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 1, { pad_21_output.data->data});
  forward_slice(&slice_22_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, { slice_22_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_34(_stai_network_context* net_ctx)
{
  conv2d_32_output_array.data = AI_PTR(net_ctx->_activations[0] + 7616);
  conv2d_32_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7616);
  conv2d_33_output_array.data = AI_PTR(net_ctx->_activations[0] + 6464);
  conv2d_33_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6464);
  eltwise_34_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_34_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 2, { conv2d_32_output.data->data,conv2d_33_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_34_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, { eltwise_34_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_36(_stai_network_context* net_ctx)
{
  eltwise_34_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_34_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_35_output_array.data = AI_PTR(net_ctx->_activations[0] + 5312);
  conv2d_35_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5312);
  eltwise_36_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  eltwise_36_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  _STAI_NETWORK_EVENT_NODE_START_CB(36, 2, { eltwise_34_output.data->data,conv2d_35_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_36_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(36, 1, { eltwise_36_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_38(_stai_network_context* net_ctx)
{
  eltwise_36_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  eltwise_36_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  conv2d_37_output_array.data = AI_PTR(net_ctx->_activations[0] + 4160);
  conv2d_37_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4160);
  eltwise_38_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_38_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(38, 2, { eltwise_36_output.data->data,conv2d_37_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_38_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(38, 1, { eltwise_38_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_40(_stai_network_context* net_ctx)
{
  eltwise_38_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_38_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_39_output_array.data = AI_PTR(net_ctx->_activations[0] + 3008);
  conv2d_39_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3008);
  eltwise_40_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  eltwise_40_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 2, { eltwise_38_output.data->data,conv2d_39_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_40_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, { eltwise_40_output.data->data});
}
void forward_lite_slice_slice_51(_stai_network_context* net_ctx)
{
  pad_42_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  pad_42_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  slice_51_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  slice_51_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(51, 1, { pad_42_output.data->data});
  forward_slice(&slice_51_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(51, 1, { slice_51_output.data->data});
}
void forward_lite_slice_slice_49(_stai_network_context* net_ctx)
{
  pad_42_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  pad_42_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  slice_49_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  slice_49_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(49, 1, { pad_42_output.data->data});
  forward_slice(&slice_49_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(49, 1, { slice_49_output.data->data});
}
void forward_lite_slice_slice_47(_stai_network_context* net_ctx)
{
  pad_42_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  pad_42_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  slice_47_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  slice_47_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(47, 1, { pad_42_output.data->data});
  forward_slice(&slice_47_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(47, 1, { slice_47_output.data->data});
}
void forward_lite_slice_slice_45(_stai_network_context* net_ctx)
{
  pad_42_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  pad_42_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  slice_45_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  slice_45_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, { pad_42_output.data->data});
  forward_slice(&slice_45_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, { slice_45_output.data->data});
}
void forward_lite_slice_slice_43(_stai_network_context* net_ctx)
{
  pad_42_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  pad_42_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  slice_43_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  slice_43_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(43, 1, { pad_42_output.data->data});
  forward_slice(&slice_43_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(43, 1, { slice_43_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_55(_stai_network_context* net_ctx)
{
  conv2d_53_output_array.data = AI_PTR(net_ctx->_activations[0] + 13184);
  conv2d_53_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 13184);
  conv2d_54_output_array.data = AI_PTR(net_ctx->_activations[0] + 10880);
  conv2d_54_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10880);
  eltwise_55_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_55_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(55, 2, { conv2d_53_output.data->data,conv2d_54_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_55_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(55, 1, { eltwise_55_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_57(_stai_network_context* net_ctx)
{
  eltwise_55_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_55_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_56_output_array.data = AI_PTR(net_ctx->_activations[0] + 8576);
  conv2d_56_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8576);
  eltwise_57_output_array.data = AI_PTR(net_ctx->_activations[0] + 10880);
  eltwise_57_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10880);
  _STAI_NETWORK_EVENT_NODE_START_CB(57, 2, { eltwise_55_output.data->data,conv2d_56_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_57_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(57, 1, { eltwise_57_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_59(_stai_network_context* net_ctx)
{
  eltwise_57_output_array.data = AI_PTR(net_ctx->_activations[0] + 10880);
  eltwise_57_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 10880);
  conv2d_58_output_array.data = AI_PTR(net_ctx->_activations[0] + 6272);
  conv2d_58_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6272);
  eltwise_59_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_59_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(59, 2, { eltwise_57_output.data->data,conv2d_58_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_59_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(59, 1, { eltwise_59_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_61(_stai_network_context* net_ctx)
{
  eltwise_59_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_59_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_60_output_array.data = AI_PTR(net_ctx->_activations[0] + 3968);
  conv2d_60_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3968);
  eltwise_61_output_array.data = AI_PTR(net_ctx->_activations[0] + 6272);
  eltwise_61_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6272);
  _STAI_NETWORK_EVENT_NODE_START_CB(61, 2, { eltwise_59_output.data->data,conv2d_60_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_61_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(61, 1, { eltwise_61_output.data->data});
}
void forward_lite_slice_slice_72(_stai_network_context* net_ctx)
{
  pad_63_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_63_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_72_output_array.data = AI_PTR(net_ctx->_activations[0] + 4352);
  slice_72_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4352);
  _STAI_NETWORK_EVENT_NODE_START_CB(72, 1, { pad_63_output.data->data});
  forward_slice(&slice_72_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(72, 1, { slice_72_output.data->data});
}
void forward_lite_slice_slice_70(_stai_network_context* net_ctx)
{
  pad_63_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_63_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_70_output_array.data = AI_PTR(net_ctx->_activations[0] + 4352);
  slice_70_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4352);
  _STAI_NETWORK_EVENT_NODE_START_CB(70, 1, { pad_63_output.data->data});
  forward_slice(&slice_70_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(70, 1, { slice_70_output.data->data});
}
void forward_lite_slice_slice_68(_stai_network_context* net_ctx)
{
  pad_63_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_63_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_68_output_array.data = AI_PTR(net_ctx->_activations[0] + 4352);
  slice_68_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4352);
  _STAI_NETWORK_EVENT_NODE_START_CB(68, 1, { pad_63_output.data->data});
  forward_slice(&slice_68_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(68, 1, { slice_68_output.data->data});
}
void forward_lite_slice_slice_66(_stai_network_context* net_ctx)
{
  pad_63_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_63_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_66_output_array.data = AI_PTR(net_ctx->_activations[0] + 4352);
  slice_66_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4352);
  _STAI_NETWORK_EVENT_NODE_START_CB(66, 1, { pad_63_output.data->data});
  forward_slice(&slice_66_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(66, 1, { slice_66_output.data->data});
}
void forward_lite_slice_slice_64(_stai_network_context* net_ctx)
{
  pad_63_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  pad_63_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_64_output_array.data = AI_PTR(net_ctx->_activations[0] + 4352);
  slice_64_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4352);
  _STAI_NETWORK_EVENT_NODE_START_CB(64, 1, { pad_63_output.data->data});
  forward_slice(&slice_64_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(64, 1, { slice_64_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_76(_stai_network_context* net_ctx)
{
  conv2d_74_output_array.data = AI_PTR(net_ctx->_activations[0] + 896);
  conv2d_74_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 896);
  conv2d_75_output_array.data = AI_PTR(net_ctx->_activations[0] + 14464);
  conv2d_75_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 14464);
  eltwise_76_output_array.data = AI_PTR(net_ctx->_activations[0] + 3200);
  eltwise_76_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3200);
  _STAI_NETWORK_EVENT_NODE_START_CB(76, 2, { conv2d_74_output.data->data,conv2d_75_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_76_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(76, 1, { eltwise_76_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_78(_stai_network_context* net_ctx)
{
  eltwise_76_output_array.data = AI_PTR(net_ctx->_activations[0] + 3200);
  eltwise_76_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3200);
  conv2d_77_output_array.data = AI_PTR(net_ctx->_activations[0] + 12160);
  conv2d_77_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12160);
  eltwise_78_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_78_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(78, 2, { eltwise_76_output.data->data,conv2d_77_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_78_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(78, 1, { eltwise_78_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_80(_stai_network_context* net_ctx)
{
  eltwise_78_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_78_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_79_output_array.data = AI_PTR(net_ctx->_activations[0] + 9856);
  conv2d_79_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 9856);
  eltwise_80_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  eltwise_80_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  _STAI_NETWORK_EVENT_NODE_START_CB(80, 2, { eltwise_78_output.data->data,conv2d_79_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_80_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(80, 1, { eltwise_80_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_82(_stai_network_context* net_ctx)
{
  eltwise_80_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  eltwise_80_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  conv2d_81_output_array.data = AI_PTR(net_ctx->_activations[0] + 7552);
  conv2d_81_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7552);
  eltwise_82_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_82_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(82, 2, { eltwise_80_output.data->data,conv2d_81_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_82_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(82, 1, { eltwise_82_output.data->data});
}
void forward_lite_gather_slice_84_gather_0(_stai_network_context* net_ctx)
{
  eltwise_82_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_82_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_84_gather_0_placeholder_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  slice_84_gather_0_placeholder_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  slice_84_gather_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  slice_84_gather_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  _STAI_NETWORK_EVENT_NODE_START_CB(84, 2, { eltwise_82_output0.data->data,slice_84_gather_0_placeholder.data->data});
  forward_gather(&slice_84_gather_0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(84, 1, { slice_84_gather_0_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_85(_stai_network_context* net_ctx)
{
  slice_84_gather_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 2304);
  slice_84_gather_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2304);
  gemm_85_weights_array.data = AI_PTR(net_ctx->_weights[0] + 40484);
  gemm_85_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 40484);
  gemm_85_bias_array.data = AI_PTR(net_ctx->_weights[0] + 41636);
  gemm_85_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 41636);
  gemm_85_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  gemm_85_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  gemm_85_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  gemm_85_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(85, 1, { slice_84_gather_0_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_85_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(85, 1, { gemm_85_output.data->data});
}

/*****************************************************************************/


static const ai_i8 pad_0_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-79);
static const ai_i16 pad_0_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_0_t_in_0_shape_h_const_u32 = 36;


static const ai_u16 conv2d_18_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_18_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_18_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_18_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_18_t_in_0_shape_ch_const_u16 = 5;
static const ai_u16 conv2d_18_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_18_t_in_0_fmt_zero_const_s8 = -79;
static const ai_i8 conv2d_18_t_out_0_fmt_zero_const_s8 = 55;
static const ai_float conv2d_18_t_in_0_fmt_scale_const_f32 = 0.020258275792002678f;
static const ai_float conv2d_18_t_out_0_fmt_scale_const_f32 = 0.059947796165943146f;
static const ai_float conv2d_18_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.005067957099527121f, 0.0028767662588506937f, 0.0056360624730587006f, 0.0026845349930226803f, 0.00652534794062376f, 0.003272122470661998f, 0.008499928750097752f, 0.015922697260975838f, 0.013015507720410824f, 0.003663925686851144f, 0.0027178567834198475f, 0.004780116956681013f, 0.003111938014626503f, 0.006033801008015871f, 0.007099959068000317f, 0.01946636661887169f, 0.0025204436387866735f, 0.0015926198102533817f, 0.0030835571233183146f, 0.004813634790480137f, 0.003695171559229493f, 0.003548339707776904f, 0.003436154453083873f, 0.004678633995354176f, 0.0048944237641990185f, 0.002722727367654443f, 0.004308423958718777f, 0.003495212644338608f, 0.0020885250996798277f, 0.0019438627641648054f, 0.004660833161324263f, 0.0012014276580885053f);
static const ai_layer_format_type conv2d_18_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_16_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_16_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_16_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_16_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_16_t_in_0_shape_ch_const_u16 = 5;
static const ai_u16 conv2d_16_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_16_t_in_0_fmt_zero_const_s8 = -79;
static const ai_i8 conv2d_16_t_out_0_fmt_zero_const_s8 = 2;
static const ai_float conv2d_16_t_in_0_fmt_scale_const_f32 = 0.020258275792002678f;
static const ai_float conv2d_16_t_out_0_fmt_scale_const_f32 = 0.0321701355278492f;
static const ai_float conv2d_16_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006660050246864557f, 0.004605495370924473f, 0.00738275982439518f, 0.003600077936425805f, 0.0029554502107203007f, 0.003008517436683178f, 0.0077493092976510525f, 0.0021202685311436653f, 0.003742368659004569f, 0.002203907584771514f, 0.0026333846617490053f, 0.0029802375938743353f, 0.00411535007879138f, 0.0026036857161670923f, 0.0013147011632099748f, 0.004444606602191925f, 0.0026098627131432295f, 0.0031724688597023487f, 0.004896614234894514f, 0.003916692454367876f, 0.0035823150537908077f, 0.002385025378316641f, 0.0031741983257234097f, 0.003518479410558939f, 0.0034303537104278803f, 0.006783959921449423f, 0.004409813787788153f, 0.0025312805082648993f, 0.004278861451894045f, 0.002474590204656124f, 0.003919366747140884f, 0.0021730847656726837f);
static const ai_layer_format_type conv2d_16_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_14_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_14_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_14_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_14_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_14_t_in_0_shape_ch_const_u16 = 5;
static const ai_u16 conv2d_14_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_14_t_in_0_fmt_zero_const_s8 = -79;
static const ai_i8 conv2d_14_t_out_0_fmt_zero_const_s8 = 6;
static const ai_float conv2d_14_t_in_0_fmt_scale_const_f32 = 0.020258275792002678f;
static const ai_float conv2d_14_t_out_0_fmt_scale_const_f32 = 0.024298058822751045f;
static const ai_float conv2d_14_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.005927613005042076f, 0.003493215423077345f, 0.001900283619761467f, 0.003143785521388054f, 0.0008614944526925683f, 0.0025473060086369514f, 0.0010586136486381292f, 0.0035522934049367905f, 0.00401476351544261f, 0.0027552489191293716f, 0.0010133050382137299f, 0.0019977635238319635f, 0.0020601472351700068f, 0.0032487742137163877f, 0.005083167459815741f, 0.0013801755849272013f, 0.001678477507084608f, 0.0025070032570511103f, 0.0017863217508420348f, 0.004478275775909424f, 0.002402057172730565f, 0.0018233711598441005f, 0.0030457056127488613f, 0.0027730418369174004f, 0.0025741923600435257f, 0.005492945201694965f, 0.00416283356025815f, 0.0013203711714595556f, 0.001366738579235971f, 0.0021592164412140846f, 0.002983906539157033f, 0.0016713707009330392f);
static const ai_layer_format_type conv2d_14_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_12_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_12_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_12_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_12_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_12_t_in_0_shape_ch_const_u16 = 5;
static const ai_u16 conv2d_12_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_12_t_in_0_fmt_zero_const_s8 = -79;
static const ai_i8 conv2d_12_t_out_0_fmt_zero_const_s8 = -31;
static const ai_float conv2d_12_t_in_0_fmt_scale_const_f32 = 0.020258275792002678f;
static const ai_float conv2d_12_t_out_0_fmt_scale_const_f32 = 0.021649004891514778f;
static const ai_float conv2d_12_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.005102890077978373f, 0.0040964693762362f, 0.0010533484164625406f, 0.0033382612746208906f, 0.0009580737678334117f, 0.0021556394640356302f, 0.0010644921567291021f, 0.001968336757272482f, 0.001319556264206767f, 0.001784818945452571f, 0.0028403729666024446f, 0.0011859505902975798f, 0.0024218985345214605f, 0.002352575771510601f, 0.002328373957425356f, 0.0023505864664912224f, 0.0018726337002590299f, 0.006023310124874115f, 0.0007829078240320086f, 0.00375269609503448f, 0.002379258628934622f, 0.0031164924148470163f, 0.002906551817432046f, 0.002372748451307416f, 0.003977190237492323f, 0.0015780043322592974f, 0.0022423991467803717f, 0.002584230387583375f, 0.002103450708091259f, 0.0027932128868997097f, 0.0029295191634446383f, 0.0006379126571118832f);
static const ai_layer_format_type conv2d_12_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_11_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_11_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_11_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_11_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_11_t_in_0_shape_ch_const_u16 = 5;
static const ai_u16 conv2d_11_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_11_t_in_0_fmt_zero_const_s8 = -79;
static const ai_i8 conv2d_11_t_out_0_fmt_zero_const_s8 = 11;
static const ai_float conv2d_11_t_in_0_fmt_scale_const_f32 = 0.020258275792002678f;
static const ai_float conv2d_11_t_out_0_fmt_scale_const_f32 = 0.029205355793237686f;
static const ai_float conv2d_11_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.004926035646349192f, 0.005315876100212336f, 0.0025135434698313475f, 0.002731999382376671f, 0.0012619126355275512f, 0.0023062662221491337f, 0.002547695767134428f, 0.0024948823265731335f, 0.0020294131245464087f, 0.004660170525312424f, 0.002111629117280245f, 0.0014980243286117911f, 0.003344189142808318f, 0.0018643038347363472f, 0.001377590699121356f, 0.002042644191533327f, 0.0026536190416663885f, 0.0032991846092045307f, 0.00343066593632102f, 0.004301752429455519f, 0.006407108623534441f, 0.0007841636543162167f, 0.005040908697992563f, 0.0036514336243271828f, 0.0034472192637622356f, 0.0014617107808589935f, 0.0027026180177927017f, 0.0026297520380467176f, 0.004133943002671003f, 0.002599452855065465f, 0.003152098273858428f, 0.0012924964539706707f);
static const ai_layer_format_type conv2d_11_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;





static const ai_i8 pad_21_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_21_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_21_t_in_0_shape_h_const_u32 = 36;


static const ai_u16 conv2d_39_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_39_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_39_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_39_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_39_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_39_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_39_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_39_t_out_0_fmt_zero_const_s8 = 69;
static const ai_float conv2d_39_t_in_0_fmt_scale_const_f32 = 0.018089503049850464f;
static const ai_float conv2d_39_t_out_0_fmt_scale_const_f32 = 0.10019893199205399f;
static const ai_float conv2d_39_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.01186366006731987f, 0.010793269611895084f, 0.006670406553894281f, 0.01010843738913536f, 0.008023221977055073f, 0.007320745382457972f, 0.006964652333408594f, 0.010917450301349163f, 0.009318972006440163f, 0.010426013730466366f, 0.008870696648955345f, 0.007436016574501991f, 0.013014719821512699f, 0.010586484335362911f, 0.010143178515136242f, 0.005291686858981848f, 0.014731369912624359f, 0.008135355077683926f, 0.016908466815948486f, 0.018148398026823997f, 0.009234112687408924f, 0.027417320758104324f, 0.020561857149004936f, 0.009394153952598572f, 0.01168648898601532f, 0.022061137482523918f, 0.007465533446520567f, 0.007668893784284592f, 0.002580994041636586f, 0.012257195077836514f, 0.006083840038627386f, 0.0022906423546373844f);
static const ai_layer_format_type conv2d_39_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_37_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_37_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_37_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_37_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_37_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_37_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_37_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_37_t_out_0_fmt_zero_const_s8 = 70;
static const ai_float conv2d_37_t_in_0_fmt_scale_const_f32 = 0.018089503049850464f;
static const ai_float conv2d_37_t_out_0_fmt_scale_const_f32 = 0.050918929278850555f;
static const ai_float conv2d_37_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.013744818046689034f, 0.008028565905988216f, 0.006493261083960533f, 0.006331305485218763f, 0.007188533898442984f, 0.005197266582399607f, 0.008426150307059288f, 0.00432893680408597f, 0.007703164126724005f, 0.016972679644823074f, 0.009170184843242168f, 0.00976622011512518f, 0.006966914515942335f, 0.009265800006687641f, 0.005857688374817371f, 0.007292239926755428f, 0.011303541250526905f, 0.00837360043078661f, 0.01344231516122818f, 0.016146158799529076f, 0.0056494614109396935f, 0.008102784864604473f, 0.008666194044053555f, 0.01717696152627468f, 0.005572604015469551f, 0.008746170438826084f, 0.008867766708135605f, 0.006272086873650551f, 0.0017736664740368724f, 0.007151161320507526f, 0.008280628360807896f, 0.0015141080366447568f);
static const ai_layer_format_type conv2d_37_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_35_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_35_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_35_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_35_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_35_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_35_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_35_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_35_t_out_0_fmt_zero_const_s8 = 75;
static const ai_float conv2d_35_t_in_0_fmt_scale_const_f32 = 0.018089503049850464f;
static const ai_float conv2d_35_t_out_0_fmt_scale_const_f32 = 0.06250172108411789f;
static const ai_float conv2d_35_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.005320250988006592f, 0.011432788334786892f, 0.003821037709712982f, 0.007447374518960714f, 0.005486380774527788f, 0.0035415603779256344f, 0.0038267401978373528f, 0.00546449376270175f, 0.007639617659151554f, 0.005965702701359987f, 0.005623030010610819f, 0.004703879356384277f, 0.01130787841975689f, 0.007022269535809755f, 0.004374395124614239f, 0.004662839695811272f, 0.004131923895329237f, 0.005576536525040865f, 0.009175760671496391f, 0.005430291872471571f, 0.007322792895138264f, 0.0038090189918875694f, 0.0206536203622818f, 0.019630953669548035f, 0.008100414648652077f, 0.0028762840665876865f, 0.007377562578767538f, 0.008830556645989418f, 0.002273592399433255f, 0.003522289451211691f, 0.0033785635605454445f, 0.001805984415113926f);
static const ai_layer_format_type conv2d_35_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_33_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_33_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_33_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_33_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_33_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_33_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_33_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_33_t_out_0_fmt_zero_const_s8 = 84;
static const ai_float conv2d_33_t_in_0_fmt_scale_const_f32 = 0.018089503049850464f;
static const ai_float conv2d_33_t_out_0_fmt_scale_const_f32 = 0.08613935112953186f;
static const ai_float conv2d_33_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0076363119296729565f, 0.0075243874453008175f, 0.005133591126650572f, 0.009452147409319878f, 0.010225324891507626f, 0.00278281862847507f, 0.003424954367801547f, 0.003879107302054763f, 0.009703134186565876f, 0.008395685814321041f, 0.0043561868369579315f, 0.005732813384383917f, 0.01714053936302662f, 0.009543717838823795f, 0.003499682992696762f, 0.007616529241204262f, 0.005455814767628908f, 0.003369970479980111f, 0.006169282831251621f, 0.0028199150692671537f, 0.006774163339287043f, 0.002500639297068119f, 0.023115919902920723f, 0.015388164669275284f, 0.0075706192292273045f, 0.005381882190704346f, 0.008265995420515537f, 0.006545139942318201f, 0.0018426255555823445f, 0.007463878486305475f, 0.005449183285236359f, 0.0014911057660356164f);
static const ai_layer_format_type conv2d_33_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_32_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_32_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_32_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_32_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_32_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_32_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_32_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_32_t_out_0_fmt_zero_const_s8 = 88;
static const ai_float conv2d_32_t_in_0_fmt_scale_const_f32 = 0.018089503049850464f;
static const ai_float conv2d_32_t_out_0_fmt_scale_const_f32 = 0.11588045209646225f;
static const ai_float conv2d_32_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.005847741384059191f, 0.004816038534045219f, 0.004055524244904518f, 0.012894891202449799f, 0.008040587417781353f, 0.004676547832787037f, 0.0026376936584711075f, 0.0018458456033840775f, 0.007230961695313454f, 0.006679101847112179f, 0.005549816880375147f, 0.004266202449798584f, 0.007537039928138256f, 0.007947630248963833f, 0.004141840618103743f, 0.006199989467859268f, 0.007511948235332966f, 0.0038623949512839317f, 0.005441158544272184f, 0.003257722593843937f, 0.004524814896285534f, 0.005875532515347004f, 0.008896449580788612f, 0.01908513531088829f, 0.00802655704319477f, 0.0030347262509167194f, 0.00668342737480998f, 0.007985357195138931f, 0.0012702939566224813f, 0.004883514251559973f, 0.010348538868129253f, 0.0015016513643786311f);
static const ai_layer_format_type conv2d_32_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;





static const ai_i8 pad_42_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_42_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_42_t_in_0_shape_h_const_u32 = 36;


static const ai_u16 conv2d_60_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_60_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_60_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_60_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_60_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_60_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_60_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_60_t_out_0_fmt_zero_const_s8 = 64;
static const ai_float conv2d_60_t_in_0_fmt_scale_const_f32 = 0.02584196999669075f;
static const ai_float conv2d_60_t_out_0_fmt_scale_const_f32 = 0.14971131086349487f;
static const ai_float conv2d_60_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.023584440350532532f, 0.006886603776365519f, 0.008270791731774807f, 0.00705513171851635f, 0.004895553458482027f, 0.006690950598567724f, 0.009576729498803616f, 0.007938016206026077f, 0.007553701754659414f, 0.008660083636641502f, 0.0032553072087466717f, 0.007968931458890438f, 0.006374853663146496f, 0.006226337514817715f, 0.007620074786245823f, 0.005005958024412394f, 0.003538780612871051f, 0.006630877964198589f, 0.005422016140073538f, 0.005794634111225605f, 0.006335299927741289f, 0.010995433665812016f, 0.00768909091129899f, 0.014380763284862041f, 0.00708600040525198f, 0.006676636170595884f, 0.006559495348483324f, 0.0058885011821985245f, 0.001449241884984076f, 0.009769883938133717f, 0.006820480339229107f, 0.005793518386781216f, 0.0076612127013504505f, 0.0034535566810518503f, 0.0009176076855510473f, 0.007151528261601925f, 0.004415534436702728f, 0.006010981742292643f, 0.026794468984007835f, 0.007233982905745506f, 0.012274456210434437f, 0.011477350257337093f, 0.003788759233430028f, 0.007211590185761452f, 0.008895937353372574f, 0.0012653636513277888f, 0.016379518434405327f, 0.009089585393667221f, 0.008973220363259315f, 0.004353004042059183f, 0.008774669840931892f, 0.001782048144377768f, 0.0036011445336043835f, 0.007290054112672806f, 0.013679294846951962f, 0.02327742613852024f, 0.008089921437203884f, 0.009468146599829197f, 0.0067704813554883f, 0.011636949144303799f, 0.007956483401358128f, 0.009739111177623272f, 0.007585447747260332f, 0.017096368595957756f);
static const ai_layer_format_type conv2d_60_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_58_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_58_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_58_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_58_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_58_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_58_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_58_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_58_t_out_0_fmt_zero_const_s8 = 87;
static const ai_float conv2d_58_t_in_0_fmt_scale_const_f32 = 0.02584196999669075f;
static const ai_float conv2d_58_t_out_0_fmt_scale_const_f32 = 0.09137392789125443f;
static const ai_float conv2d_58_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.02746071293950081f, 0.004339249804615974f, 0.006911101751029491f, 0.007261958904564381f, 0.003194740740582347f, 0.008315697312355042f, 0.008656446821987629f, 0.009697999805212021f, 0.008740326389670372f, 0.0052126492373645306f, 0.004511886741966009f, 0.005854828283190727f, 0.007907298393547535f, 0.007916250266134739f, 0.006602541543543339f, 0.0068309553898870945f, 0.004342629574239254f, 0.008743495680391788f, 0.0070934719406068325f, 0.004663942847400904f, 0.007688775192946196f, 0.006980229634791613f, 0.0057592084631323814f, 0.005942918825894594f, 0.013970348052680492f, 0.009928067214787006f, 0.005922124721109867f, 0.007145135197788477f, 0.0011074870126321912f, 0.006631216499954462f, 0.010011589154601097f, 0.00626453896984458f, 0.008851515129208565f, 0.011123419739305973f, 0.0008451113826595247f, 0.006818766240030527f, 0.004738659597933292f, 0.004730341024696827f, 0.015139014460146427f, 0.005431167781352997f, 0.004570809658616781f, 0.01370510645210743f, 0.006119093392044306f, 0.004823659546673298f, 0.0038075288757681847f, 0.0012186338426545262f, 0.005817755125463009f, 0.008930884301662445f, 0.011319713667035103f, 0.0060677737928926945f, 0.006932509131729603f, 0.001180289313197136f, 0.00706797419115901f, 0.0076816813088953495f, 0.00681843189522624f, 0.007324418518692255f, 0.004747698083519936f, 0.005492985714226961f, 0.006449977867305279f, 0.009632387198507786f, 0.004095830488950014f, 0.005588484462350607f, 0.009799765422940254f, 0.008277172222733498f);
static const ai_layer_format_type conv2d_58_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_56_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_56_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_56_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_56_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_56_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_56_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_56_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_56_t_out_0_fmt_zero_const_s8 = 59;
static const ai_float conv2d_56_t_in_0_fmt_scale_const_f32 = 0.02584196999669075f;
static const ai_float conv2d_56_t_out_0_fmt_scale_const_f32 = 0.06863502413034439f;
static const ai_float conv2d_56_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.023002680391073227f, 0.005029221065342426f, 0.004030009265989065f, 0.004085481632500887f, 0.004546402953565121f, 0.008098047226667404f, 0.0035646844189614058f, 0.007505639456212521f, 0.006713546812534332f, 0.0051346165128052235f, 0.0035199522972106934f, 0.006578161381185055f, 0.0058084893971681595f, 0.006661576684564352f, 0.00601462135091424f, 0.004799210000783205f, 0.0068666101433336735f, 0.009456089697778225f, 0.006522048264741898f, 0.004453027155250311f, 0.006534283980727196f, 0.007639946881681681f, 0.0063560232520103455f, 0.004699409939348698f, 0.006167041603475809f, 0.013286417350172997f, 0.00483568012714386f, 0.005451759323477745f, 0.0011462188558652997f, 0.006943758111447096f, 0.008214625529944897f, 0.0044809430837631226f, 0.006800337228924036f, 0.009566795080900192f, 0.0009066879283636808f, 0.006248443387448788f, 0.006236399058252573f, 0.004999152850359678f, 0.0045755840837955475f, 0.005631828214973211f, 0.004048632923513651f, 0.006936526857316494f, 0.003888112958520651f, 0.004538241773843765f, 0.0031530384439975023f, 0.0013917823089286685f, 0.005482861306518316f, 0.008448339067399502f, 0.010582523420453072f, 0.004623652435839176f, 0.005197619553655386f, 0.001090841949917376f, 0.0066250902600586414f, 0.004086183849722147f, 0.004710646811872721f, 0.006261983886361122f, 0.006833507679402828f, 0.004849355202168226f, 0.010333321988582611f, 0.009453464299440384f, 0.00847382191568613f, 0.005470174830406904f, 0.00662390748038888f, 0.008638101629912853f);
static const ai_layer_format_type conv2d_56_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_54_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_54_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_54_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_54_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_54_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_54_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_54_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_54_t_out_0_fmt_zero_const_s8 = 37;
static const ai_float conv2d_54_t_in_0_fmt_scale_const_f32 = 0.02584196999669075f;
static const ai_float conv2d_54_t_out_0_fmt_scale_const_f32 = 0.05569436401128769f;
static const ai_float conv2d_54_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.017672428861260414f, 0.004723104648292065f, 0.0049975039437413216f, 0.005132142920047045f, 0.002798955887556076f, 0.006447845138609409f, 0.00458114268258214f, 0.013289084658026695f, 0.009903796948492527f, 0.006482062861323357f, 0.004005677066743374f, 0.004685596562922001f, 0.003935187589377165f, 0.0033079825807362795f, 0.004812706261873245f, 0.0031296471133828163f, 0.007239422760903835f, 0.006813755724579096f, 0.0053282231092453f, 0.004294975660741329f, 0.005049896892160177f, 0.004144858103245497f, 0.005669260397553444f, 0.006490830332040787f, 0.009998581372201443f, 0.010984965600073338f, 0.008018361404538155f, 0.005677453242242336f, 0.0016100250650197268f, 0.00620471453294158f, 0.008171594701707363f, 0.0033403108827769756f, 0.005374768748879433f, 0.00811159610748291f, 0.0007645302684977651f, 0.005495288409292698f, 0.007964408956468105f, 0.006490687839686871f, 0.005460395477712154f, 0.005983057431876659f, 0.0033963194582611322f, 0.005815999116748571f, 0.004516097251325846f, 0.003269375301897526f, 0.004960900172591209f, 0.001467236434109509f, 0.0030966312624514103f, 0.005095120053738356f, 0.004596778191626072f, 0.00406482582911849f, 0.007612462155520916f, 0.0013341655721887946f, 0.008162292651832104f, 0.005327939987182617f, 0.0072225783951580524f, 0.0058876583352684975f, 0.0060422890819609165f, 0.003451191121712327f, 0.005335141904652119f, 0.010021377354860306f, 0.006020971108227968f, 0.006616218946874142f, 0.005577729549258947f, 0.006002016365528107f);
static const ai_layer_format_type conv2d_54_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_53_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_53_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_53_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_53_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_53_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_53_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_53_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_53_t_out_0_fmt_zero_const_s8 = 47;
static const ai_float conv2d_53_t_in_0_fmt_scale_const_f32 = 0.02584196999669075f;
static const ai_float conv2d_53_t_out_0_fmt_scale_const_f32 = 0.09451217204332352f;
static const ai_float conv2d_53_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.010691424831748009f, 0.004178167320787907f, 0.008455551229417324f, 0.005545607302337885f, 0.003232670249417424f, 0.005447716917842627f, 0.007182997185736895f, 0.01076432690024376f, 0.008422354236245155f, 0.008233168162405491f, 0.0048151384107768536f, 0.005208331160247326f, 0.004927508998662233f, 0.0036423359997570515f, 0.005581345409154892f, 0.003340465947985649f, 0.011325751431286335f, 0.009525831788778305f, 0.0042296820320189f, 0.003503932151943445f, 0.00681167421862483f, 0.004984577186405659f, 0.0060119833797216415f, 0.008277871645987034f, 0.005857178475707769f, 0.0074858227744698524f, 0.011018595658242702f, 0.00552021712064743f, 0.0015811858465895057f, 0.007746486458927393f, 0.009669800288975239f, 0.004823782481253147f, 0.005517985671758652f, 0.005578388459980488f, 0.0007292140508070588f, 0.005281788296997547f, 0.01435765065252781f, 0.004840027075260878f, 0.0033415008801966906f, 0.007375776767730713f, 0.0053479173220694065f, 0.004389075096696615f, 0.0029694659169763327f, 0.003546219551935792f, 0.003847916144877672f, 0.0012382416753098369f, 0.004885523580014706f, 0.00553384143859148f, 0.005742278415709734f, 0.004013983532786369f, 0.005381616298109293f, 0.0013083856320008636f, 0.008187194354832172f, 0.004528744146227837f, 0.0063695949502289295f, 0.005524647422134876f, 0.005343312863260508f, 0.00299745611846447f, 0.0059463875368237495f, 0.0054924460127949715f, 0.006149664521217346f, 0.005005555227398872f, 0.00515018729493022f, 0.007459322456270456f);
static const ai_layer_format_type conv2d_53_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;





static const ai_i8 pad_63_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_63_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_63_t_in_0_shape_h_const_u32 = 36;


static const ai_u16 conv2d_81_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_81_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_81_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_81_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_81_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_81_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_81_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_81_t_out_0_fmt_zero_const_s8 = 39;
static const ai_float conv2d_81_t_in_0_fmt_scale_const_f32 = 0.05029941350221634f;
static const ai_float conv2d_81_t_out_0_fmt_scale_const_f32 = 0.19084693491458893f;
static const ai_float conv2d_81_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0012504096375778317f, 0.0058128731325268745f, 0.0005321576609276235f, 0.006066199392080307f, 0.002211417770013213f, 0.012687157839536667f, 0.00831177830696106f, 0.008697552606463432f, 0.007882985286414623f, 0.005129409488290548f, 0.012306316755712032f, 0.0007997534121386707f, 0.00838465429842472f, 0.0021518897265195847f, 0.010588216595351696f, 0.00822302047163248f, 0.008896395564079285f, 0.017358284443616867f, 0.0010456695454195142f, 0.001594871049746871f, 0.0009155870066024363f, 0.0008863717666827142f, 0.002291066339239478f, 0.0006412363145500422f, 0.0032819644547998905f, 0.000994651229120791f, 0.0033874588552862406f, 0.0065874019637703896f, 0.00983365811407566f, 0.0007120500667952001f, 0.0007958783535286784f, 0.0008954235236160457f, 0.0007632373599335551f, 0.007006964646279812f, 0.008395188488066196f, 0.004700847901403904f, 0.0012709879083558917f, 0.002491388004273176f, 0.01272381842136383f, 0.010231897234916687f, 0.0007170800818130374f, 0.0009433773229829967f, 0.0006816678796894848f, 0.006645608227699995f, 0.0007810474489815533f, 0.0008001021342352033f, 0.0007697467808611691f, 0.0008388052228838205f, 0.0058981068432331085f, 0.0007623223937116563f, 0.006337590515613556f, 0.0015766691649332643f, 0.014839247800409794f, 0.010429266840219498f, 0.0008884346461854875f, 0.0019630836322903633f, 0.0006346222362481058f, 0.002328675240278244f, 0.021993812173604965f, 0.004037224687635899f, 0.003481464460492134f, 0.0048868232406675816f, 0.010001733899116516f, 0.0019100853241980076f);
static const ai_layer_format_type conv2d_81_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_79_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_79_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_79_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_79_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_79_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_79_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_79_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_79_t_out_0_fmt_zero_const_s8 = 75;
static const ai_float conv2d_79_t_in_0_fmt_scale_const_f32 = 0.05029941350221634f;
static const ai_float conv2d_79_t_out_0_fmt_scale_const_f32 = 0.1060132309794426f;
static const ai_float conv2d_79_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0011369066778570414f, 0.006388429086655378f, 0.0006817360990680754f, 0.0043937694281339645f, 0.0020104243885725737f, 0.007931353524327278f, 0.00613761693239212f, 0.007508817594498396f, 0.006351636257022619f, 0.0029387441463768482f, 0.007670631632208824f, 0.0009141612681560218f, 0.003866154933348298f, 0.002055913442745805f, 0.009391882456839085f, 0.004172370303422213f, 0.009123904630541801f, 0.011142580769956112f, 0.0008761856006458402f, 0.0014108262257650495f, 0.0007166761206462979f, 0.000894423108547926f, 0.0013129530707374215f, 0.0006172274006530643f, 0.002434641122817993f, 0.0006424699677154422f, 0.003631552215665579f, 0.0044181328266859055f, 0.006723457481712103f, 0.0007242396241053939f, 0.0007499320199713111f, 0.000866775109898299f, 0.0009232872980646789f, 0.007530079688876867f, 0.0036646227817982435f, 0.002222833689302206f, 0.0009589515975676477f, 0.0022220052778720856f, 0.009600742720067501f, 0.0032005351968109608f, 0.0008003478869795799f, 0.0008815184119157493f, 0.0006277975044213235f, 0.006066462025046349f, 0.0006501827156171203f, 0.0008573415107093751f, 0.0006148361135274172f, 0.0007616743678227067f, 0.0050813849084079266f, 0.0008566355681978166f, 0.0037624777760356665f, 0.0009084135526791215f, 0.005612869281321764f, 0.005254549439996481f, 0.0009018589043989778f, 0.002029994735494256f, 0.0006767283193767071f, 0.002304414752870798f, 0.011761273257434368f, 0.006982847582548857f, 0.0036935315001755953f, 0.0038832665886729956f, 0.004448584280908108f, 0.0017060021637007594f);
static const ai_layer_format_type conv2d_79_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_77_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_77_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_77_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_77_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_77_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_77_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_77_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_77_t_out_0_fmt_zero_const_s8 = 91;
static const ai_float conv2d_77_t_in_0_fmt_scale_const_f32 = 0.05029941350221634f;
static const ai_float conv2d_77_t_out_0_fmt_scale_const_f32 = 0.12077406048774719f;
static const ai_float conv2d_77_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0015991796972230077f, 0.003648982150480151f, 0.0005973725928924978f, 0.007670808583498001f, 0.0029967413283884525f, 0.00899110920727253f, 0.006386891473084688f, 0.007067433558404446f, 0.005079204682260752f, 0.003353073727339506f, 0.005907014012336731f, 0.0008295061998069286f, 0.003963916562497616f, 0.0017958703683689237f, 0.005228205118328333f, 0.005702183581888676f, 0.0043810331262648106f, 0.009389163926243782f, 0.0009309326997026801f, 0.0018360563553869724f, 0.0008080149418674409f, 0.0009177916799671948f, 0.002024797024205327f, 0.0006234447937458754f, 0.0034163992386311293f, 0.0009314632625319064f, 0.003638125490397215f, 0.004408008884638548f, 0.004853380843997002f, 0.0008462241385132074f, 0.0009697361383587122f, 0.0008413238101638854f, 0.0011189188808202744f, 0.007574292831122875f, 0.003886349266394973f, 0.003555209143087268f, 0.0008436007192358375f, 0.0033351124729961157f, 0.0074333567172288895f, 0.004662944003939629f, 0.0008524784934706986f, 0.0010157286887988448f, 0.0007387571968138218f, 0.00509015005081892f, 0.0008047522860579193f, 0.0008811010047793388f, 0.0006219606148079038f, 0.0008307943935506046f, 0.0029514466878026724f, 0.0007096734480001032f, 0.004397837445139885f, 0.0010115604382008314f, 0.004817766137421131f, 0.007485097739845514f, 0.0010851307306438684f, 0.0015882692532613873f, 0.0005800423095934093f, 0.00237546325661242f, 0.011521818116307259f, 0.0034770050551742315f, 0.0037970710545778275f, 0.004632922820746899f, 0.007609159220010042f, 0.0014549007173627615f);
static const ai_layer_format_type conv2d_77_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_75_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_75_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_75_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_75_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_75_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_75_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_75_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_75_t_out_0_fmt_zero_const_s8 = 79;
static const ai_float conv2d_75_t_in_0_fmt_scale_const_f32 = 0.05029941350221634f;
static const ai_float conv2d_75_t_out_0_fmt_scale_const_f32 = 0.14425751566886902f;
static const ai_float conv2d_75_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0018866024911403656f, 0.006260234862565994f, 0.0006819243426434696f, 0.0049243164248764515f, 0.0024129776284098625f, 0.005625334568321705f, 0.00699264369904995f, 0.006548008881509304f, 0.0045412261970341206f, 0.0033035993110388517f, 0.013871962204575539f, 0.0007637946982868016f, 0.004585233051329851f, 0.001986807445064187f, 0.005710932891815901f, 0.00571082066744566f, 0.008636881597340107f, 0.010422461666166782f, 0.0009270839509554207f, 0.002219997113570571f, 0.0008809485007077456f, 0.0006569916731677949f, 0.0017940683756023645f, 0.0006408605840988457f, 0.003170772222802043f, 0.0006882665911689401f, 0.0037388738710433245f, 0.007284902036190033f, 0.00428159860894084f, 0.0007137377979233861f, 0.0008148741326294839f, 0.0008031419129110873f, 0.0011762181529775262f, 0.005005739163607359f, 0.002395868534222245f, 0.0027034685481339693f, 0.0010254139779135585f, 0.0019457237794995308f, 0.008279512636363506f, 0.0035169320181012154f, 0.0008977118995971978f, 0.0008354591554962099f, 0.0006251359009183943f, 0.006170975975692272f, 0.0006973627023398876f, 0.0007016129675321281f, 0.0007304160390049219f, 0.0006747248698957264f, 0.0038879821076989174f, 0.0007244542357511818f, 0.004851704929023981f, 0.0010094295721501112f, 0.0047797542065382f, 0.008030660450458527f, 0.0011897607473656535f, 0.0019119539065286517f, 0.000617648009210825f, 0.0026271380484104156f, 0.013531569391489029f, 0.0031306834425777197f, 0.0031448781955987215f, 0.005479766055941582f, 0.005528809502720833f, 0.002981108846142888f);
static const ai_layer_format_type conv2d_75_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_74_t_in_0_shape_w_const_u16 = 36;
static const ai_u16 conv2d_74_t_in_0_shape_h_const_u16 = 1;
static const ai_u16 conv2d_74_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_74_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_74_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 conv2d_74_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_74_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_74_t_out_0_fmt_zero_const_s8 = 89;
static const ai_float conv2d_74_t_in_0_fmt_scale_const_f32 = 0.05029941350221634f;
static const ai_float conv2d_74_t_out_0_fmt_scale_const_f32 = 0.1846686601638794f;
static const ai_float conv2d_74_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0013931408757343888f, 0.006662474479526281f, 0.0007881636847741902f, 0.010331179015338421f, 0.003420088440179825f, 0.009219982661306858f, 0.006146042607724667f, 0.01196266245096922f, 0.003734895493835211f, 0.0026992023922502995f, 0.007749039679765701f, 0.0007130408193916082f, 0.006753893569111824f, 0.0019200221868231893f, 0.0047716121189296246f, 0.006257767789065838f, 0.006252717226743698f, 0.014232484623789787f, 0.0009410087368451059f, 0.002974928356707096f, 0.0007355890120379627f, 0.0008102463325485587f, 0.001445319619961083f, 0.0006297047366388142f, 0.00212052627466619f, 0.0008203036850318313f, 0.004464522935450077f, 0.0057363747619092464f, 0.006797709967941046f, 0.000717024493496865f, 0.0008290567202493548f, 0.0010280328569933772f, 0.0012070037191733718f, 0.00888077262789011f, 0.0039038825780153275f, 0.004223662428557873f, 0.0010585018899291754f, 0.002505130833014846f, 0.010033001191914082f, 0.008167438209056854f, 0.0009480807930231094f, 0.0008088512113317847f, 0.0008114364463835955f, 0.007945789024233818f, 0.0007805038476362824f, 0.00065928342519328f, 0.0007495454046875238f, 0.0005940491682849824f, 0.008449283428490162f, 0.0008786260150372982f, 0.00596630247309804f, 0.0012039551511406898f, 0.005033345427364111f, 0.008946924470365047f, 0.0011501738335937262f, 0.0015870224451646209f, 0.0006355878431349993f, 0.0025856802240014076f, 0.013647296465933323f, 0.005871884990483522f, 0.004279629327356815f, 0.009461679495871067f, 0.006716578733175993f, 0.001774681149981916f);
static const ai_layer_format_type conv2d_74_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;






STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN pad_0 */
  {
      const ai_ptr pad_0_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_inputs[0] + 0);
    ai_ptr pad_0_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 952);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(0, 1, {(stai_ptr) pad_0_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_0_t_in_0_ptr_const_ptr, pad_0_t_out_0_ptr_ptr, (ai_handle)(pad_0_v_pad_constant_value_const_s8), pad_0_t_in_0_fmt_bitsize_const_s16, pad_0_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(5), (ai_i32)(20), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(0, 1, {(stai_ptr) pad_0_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_0 */
  /* LITE_KERNEL_SECTION BEGIN slice_9 */
  {
    
  forward_lite_slice_slice_9(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_9 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_18 */
  {
      const ai_i8* conv2d_18_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1152);
    const ai_i8* conv2d_18_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4);
    const ai_i32* conv2d_18_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 164);
    ai_i8* conv2d_18_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 1672);
    ai_i16* conv2d_18_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1332);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(18, 1, {(stai_ptr) conv2d_18_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_18_t_in_0_ptr_const_s8, conv2d_18_t_in_0_shape_w_const_u16, conv2d_18_t_in_0_shape_h_const_u16, conv2d_18_l_stride_1_const_u16, conv2d_18_l_stride_0_const_u16, conv2d_18_t_in_0_shape_ch_const_u16, conv2d_18_t_weight_0_ptr_const_s8, conv2d_18_t_out_0_shape_ch_const_u16, conv2d_18_t_weight_1_ptr_const_s32, conv2d_18_t_in_0_fmt_zero_const_s8, conv2d_18_t_out_0_fmt_zero_const_s8, conv2d_18_t_in_0_fmt_scale_const_f32, conv2d_18_t_out_0_fmt_scale_const_f32, conv2d_18_t_weight_0_fmt_scale_const_f32, conv2d_18_l_out_ch_format_const_layer_format_type, conv2d_18_t_out_0_ptr_s8, 1, 340, conv2d_18_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(18, 1, {(stai_ptr) conv2d_18_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_18 */
  /* LITE_KERNEL_SECTION BEGIN slice_7 */
  {
    
  forward_lite_slice_slice_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_7 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_16 */
  {
      const ai_i8* conv2d_16_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1152);
    const ai_i8* conv2d_16_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 292);
    const ai_i32* conv2d_16_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 452);
    ai_i8* conv2d_16_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 2824);
    ai_i16* conv2d_16_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1332);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, {(stai_ptr) conv2d_16_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_16_t_in_0_ptr_const_s8, conv2d_16_t_in_0_shape_w_const_u16, conv2d_16_t_in_0_shape_h_const_u16, conv2d_16_l_stride_1_const_u16, conv2d_16_l_stride_0_const_u16, conv2d_16_t_in_0_shape_ch_const_u16, conv2d_16_t_weight_0_ptr_const_s8, conv2d_16_t_out_0_shape_ch_const_u16, conv2d_16_t_weight_1_ptr_const_s32, conv2d_16_t_in_0_fmt_zero_const_s8, conv2d_16_t_out_0_fmt_zero_const_s8, conv2d_16_t_in_0_fmt_scale_const_f32, conv2d_16_t_out_0_fmt_scale_const_f32, conv2d_16_t_weight_0_fmt_scale_const_f32, conv2d_16_l_out_ch_format_const_layer_format_type, conv2d_16_t_out_0_ptr_s8, 1, 340, conv2d_16_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, {(stai_ptr) conv2d_16_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_16 */
  /* LITE_KERNEL_SECTION BEGIN slice_5 */
  {
    
  forward_lite_slice_slice_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_5 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_14 */
  {
      const ai_i8* conv2d_14_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1152);
    const ai_i8* conv2d_14_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 580);
    const ai_i32* conv2d_14_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 740);
    ai_i8* conv2d_14_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3976);
    ai_i16* conv2d_14_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1332);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(14, 1, {(stai_ptr) conv2d_14_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_14_t_in_0_ptr_const_s8, conv2d_14_t_in_0_shape_w_const_u16, conv2d_14_t_in_0_shape_h_const_u16, conv2d_14_l_stride_1_const_u16, conv2d_14_l_stride_0_const_u16, conv2d_14_t_in_0_shape_ch_const_u16, conv2d_14_t_weight_0_ptr_const_s8, conv2d_14_t_out_0_shape_ch_const_u16, conv2d_14_t_weight_1_ptr_const_s32, conv2d_14_t_in_0_fmt_zero_const_s8, conv2d_14_t_out_0_fmt_zero_const_s8, conv2d_14_t_in_0_fmt_scale_const_f32, conv2d_14_t_out_0_fmt_scale_const_f32, conv2d_14_t_weight_0_fmt_scale_const_f32, conv2d_14_l_out_ch_format_const_layer_format_type, conv2d_14_t_out_0_ptr_s8, 1, 340, conv2d_14_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(14, 1, {(stai_ptr) conv2d_14_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_14 */
  /* LITE_KERNEL_SECTION BEGIN slice_3 */
  {
    
  forward_lite_slice_slice_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_3 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_12 */
  {
      const ai_i8* conv2d_12_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1152);
    const ai_i8* conv2d_12_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 868);
    const ai_i32* conv2d_12_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1028);
    ai_i8* conv2d_12_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5128);
    ai_i16* conv2d_12_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1332);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(12, 1, {(stai_ptr) conv2d_12_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_12_t_in_0_ptr_const_s8, conv2d_12_t_in_0_shape_w_const_u16, conv2d_12_t_in_0_shape_h_const_u16, conv2d_12_l_stride_1_const_u16, conv2d_12_l_stride_0_const_u16, conv2d_12_t_in_0_shape_ch_const_u16, conv2d_12_t_weight_0_ptr_const_s8, conv2d_12_t_out_0_shape_ch_const_u16, conv2d_12_t_weight_1_ptr_const_s32, conv2d_12_t_in_0_fmt_zero_const_s8, conv2d_12_t_out_0_fmt_zero_const_s8, conv2d_12_t_in_0_fmt_scale_const_f32, conv2d_12_t_out_0_fmt_scale_const_f32, conv2d_12_t_weight_0_fmt_scale_const_f32, conv2d_12_l_out_ch_format_const_layer_format_type, conv2d_12_t_out_0_ptr_s8, 1, 340, conv2d_12_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(12, 1, {(stai_ptr) conv2d_12_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_12 */
  /* LITE_KERNEL_SECTION BEGIN slice_1 */
  {
    
  forward_lite_slice_slice_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_1 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_11 */
  {
      const ai_i8* conv2d_11_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1152);
    const ai_i8* conv2d_11_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1156);
    const ai_i32* conv2d_11_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1316);
    ai_i8* conv2d_11_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* conv2d_11_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1332);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(11, 1, {(stai_ptr) conv2d_11_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_11_t_in_0_ptr_const_s8, conv2d_11_t_in_0_shape_w_const_u16, conv2d_11_t_in_0_shape_h_const_u16, conv2d_11_l_stride_1_const_u16, conv2d_11_l_stride_0_const_u16, conv2d_11_t_in_0_shape_ch_const_u16, conv2d_11_t_weight_0_ptr_const_s8, conv2d_11_t_out_0_shape_ch_const_u16, conv2d_11_t_weight_1_ptr_const_s32, conv2d_11_t_in_0_fmt_zero_const_s8, conv2d_11_t_out_0_fmt_zero_const_s8, conv2d_11_t_in_0_fmt_scale_const_f32, conv2d_11_t_out_0_fmt_scale_const_f32, conv2d_11_t_weight_0_fmt_scale_const_f32, conv2d_11_l_out_ch_format_const_layer_format_type, conv2d_11_t_out_0_ptr_s8, 1, 340, conv2d_11_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(11, 1, {(stai_ptr) conv2d_11_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_11 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_13 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_13(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_13 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_15 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_15(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_15 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_17 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_17(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_17 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_19 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_19(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_19 */
  /* LITE_KERNEL_SECTION BEGIN pad_21 */
  {
      const ai_ptr pad_21_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 2824);
    ai_ptr pad_21_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(21, 1, {(stai_ptr) pad_21_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_21_t_in_0_ptr_const_ptr, pad_21_t_out_0_ptr_ptr, (ai_handle)(pad_21_v_pad_constant_value_const_s8), pad_21_t_in_0_fmt_bitsize_const_s16, pad_21_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(32), (ai_i32)(256), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(21, 1, {(stai_ptr) pad_21_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_21 */
  /* LITE_KERNEL_SECTION BEGIN slice_30 */
  {
    
  forward_lite_slice_slice_30(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_30 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_39 */
  {
      const ai_i8* conv2d_39_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1408);
    const ai_i8* conv2d_39_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1444);
    const ai_i32* conv2d_39_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2468);
    ai_i8* conv2d_39_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3008);
    ai_i16* conv2d_39_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 2560);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(39, 1, {(stai_ptr) conv2d_39_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_39_t_in_0_ptr_const_s8, conv2d_39_t_in_0_shape_w_const_u16, conv2d_39_t_in_0_shape_h_const_u16, conv2d_39_l_stride_1_const_u16, conv2d_39_l_stride_0_const_u16, conv2d_39_t_in_0_shape_ch_const_u16, conv2d_39_t_weight_0_ptr_const_s8, conv2d_39_t_out_0_shape_ch_const_u16, conv2d_39_t_weight_1_ptr_const_s32, conv2d_39_t_in_0_fmt_zero_const_s8, conv2d_39_t_out_0_fmt_zero_const_s8, conv2d_39_t_in_0_fmt_scale_const_f32, conv2d_39_t_out_0_fmt_scale_const_f32, conv2d_39_t_weight_0_fmt_scale_const_f32, conv2d_39_l_out_ch_format_const_layer_format_type, conv2d_39_t_out_0_ptr_s8, 1, 448, conv2d_39_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(39, 1, {(stai_ptr) conv2d_39_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_39 */
  /* LITE_KERNEL_SECTION BEGIN slice_28 */
  {
    
  forward_lite_slice_slice_28(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_28 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_37 */
  {
      const ai_i8* conv2d_37_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1408);
    const ai_i8* conv2d_37_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2596);
    const ai_i32* conv2d_37_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3620);
    ai_i8* conv2d_37_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 4160);
    ai_i16* conv2d_37_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 2560);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(37, 1, {(stai_ptr) conv2d_37_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_37_t_in_0_ptr_const_s8, conv2d_37_t_in_0_shape_w_const_u16, conv2d_37_t_in_0_shape_h_const_u16, conv2d_37_l_stride_1_const_u16, conv2d_37_l_stride_0_const_u16, conv2d_37_t_in_0_shape_ch_const_u16, conv2d_37_t_weight_0_ptr_const_s8, conv2d_37_t_out_0_shape_ch_const_u16, conv2d_37_t_weight_1_ptr_const_s32, conv2d_37_t_in_0_fmt_zero_const_s8, conv2d_37_t_out_0_fmt_zero_const_s8, conv2d_37_t_in_0_fmt_scale_const_f32, conv2d_37_t_out_0_fmt_scale_const_f32, conv2d_37_t_weight_0_fmt_scale_const_f32, conv2d_37_l_out_ch_format_const_layer_format_type, conv2d_37_t_out_0_ptr_s8, 1, 448, conv2d_37_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(37, 1, {(stai_ptr) conv2d_37_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_37 */
  /* LITE_KERNEL_SECTION BEGIN slice_26 */
  {
    
  forward_lite_slice_slice_26(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_26 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_35 */
  {
      const ai_i8* conv2d_35_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1408);
    const ai_i8* conv2d_35_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 3748);
    const ai_i32* conv2d_35_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4772);
    ai_i8* conv2d_35_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5312);
    ai_i16* conv2d_35_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 2560);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(35, 1, {(stai_ptr) conv2d_35_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_35_t_in_0_ptr_const_s8, conv2d_35_t_in_0_shape_w_const_u16, conv2d_35_t_in_0_shape_h_const_u16, conv2d_35_l_stride_1_const_u16, conv2d_35_l_stride_0_const_u16, conv2d_35_t_in_0_shape_ch_const_u16, conv2d_35_t_weight_0_ptr_const_s8, conv2d_35_t_out_0_shape_ch_const_u16, conv2d_35_t_weight_1_ptr_const_s32, conv2d_35_t_in_0_fmt_zero_const_s8, conv2d_35_t_out_0_fmt_zero_const_s8, conv2d_35_t_in_0_fmt_scale_const_f32, conv2d_35_t_out_0_fmt_scale_const_f32, conv2d_35_t_weight_0_fmt_scale_const_f32, conv2d_35_l_out_ch_format_const_layer_format_type, conv2d_35_t_out_0_ptr_s8, 1, 448, conv2d_35_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(35, 1, {(stai_ptr) conv2d_35_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_35 */
  /* LITE_KERNEL_SECTION BEGIN slice_24 */
  {
    
  forward_lite_slice_slice_24(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_24 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_33 */
  {
      const ai_i8* conv2d_33_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1408);
    const ai_i8* conv2d_33_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4900);
    const ai_i32* conv2d_33_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 5924);
    ai_i8* conv2d_33_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 6464);
    ai_i16* conv2d_33_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 2560);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(33, 1, {(stai_ptr) conv2d_33_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_33_t_in_0_ptr_const_s8, conv2d_33_t_in_0_shape_w_const_u16, conv2d_33_t_in_0_shape_h_const_u16, conv2d_33_l_stride_1_const_u16, conv2d_33_l_stride_0_const_u16, conv2d_33_t_in_0_shape_ch_const_u16, conv2d_33_t_weight_0_ptr_const_s8, conv2d_33_t_out_0_shape_ch_const_u16, conv2d_33_t_weight_1_ptr_const_s32, conv2d_33_t_in_0_fmt_zero_const_s8, conv2d_33_t_out_0_fmt_zero_const_s8, conv2d_33_t_in_0_fmt_scale_const_f32, conv2d_33_t_out_0_fmt_scale_const_f32, conv2d_33_t_weight_0_fmt_scale_const_f32, conv2d_33_l_out_ch_format_const_layer_format_type, conv2d_33_t_out_0_ptr_s8, 1, 448, conv2d_33_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(33, 1, {(stai_ptr) conv2d_33_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_33 */
  /* LITE_KERNEL_SECTION BEGIN slice_22 */
  {
    
  forward_lite_slice_slice_22(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_22 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_32 */
  {
      const ai_i8* conv2d_32_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1408);
    const ai_i8* conv2d_32_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 6052);
    const ai_i32* conv2d_32_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 7076);
    ai_i8* conv2d_32_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7616);
    ai_i16* conv2d_32_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(32, 1, {(stai_ptr) conv2d_32_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_32_t_in_0_ptr_const_s8, conv2d_32_t_in_0_shape_w_const_u16, conv2d_32_t_in_0_shape_h_const_u16, conv2d_32_l_stride_1_const_u16, conv2d_32_l_stride_0_const_u16, conv2d_32_t_in_0_shape_ch_const_u16, conv2d_32_t_weight_0_ptr_const_s8, conv2d_32_t_out_0_shape_ch_const_u16, conv2d_32_t_weight_1_ptr_const_s32, conv2d_32_t_in_0_fmt_zero_const_s8, conv2d_32_t_out_0_fmt_zero_const_s8, conv2d_32_t_in_0_fmt_scale_const_f32, conv2d_32_t_out_0_fmt_scale_const_f32, conv2d_32_t_weight_0_fmt_scale_const_f32, conv2d_32_l_out_ch_format_const_layer_format_type, conv2d_32_t_out_0_ptr_s8, 1, 448, conv2d_32_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(32, 1, {(stai_ptr) conv2d_32_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_32 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_34 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_34(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_34 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_36 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_36(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_36 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_38 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_38(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_38 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_40 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_40(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_40 */
  /* LITE_KERNEL_SECTION BEGIN pad_42 */
  {
      const ai_ptr pad_42_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 1152);
    ai_ptr pad_42_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 2304);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(42, 1, {(stai_ptr) pad_42_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_42_t_in_0_ptr_const_ptr, pad_42_t_out_0_ptr_ptr, (ai_handle)(pad_42_v_pad_constant_value_const_s8), pad_42_t_in_0_fmt_bitsize_const_s16, pad_42_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(32), (ai_i32)(512), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(42, 1, {(stai_ptr) pad_42_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_42 */
  /* LITE_KERNEL_SECTION BEGIN slice_51 */
  {
    
  forward_lite_slice_slice_51(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_51 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_60 */
  {
      const ai_i8* conv2d_60_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* conv2d_60_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 7204);
    const ai_i32* conv2d_60_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 9252);
    ai_i8* conv2d_60_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3968);
    ai_i16* conv2d_60_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1152);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(60, 1, {(stai_ptr) conv2d_60_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_60_t_in_0_ptr_const_s8, conv2d_60_t_in_0_shape_w_const_u16, conv2d_60_t_in_0_shape_h_const_u16, conv2d_60_l_stride_1_const_u16, conv2d_60_l_stride_0_const_u16, conv2d_60_t_in_0_shape_ch_const_u16, conv2d_60_t_weight_0_ptr_const_s8, conv2d_60_t_out_0_shape_ch_const_u16, conv2d_60_t_weight_1_ptr_const_s32, conv2d_60_t_in_0_fmt_zero_const_s8, conv2d_60_t_out_0_fmt_zero_const_s8, conv2d_60_t_in_0_fmt_scale_const_f32, conv2d_60_t_out_0_fmt_scale_const_f32, conv2d_60_t_weight_0_fmt_scale_const_f32, conv2d_60_l_out_ch_format_const_layer_format_type, conv2d_60_t_out_0_ptr_s8, 1, 768, conv2d_60_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(60, 1, {(stai_ptr) conv2d_60_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_60 */
  /* LITE_KERNEL_SECTION BEGIN slice_49 */
  {
    
  forward_lite_slice_slice_49(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_49 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_58 */
  {
      const ai_i8* conv2d_58_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* conv2d_58_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 9508);
    const ai_i32* conv2d_58_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 11556);
    ai_i8* conv2d_58_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 6272);
    ai_i16* conv2d_58_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1152);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(58, 1, {(stai_ptr) conv2d_58_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_58_t_in_0_ptr_const_s8, conv2d_58_t_in_0_shape_w_const_u16, conv2d_58_t_in_0_shape_h_const_u16, conv2d_58_l_stride_1_const_u16, conv2d_58_l_stride_0_const_u16, conv2d_58_t_in_0_shape_ch_const_u16, conv2d_58_t_weight_0_ptr_const_s8, conv2d_58_t_out_0_shape_ch_const_u16, conv2d_58_t_weight_1_ptr_const_s32, conv2d_58_t_in_0_fmt_zero_const_s8, conv2d_58_t_out_0_fmt_zero_const_s8, conv2d_58_t_in_0_fmt_scale_const_f32, conv2d_58_t_out_0_fmt_scale_const_f32, conv2d_58_t_weight_0_fmt_scale_const_f32, conv2d_58_l_out_ch_format_const_layer_format_type, conv2d_58_t_out_0_ptr_s8, 1, 768, conv2d_58_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(58, 1, {(stai_ptr) conv2d_58_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_58 */
  /* LITE_KERNEL_SECTION BEGIN slice_47 */
  {
    
  forward_lite_slice_slice_47(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_47 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_56 */
  {
      const ai_i8* conv2d_56_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* conv2d_56_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 11812);
    const ai_i32* conv2d_56_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 13860);
    ai_i8* conv2d_56_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8576);
    ai_i16* conv2d_56_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1152);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(56, 1, {(stai_ptr) conv2d_56_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_56_t_in_0_ptr_const_s8, conv2d_56_t_in_0_shape_w_const_u16, conv2d_56_t_in_0_shape_h_const_u16, conv2d_56_l_stride_1_const_u16, conv2d_56_l_stride_0_const_u16, conv2d_56_t_in_0_shape_ch_const_u16, conv2d_56_t_weight_0_ptr_const_s8, conv2d_56_t_out_0_shape_ch_const_u16, conv2d_56_t_weight_1_ptr_const_s32, conv2d_56_t_in_0_fmt_zero_const_s8, conv2d_56_t_out_0_fmt_zero_const_s8, conv2d_56_t_in_0_fmt_scale_const_f32, conv2d_56_t_out_0_fmt_scale_const_f32, conv2d_56_t_weight_0_fmt_scale_const_f32, conv2d_56_l_out_ch_format_const_layer_format_type, conv2d_56_t_out_0_ptr_s8, 1, 768, conv2d_56_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(56, 1, {(stai_ptr) conv2d_56_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_56 */
  /* LITE_KERNEL_SECTION BEGIN slice_45 */
  {
    
  forward_lite_slice_slice_45(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_45 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_54 */
  {
      const ai_i8* conv2d_54_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* conv2d_54_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 14116);
    const ai_i32* conv2d_54_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 16164);
    ai_i8* conv2d_54_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 10880);
    ai_i16* conv2d_54_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1152);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, {(stai_ptr) conv2d_54_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_54_t_in_0_ptr_const_s8, conv2d_54_t_in_0_shape_w_const_u16, conv2d_54_t_in_0_shape_h_const_u16, conv2d_54_l_stride_1_const_u16, conv2d_54_l_stride_0_const_u16, conv2d_54_t_in_0_shape_ch_const_u16, conv2d_54_t_weight_0_ptr_const_s8, conv2d_54_t_out_0_shape_ch_const_u16, conv2d_54_t_weight_1_ptr_const_s32, conv2d_54_t_in_0_fmt_zero_const_s8, conv2d_54_t_out_0_fmt_zero_const_s8, conv2d_54_t_in_0_fmt_scale_const_f32, conv2d_54_t_out_0_fmt_scale_const_f32, conv2d_54_t_weight_0_fmt_scale_const_f32, conv2d_54_l_out_ch_format_const_layer_format_type, conv2d_54_t_out_0_ptr_s8, 1, 768, conv2d_54_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, {(stai_ptr) conv2d_54_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_54 */
  /* LITE_KERNEL_SECTION BEGIN slice_43 */
  {
    
  forward_lite_slice_slice_43(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_43 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_53 */
  {
      const ai_i8* conv2d_53_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* conv2d_53_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 16420);
    const ai_i32* conv2d_53_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 18468);
    ai_i8* conv2d_53_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 13184);
    ai_i16* conv2d_53_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1152);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(53, 1, {(stai_ptr) conv2d_53_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_53_t_in_0_ptr_const_s8, conv2d_53_t_in_0_shape_w_const_u16, conv2d_53_t_in_0_shape_h_const_u16, conv2d_53_l_stride_1_const_u16, conv2d_53_l_stride_0_const_u16, conv2d_53_t_in_0_shape_ch_const_u16, conv2d_53_t_weight_0_ptr_const_s8, conv2d_53_t_out_0_shape_ch_const_u16, conv2d_53_t_weight_1_ptr_const_s32, conv2d_53_t_in_0_fmt_zero_const_s8, conv2d_53_t_out_0_fmt_zero_const_s8, conv2d_53_t_in_0_fmt_scale_const_f32, conv2d_53_t_out_0_fmt_scale_const_f32, conv2d_53_t_weight_0_fmt_scale_const_f32, conv2d_53_l_out_ch_format_const_layer_format_type, conv2d_53_t_out_0_ptr_s8, 1, 768, conv2d_53_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(53, 1, {(stai_ptr) conv2d_53_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_53 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_55 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_55(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_55 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_57 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_57(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_57 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_59 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_59(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_59 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_61 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_61(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_61 */
  /* LITE_KERNEL_SECTION BEGIN pad_63 */
  {
      const ai_ptr pad_63_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 6272);
    ai_ptr pad_63_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(63, 1, {(stai_ptr) pad_63_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_63_t_in_0_ptr_const_ptr, pad_63_t_out_0_ptr_ptr, (ai_handle)(pad_63_v_pad_constant_value_const_s8), pad_63_t_in_0_fmt_bitsize_const_s16, pad_63_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(64), (ai_i32)(2048), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(63, 1, {(stai_ptr) pad_63_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_63 */
  /* LITE_KERNEL_SECTION BEGIN slice_72 */
  {
    
  forward_lite_slice_slice_72(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_72 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_81 */
  {
      const ai_i8* conv2d_81_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4352);
    const ai_i8* conv2d_81_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 18724);
    const ai_i32* conv2d_81_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 22820);
    ai_i8* conv2d_81_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7552);
    ai_i16* conv2d_81_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 6656);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(81, 1, {(stai_ptr) conv2d_81_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_81_t_in_0_ptr_const_s8, conv2d_81_t_in_0_shape_w_const_u16, conv2d_81_t_in_0_shape_h_const_u16, conv2d_81_l_stride_1_const_u16, conv2d_81_l_stride_0_const_u16, conv2d_81_t_in_0_shape_ch_const_u16, conv2d_81_t_weight_0_ptr_const_s8, conv2d_81_t_out_0_shape_ch_const_u16, conv2d_81_t_weight_1_ptr_const_s32, conv2d_81_t_in_0_fmt_zero_const_s8, conv2d_81_t_out_0_fmt_zero_const_s8, conv2d_81_t_in_0_fmt_scale_const_f32, conv2d_81_t_out_0_fmt_scale_const_f32, conv2d_81_t_weight_0_fmt_scale_const_f32, conv2d_81_l_out_ch_format_const_layer_format_type, conv2d_81_t_out_0_ptr_s8, 1, 896, conv2d_81_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(81, 1, {(stai_ptr) conv2d_81_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_81 */
  /* LITE_KERNEL_SECTION BEGIN slice_70 */
  {
    
  forward_lite_slice_slice_70(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_70 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_79 */
  {
      const ai_i8* conv2d_79_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4352);
    const ai_i8* conv2d_79_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23076);
    const ai_i32* conv2d_79_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 27172);
    ai_i8* conv2d_79_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 9856);
    ai_i16* conv2d_79_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 6656);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(79, 1, {(stai_ptr) conv2d_79_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_79_t_in_0_ptr_const_s8, conv2d_79_t_in_0_shape_w_const_u16, conv2d_79_t_in_0_shape_h_const_u16, conv2d_79_l_stride_1_const_u16, conv2d_79_l_stride_0_const_u16, conv2d_79_t_in_0_shape_ch_const_u16, conv2d_79_t_weight_0_ptr_const_s8, conv2d_79_t_out_0_shape_ch_const_u16, conv2d_79_t_weight_1_ptr_const_s32, conv2d_79_t_in_0_fmt_zero_const_s8, conv2d_79_t_out_0_fmt_zero_const_s8, conv2d_79_t_in_0_fmt_scale_const_f32, conv2d_79_t_out_0_fmt_scale_const_f32, conv2d_79_t_weight_0_fmt_scale_const_f32, conv2d_79_l_out_ch_format_const_layer_format_type, conv2d_79_t_out_0_ptr_s8, 1, 896, conv2d_79_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(79, 1, {(stai_ptr) conv2d_79_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_79 */
  /* LITE_KERNEL_SECTION BEGIN slice_68 */
  {
    
  forward_lite_slice_slice_68(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_68 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_77 */
  {
      const ai_i8* conv2d_77_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4352);
    const ai_i8* conv2d_77_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 27428);
    const ai_i32* conv2d_77_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 31524);
    ai_i8* conv2d_77_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 12160);
    ai_i16* conv2d_77_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 6656);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(77, 1, {(stai_ptr) conv2d_77_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_77_t_in_0_ptr_const_s8, conv2d_77_t_in_0_shape_w_const_u16, conv2d_77_t_in_0_shape_h_const_u16, conv2d_77_l_stride_1_const_u16, conv2d_77_l_stride_0_const_u16, conv2d_77_t_in_0_shape_ch_const_u16, conv2d_77_t_weight_0_ptr_const_s8, conv2d_77_t_out_0_shape_ch_const_u16, conv2d_77_t_weight_1_ptr_const_s32, conv2d_77_t_in_0_fmt_zero_const_s8, conv2d_77_t_out_0_fmt_zero_const_s8, conv2d_77_t_in_0_fmt_scale_const_f32, conv2d_77_t_out_0_fmt_scale_const_f32, conv2d_77_t_weight_0_fmt_scale_const_f32, conv2d_77_l_out_ch_format_const_layer_format_type, conv2d_77_t_out_0_ptr_s8, 1, 896, conv2d_77_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(77, 1, {(stai_ptr) conv2d_77_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_77 */
  /* LITE_KERNEL_SECTION BEGIN slice_66 */
  {
    
  forward_lite_slice_slice_66(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_66 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_75 */
  {
      const ai_i8* conv2d_75_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4352);
    const ai_i8* conv2d_75_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 31780);
    const ai_i32* conv2d_75_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 35876);
    ai_i8* conv2d_75_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 14464);
    ai_i16* conv2d_75_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 6656);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(75, 1, {(stai_ptr) conv2d_75_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_75_t_in_0_ptr_const_s8, conv2d_75_t_in_0_shape_w_const_u16, conv2d_75_t_in_0_shape_h_const_u16, conv2d_75_l_stride_1_const_u16, conv2d_75_l_stride_0_const_u16, conv2d_75_t_in_0_shape_ch_const_u16, conv2d_75_t_weight_0_ptr_const_s8, conv2d_75_t_out_0_shape_ch_const_u16, conv2d_75_t_weight_1_ptr_const_s32, conv2d_75_t_in_0_fmt_zero_const_s8, conv2d_75_t_out_0_fmt_zero_const_s8, conv2d_75_t_in_0_fmt_scale_const_f32, conv2d_75_t_out_0_fmt_scale_const_f32, conv2d_75_t_weight_0_fmt_scale_const_f32, conv2d_75_l_out_ch_format_const_layer_format_type, conv2d_75_t_out_0_ptr_s8, 1, 896, conv2d_75_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(75, 1, {(stai_ptr) conv2d_75_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_75 */
  /* LITE_KERNEL_SECTION BEGIN slice_64 */
  {
    
  forward_lite_slice_slice_64(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_64 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_74 */
  {
      const ai_i8* conv2d_74_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4352);
    const ai_i8* conv2d_74_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 36132);
    const ai_i32* conv2d_74_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 40228);
    ai_i8* conv2d_74_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 896);
    ai_i16* conv2d_74_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(74, 1, {(stai_ptr) conv2d_74_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_74_t_in_0_ptr_const_s8, conv2d_74_t_in_0_shape_w_const_u16, conv2d_74_t_in_0_shape_h_const_u16, conv2d_74_l_stride_1_const_u16, conv2d_74_l_stride_0_const_u16, conv2d_74_t_in_0_shape_ch_const_u16, conv2d_74_t_weight_0_ptr_const_s8, conv2d_74_t_out_0_shape_ch_const_u16, conv2d_74_t_weight_1_ptr_const_s32, conv2d_74_t_in_0_fmt_zero_const_s8, conv2d_74_t_out_0_fmt_zero_const_s8, conv2d_74_t_in_0_fmt_scale_const_f32, conv2d_74_t_out_0_fmt_scale_const_f32, conv2d_74_t_weight_0_fmt_scale_const_f32, conv2d_74_l_out_ch_format_const_layer_format_type, conv2d_74_t_out_0_ptr_s8, 1, 896, conv2d_74_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(74, 1, {(stai_ptr) conv2d_74_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_74 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_76 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_76(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_76 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_78 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_78(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_78 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_80 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_80(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_80 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_82 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_82(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_82 */
  /* LITE_KERNEL_SECTION BEGIN slice_84_gather_0 */
  {
    
  forward_lite_gather_slice_84_gather_0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_84_gather_0 */
  /* LITE_KERNEL_SECTION BEGIN gemm_85 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_85(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_85 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 972;

  net_ctx->_outputs[0] = activations[0] + 308;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

