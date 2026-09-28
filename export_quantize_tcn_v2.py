"""
Export a SmallTCN checkpoint to ONNX, then quantize to int8.

BACKGROUND / WHY THIS FILE CHANGED:
The original version of this script used onnxruntime.quantization.
quantize_dynamic (the same approach already confirmed working for the TTM
export). That produces DynamicQuantizeLinear + MatMulInteger + ConvInteger
ops. X-CUBE-AI / ST Edge AI Core's ONNX importer does NOT support that
format -- confirmed directly by running the resulting
small_tcn_solar_int8.onnx through AI Studio's Analyze step:

    NOT IMPLEMENTED: Unsupported layer types:
    ConvInteger, DynamicQuantizeLinear, MatMulInteger

ST's own docs confirm ST Edge AI Core accepts ONNX Runtime's *static*
quantization output instead -- either QOperator ops (QLinearConv,
QLinearMatMul, ...) or QDQ ops (QuantizeLinear/DequantizeLinear pairs
wrapping fp32 Conv/MatMul) -- but explicitly NOT dynamic quantization
("computation of the quantization parameters during execution").

This script now supports both modes via --quant_mode:

  --quant_mode static   (NEW, default -- the fix)
      Runs onnxruntime.quantization.quantize_static using a small set of
      REAL input windows pulled from --calib_dir via solar_data_pipeline,
      preprocessed exactly like training (capacity-normalize + z-score the
      generation channel; sin/cos channels left as-is). Produces QDQ ops
      by default.

  --quant_mode dynamic  (old behavior, kept for reference/comparison)
      Same as before: quantize_dynamic with Conv explicitly included.
      Known to fail X-CUBE-AI's Analyze step -- kept mainly so you can
      reproduce/compare against the earlier error.

IMPORTANT (unchanged from before): SmallTCN is a CNN -- its dilated
causal Conv1d layers hold almost all of its ~38K parameters. If 'Conv'
isn't explicitly included in op_types_to_quantize, quantization only
touches the tiny final Linear head and leaves the conv weights (the vast
majority of the model) in fp32 -- silently producing a misleading "int8"
model that barely shrank. Both quantization paths below always pass
op_types_to_quantize explicitly so that doesn't happen quietly.

Usage:
  # static (recommended -- fixes the X-CUBE-AI import error)
  python export_quantize_tcn.py --checkpoint checkpoints/small_tcn_solar.pt \\
      --out_prefix small_tcn_solar --calib_dir ./processed/val

  # dynamic (old behavior, for comparison only)
  python export_quantize_tcn.py --checkpoint checkpoints/small_tcn_solar.pt \\
      --out_prefix small_tcn_solar --quant_mode dynamic
"""

import argparse
import os

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from tcn_model import SmallTCN


class SolarCalibrationDataReader(CalibrationDataReader):
    """
    Feeds onnxruntime's static quantizer real, correctly-preprocessed
    input windows so it can observe realistic activation ranges. Static
    quantization's whole advantage over dynamic is that it calibrates
    ranges ahead of time instead of at runtime -- feeding it random noise
    here would throw that advantage away and produce meaningless scales.
    """

    def __init__(self, windows: np.ndarray, input_name: str = "input"):
        # windows: (n, context_length, in_channels) float32, already
        # preprocessed exactly like the model saw at training time.
        self.input_name = input_name
        self._windows = windows
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._windows):
            return None
        item = {self.input_name: self._windows[self._idx : self._idx + 1]}
        self._idx += 1
        return item

    def rewind(self):
        self._idx = 0


def build_calibration_windows(
    calib_dir: str,
    context_length: int,
    forecast_length: int,
    norm_mean: float,
    norm_std: float,
    num_samples: int = 128,
    stride: int = 6,
    seed: int = 0,
) -> np.ndarray:
    """
    Pull real (context, forecast) windows from `calib_dir` (a
    ./processed/{train,val,test} split directory) and preprocess the
    context window EXACTLY the way train_solar_tcn.py does:
      1. capacity-normalize the generation channel (divide by site kWp)
      2. z-score that channel using the checkpoint's saved norm_mean/std
      3. leave the 4 sin/cos channels untouched (already in [-1, 1])

    Defaults to --calib_dir=./processed/val, NOT test, on purpose --
    calibration data should never come from the split your final
    reported accuracy numbers are computed on.
    """
    from solar_data_pipeline import (
        METADATA_PATH,
        load_site_capacities,
        load_solar_split,
        make_windows_multi_site,
    )

    capacities = load_site_capacities(METADATA_PATH)
    site_data = load_solar_split(calib_dir)
    X, _, cap = make_windows_multi_site(
        site_data,
        context_length=context_length,
        forecast_length=forecast_length,
        stride=stride,
        capacities=capacities,
    )

    rng = np.random.default_rng(seed)
    if len(X) > num_samples:
        idx = rng.choice(len(X), size=num_samples, replace=False)
        X = X[idx]
        cap = cap[idx]
    else:
        print(
            f"[!] Only {len(X)} windows available in {calib_dir}, using all "
            f"of them (requested {num_samples})."
        )

    X = X.copy()
    X[..., 0] = (X[..., 0] / cap[:, None] - norm_mean) / norm_std
    print(
        f"Built {len(X)} calibration windows from {calib_dir} "
        f"(context={context_length}, in_channels={X.shape[-1]})"
    )
    return X.astype(np.float32)


def _quantize_dynamic(fp32_path, int8_path):
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QUInt8,
        # Explicit on purpose -- see module docstring.
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )


def _quantize_static(
    fp32_path,
    int8_path,
    calib_windows,
    per_channel: bool,
    quant_format: str,
    activation_type: str,
    weight_type: str,
):
    fmt = QuantFormat.QDQ if quant_format == "qdq" else QuantFormat.QOperator
    act_type = QuantType.QUInt8 if activation_type == "quint8" else QuantType.QInt8
    wt_type = QuantType.QInt8 if weight_type == "qint8" else QuantType.QUInt8

    reader = SolarCalibrationDataReader(calib_windows, input_name="input")

    # QOperator mode: ST Edge AI Core does NOT implement QGemm (confirmed
    # via Analyze: "NOT IMPLEMENTED: Unsupported layer types: QGemm").
    # It does support QLinearConv fine. Our model's only Gemm is the tiny
    # final head (~1,170 params, ~3% of the model) -- leaving just that
    # layer in fp32 costs a few KB and avoids the unsupported op entirely,
    # while the four Conv blocks (~97% of params) still get quantized.
    # QDQ mode doesn't have this problem (it never produces QGemm), so
    # only restrict ops here for QOperator.
    if quant_format == "qoperator":
        op_types = ["Conv"]
    else:
        op_types = ["Conv", "MatMul", "Gemm"]

    quantize_static(
        fp32_path,
        int8_path,
        calibration_data_reader=reader,
        quant_format=fmt,
        activation_type=act_type,
        weight_type=wt_type,
        per_channel=per_channel,
        op_types_to_quantize=op_types,
        calibrate_method=CalibrationMethod.MinMax,
    )


def _sanity_check_ops(int8_path, quant_mode, quant_format=None):
    import onnx

    m = onnx.load(int8_path)
    op_types = {n.op_type for n in m.graph.node}
    # NOTE: naive substring match on "Quant"/"Integer" MISSES QLinearConv,
    # QLinearMatMul, QGemm (they contain "Linear"/"Gemm", not "Quant") --
    # list the ones that actually matter explicitly instead.
    quant_op_names = {
        "QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear",
        "ConvInteger", "MatMulInteger",
        "QLinearConv", "QLinearMatMul", "QGemm",
    }
    quant_ops = sorted(op_types & quant_op_names)
    print(f"Quantization-related ops present: {quant_ops}")

    if quant_mode == "dynamic":
        has_conv_quant = "ConvInteger" in op_types
        print(
            f"Conv layers quantized: "
            f"{'YES' if has_conv_quant else 'NO -- CHECK THIS, something is wrong'}"
        )
    else:
        has_qdq = "QuantizeLinear" in op_types or "DequantizeLinear" in op_types
        has_qlinearconv = "QLinearConv" in op_types
        has_qgemm = "QGemm" in op_types
        print(
            f"QDQ/QLinear ops present: "
            f"{'YES' if (has_qdq or has_qlinearconv) else 'NO -- CHECK THIS'}"
        )
        if quant_format == "qoperator":
            print(f"QLinearConv present (conv layers quantized): "
                  f"{'YES' if has_qlinearconv else 'NO -- CHECK THIS'}")
            if has_qgemm:
                print("[!] QGemm present -- this WILL fail ST Edge AI Core's "
                      "Analyze step (confirmed unsupported). The head layer "
                      "should have been excluded from quantization.")
            else:
                print("Head layer (Gemm) correctly left unquantized (fp32) -- "
                      "avoids ST Edge AI Core's unsupported QGemm op.")
        leaked = {"ConvInteger", "DynamicQuantizeLinear", "MatMulInteger"} & op_types
        if leaked:
            print(
                f"[!] WARNING: dynamic-quant ops leaked into a 'static' export "
                f"({sorted(leaked)}) -- something is misconfigured."
            )


def export_and_quantize(
    checkpoint_path,
    out_prefix,
    out_dir="exports",
    quant_mode="static",
    calib_dir="./processed/val",
    num_calib_samples=128,
    calib_stride=6,
    per_channel=True,
    quant_format="qdq",
    activation_type="quint8",
    weight_type="qint8",
):
    os.makedirs(out_dir, exist_ok=True)

    # weights_only=False: safe here since this is your own checkpoint,
    # produced locally by train_solar_tcn.py -- newer PyTorch defaults
    # torch.load to weights_only=True, which refuses to unpickle the
    # numpy scalars (norm_mean/norm_std) this checkpoint also carries.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    context_length = ckpt["context_length"]
    forecast_length = ckpt["forecast_length"]
    in_channels = ckpt.get("in_channels", 1)  # defaults to 1 for old REFIT checkpoints
    norm_mean = ckpt.get("norm_mean")
    norm_std = ckpt.get("norm_std")

    print(
        f"Loaded checkpoint: context_length={context_length}, "
        f"forecast_length={forecast_length}, in_channels={in_channels}"
    )

    model = SmallTCN(
        context_length=context_length,
        forecast_length=forecast_length,
        in_channels=in_channels,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, context_length, in_channels)
    fp32_path = os.path.join(out_dir, f"{out_prefix}.onnx")
    int8_path = os.path.join(out_dir, f"{out_prefix}_int8.onnx")

    # Fixed batch size of 1 -- deployment is single-sample inference on
    # MCU, no reason to carry dynamic-batch overhead into the graph.
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["input"],
        output_names=["forecast"],
        opset_version=17,
        dynamo=False,  # stable legacy exporter -- no onnxscript dependency needed
    )
    fp32_kb = os.path.getsize(fp32_path) / 1024
    print(f"Exported fp32 ONNX -> {fp32_path} ({fp32_kb:.1f} KB)")

    if quant_mode == "dynamic":
        _quantize_dynamic(fp32_path, int8_path)

    elif quant_mode == "static":
        if in_channels != 5 or norm_mean is None or norm_std is None:
            raise ValueError(
                "Static quantization here builds calibration data via "
                "solar_data_pipeline, which assumes a solar checkpoint "
                "(in_channels=5, norm_mean/norm_std saved in the checkpoint). "
                f"This checkpoint has in_channels={in_channels}. Point "
                "--checkpoint at a solar checkpoint, or pass --quant_mode "
                "dynamic instead."
            )
        calib_windows = build_calibration_windows(
            calib_dir,
            context_length,
            forecast_length,
            norm_mean,
            norm_std,
            num_samples=num_calib_samples,
            stride=calib_stride,
        )
        _quantize_static(
            fp32_path,
            int8_path,
            calib_windows,
            per_channel=per_channel,
            quant_format=quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
        )
    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")

    int8_kb = os.path.getsize(int8_path) / 1024
    print(f"Exported int8 ONNX -> {int8_path} ({int8_kb:.1f} KB)")
    print(f"\nCompression: {fp32_kb:.1f} KB -> {int8_kb:.1f} KB ({fp32_kb / int8_kb:.2f}x)")

    _sanity_check_ops(int8_path, quant_mode, quant_format)

    return fp32_path, int8_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--out_dir", default="exports")
    parser.add_argument(
        "--quant_mode",
        choices=["static", "dynamic"],
        default="static",
        help=(
            "static (default) fixes the X-CUBE-AI 'Unsupported layer types: "
            "ConvInteger, DynamicQuantizeLinear, MatMulInteger' error. "
            "dynamic reproduces the old behavior, for comparison only."
        ),
    )
    parser.add_argument(
        "--calib_dir",
        default="./processed/val",
        help="Split directory to pull calibration windows from (static mode "
        "only). Defaults to val, not test, to avoid leaking test data into "
        "the quantization process.",
    )
    parser.add_argument("--num_calib_samples", type=int, default=128)
    parser.add_argument("--calib_stride", type=int, default=6)
    parser.add_argument("--per_channel", dest="per_channel", action="store_true", default=True)
    parser.add_argument("--no_per_channel", dest="per_channel", action="store_false")
    parser.add_argument(
        "--quant_format",
        choices=["qdq", "qoperator"],
        default="qdq",
        help="If ST Edge AI Core's Analyze rejects one format, try the other.",
    )
    parser.add_argument("--activation_type", choices=["quint8", "qint8"], default="quint8")
    parser.add_argument("--weight_type", choices=["qint8", "quint8"], default="qint8")
    args = parser.parse_args()

    export_and_quantize(
        args.checkpoint,
        args.out_prefix,
        args.out_dir,
        quant_mode=args.quant_mode,
        calib_dir=args.calib_dir,
        num_calib_samples=args.num_calib_samples,
        calib_stride=args.calib_stride,
        per_channel=args.per_channel,
        quant_format=args.quant_format,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
    )
