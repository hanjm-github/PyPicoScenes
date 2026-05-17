from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import os
import random
import re
import subprocess
import sys
import threading
import time

import numpy as np


DEFAULT_PHY = "phy0"
DEFAULT_CHANNEL = "5280 160 5250"
DEFAULT_HANDLER_NAME = "csi_motion_detector_v2"
MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}$")


@contextmanager
def suppress_native_output(enabled: bool = True):
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


@dataclass
class MotionResult:
    calibrated: bool
    motion: bool
    score: float | None
    threshold: float | None
    calibration_count: int
    ignored_reason: str | None = None


class FrameDeltaMotionDetector:
    def __init__(
        self,
        calibration_frames: int = 80,
        threshold: float | None = None,
        threshold_sigma: float = 6.0,
        hit_frames: int = 3,
        hold_seconds: float = 1.0,
        feature_bins: int = 128,
    ) -> None:
        if calibration_frames < 3:
            raise ValueError("calibration_frames must be at least 3")
        if threshold_sigma < 0:
            raise ValueError("threshold_sigma must be non-negative")
        if hit_frames < 1:
            raise ValueError("hit_frames must be at least 1")
        if hold_seconds < 0:
            raise ValueError("hold_seconds must be non-negative")
        if feature_bins < 8:
            raise ValueError("feature_bins must be at least 8")

        self.calibration_frames = calibration_frames
        self.threshold_override = threshold
        self.threshold_sigma = threshold_sigma
        self.hit_frames = hit_frames
        self.hold_seconds = hold_seconds
        self.feature_bins = feature_bins

        self.threshold = threshold
        self._previous_feature: np.ndarray | None = None
        self._noise_scores: list[float] = []
        self._valid_features = 0
        self._consecutive_hits = 0
        self._last_motion_time: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.threshold is not None and self._valid_features >= self.calibration_frames

    def current_motion(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self._last_motion_time is not None and (now - self._last_motion_time) <= self.hold_seconds

    def process_vector(self, vector: np.ndarray, now: float | None = None) -> MotionResult:
        now = time.monotonic() if now is None else now
        feature = robust_log_magnitude_feature(vector, self.feature_bins)
        if feature is None:
            return self._result(False, None, "invalid CSI magnitude vector")

        self._valid_features += 1
        if self._previous_feature is None:
            self._previous_feature = feature
            return self._result(False, None, None)

        score = frame_delta_score(self._previous_feature, feature)
        self._previous_feature = feature

        if not self.calibrated:
            self._noise_scores.append(score)
            if self._valid_features >= self.calibration_frames:
                self._finish_calibration()
            return self._result(False, score, None)

        hit = bool(score > float(self.threshold))
        if hit:
            self._consecutive_hits += 1
            if self._consecutive_hits >= self.hit_frames:
                self._last_motion_time = now
        else:
            self._consecutive_hits = 0

        return self._result(self.current_motion(now), score, None)

    def _finish_calibration(self) -> None:
        scores = np.asarray(self._noise_scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            self.threshold = self.threshold_override if self.threshold_override is not None else 0.1
            return

        if self.threshold_override is not None:
            self.threshold = self.threshold_override
            return

        center = float(np.median(scores))
        mad = float(np.median(np.abs(scores - center)))
        robust_sigma = 1.4826 * mad
        self.threshold = max(0.08, center * 3.0, center + self.threshold_sigma * robust_sigma)

    def _result(
        self,
        motion: bool,
        score: float | None,
        ignored_reason: str | None,
    ) -> MotionResult:
        return MotionResult(
            calibrated=self.calibrated,
            motion=motion,
            score=score,
            threshold=self.threshold,
            calibration_count=self._valid_features,
            ignored_reason=ignored_reason,
        )


def robust_log_magnitude_feature(vector: np.ndarray, bins: int) -> np.ndarray | None:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 4:
        return None

    log_values = np.log(np.maximum(values, 1e-12))
    center = float(np.median(log_values))
    mad = float(np.median(np.abs(log_values - center)))
    scale = max(1.4826 * mad, float(np.std(log_values)), 1e-6)
    normalized = np.clip((log_values - center) / scale, -8.0, 8.0)
    return resample_1d(normalized, bins)


def resample_1d(values: np.ndarray, bins: int) -> np.ndarray:
    if values.size == bins:
        return values.astype(np.float64, copy=False)
    if values.size == 1:
        return np.full(bins, float(values[0]), dtype=np.float64)

    source_x = np.linspace(0.0, 1.0, values.size)
    target_x = np.linspace(0.0, 1.0, bins)
    return np.interp(target_x, source_x, values).astype(np.float64, copy=False)


def frame_delta_score(previous: np.ndarray, current: np.ndarray) -> float:
    diff = np.abs(current - previous)
    return float(0.65 * np.percentile(diff, 75) + 0.35 * np.mean(diff))


def csi_magnitude_vector_from_frame(frame) -> np.ndarray | None:
    try:
        if not getattr(frame, "csiSegment", None):
            return None

        csi = frame.csiSegment.getCSI()
        if not csi:
            return None

        csi.removeCSDAndInterpolateCSI()
        magnitudes = csi.magnitudeArray.array
        size = int(magnitudes.size())
        if size == 0:
            return None

        return np.fromiter((float(magnitudes[i]) for i in range(size)), dtype=np.float64, count=size)
    except Exception:
        return None


def parse_channel_tuple(channel: str) -> tuple[float, float, float]:
    parts = channel.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError('channel must look like "5280 160 5250"')

    values = [float(part) for part in parts]
    channel_hz = tuple(value * 1e6 if abs(value) < 1e6 else value for value in values)
    validate_channel_tuple(channel_hz)
    return channel_hz


def validate_channel_tuple(channel_hz: tuple[float, float, float]) -> None:
    control_hz, bandwidth_hz, center_hz = channel_hz
    control_mhz = control_hz / 1e6
    bandwidth_mhz = bandwidth_hz / 1e6
    center_mhz = center_hz / 1e6

    if (
        abs(bandwidth_mhz - 160.0) < 0.5
        and 5000.0 <= control_mhz <= 6000.0
        and abs((control_mhz - 5000.0) % 20.0) > 0.5
    ):
        nearest_lower = control_mhz - 10.0
        nearest_upper = control_mhz + 10.0
        raise ValueError(
            f"{control_mhz:g} MHz is not a valid 5 GHz 20 MHz primary/control frequency. "
            f"For a 160 MHz channel centered at {center_mhz:g} MHz, use a valid primary such as "
            f'"{nearest_lower:g} 160 {center_mhz:g}" or "{nearest_upper:g} 160 {center_mhz:g}". '
            "Your 5290 MHz value is likely the 80 MHz center1 frequency reported by iw."
        )


def run_array_status() -> str:
    try:
        proc = subprocess.run(["array_status"], check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("array_status was not found; install PicoScenes tools or pass --nic-id") from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        raise RuntimeError(f"array_status failed:\n{output.strip()}") from exc

    return proc.stdout


def resolve_nic_id(phy: str) -> str:
    if phy.isdigit():
        return phy

    output = run_array_status()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[2] == phy:
            return parts[0]

    raise RuntimeError(f"Could not find {phy!r} in array_status output:\n{output.strip()}")


def get_nic_status(nic_id: str) -> dict[str, object] | None:
    for line in run_array_status().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] != str(nic_id):
            continue

        status: dict[str, object] = {
            "nic_id": parts[0],
            "dev": parts[1],
            "phy": parts[2],
            "monitor": None,
            "channel_mhz": None,
            "raw": line,
        }
        idx = 3
        if idx < len(parts) and not MAC_RE.match(parts[idx]):
            status["monitor"] = parts[idx]
            idx += 1

        if idx < len(parts) and MAC_RE.match(parts[idx]):
            idx += 1
        if idx < len(parts) and MAC_RE.match(parts[idx]):
            idx += 1

        if idx + 2 < len(parts):
            try:
                status["channel_mhz"] = tuple(float(value) for value in parts[idx:idx + 3])
            except ValueError:
                pass

        return status

    return None


def channel_matches_status(status: dict[str, object] | None, channel_hz: tuple[float, float, float]) -> bool:
    if not status or not status.get("channel_mhz"):
        return False

    channel_mhz = tuple(value / 1e6 for value in channel_hz)
    return all(abs(left - right) < 0.5 for left, right in zip(status["channel_mhz"], channel_mhz))


def prepare_nic(nic_id: str, channel: str) -> None:
    try:
        subprocess.run(["array_prepare_for_picoscenes", nic_id, channel], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("array_prepare_for_picoscenes was not found") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"array_prepare_for_picoscenes failed with exit code {exc.returncode}") from exc


def configure_probe_tx(
    nic,
    mcs: int,
    PacketFormatEnum,
    GuardIntervalEnum,
    ChannelBandwidthEnum,
    ChannelCodingEnum,
    setTxParameters,
) -> None:
    tx_parameters = nic.getUserSpecifiedTxParameters()
    tx_parameters.frameType = PacketFormatEnum.PacketFormat_HESU
    tx_parameters.guardInterval = GuardIntervalEnum.GI_3200
    tx_parameters.cbw = ChannelBandwidthEnum.CBW_160
    tx_parameters.coding[0] = ChannelCodingEnum.LDPC
    tx_parameters.mcs[0] = mcs
    setTxParameters(nic, tx_parameters)


def set_magic_destination_filter(nic, std, MagicIntel123456) -> None:
    macs = std.vector[std.array[std.uint8_t, 6]]()
    macs.push_back(MagicIntel123456)
    nic.getFrontEnd().setDestinationMACAddressFilter(macs)


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def format_age(seconds: float | None) -> str:
    return "never" if seconds is None else f"{seconds:.1f}s"


def compact_error(message: str | None, limit: int = 96) -> str:
    if not message:
        return "none"
    return re.sub(r"\s+", "_", message.strip())[:limit]


def format_status(
    result: MotionResult,
    rx_csi: int,
    ignored_frames: int,
    tx_frames: int,
    tx_errors: int,
    tx_active: bool,
    tx_start_error: str | None,
    no_csi_for: float | None,
    last_tx_error: str | None,
    no_csi_warn_seconds: float,
) -> str:
    parts = [
        f"motion={result.motion}",
        f"calibrated={result.calibrated}",
        f"score={format_float(result.score)}",
        f"threshold={format_float(result.threshold)}",
        f"calibration={result.calibration_count}",
        f"rx_csi={rx_csi}",
        f"ignored={ignored_frames}",
        f"tx_active={tx_active}",
        f"tx={tx_frames}",
        f"tx_errors={tx_errors}",
        f"no_csi_for={format_age(no_csi_for)}",
    ]

    if tx_start_error:
        parts.append(f"tx_start_error={compact_error(tx_start_error)}")
    if tx_errors:
        parts.append(f"last_tx_error={compact_error(last_tx_error)}")
    if tx_start_error:
        parts.append("status=tx_injection_unavailable_rx_only")
    elif rx_csi == 0 and tx_active and tx_frames > 0 and no_csi_for is not None and no_csi_for >= no_csi_warn_seconds:
        parts.append("status=no_csi_received_from_same_nic_tx_rx")
    elif rx_csi == 0:
        parts.append("status=waiting_for_csi")
    elif result.ignored_reason:
        parts.append(f"status={compact_error(result.ignored_reason)}")

    return " ".join(parts)


def run_live(args: argparse.Namespace) -> int:
    if args.probe_rate <= 0:
        raise ValueError("probe_rate must be greater than 0")
    if args.print_every <= 0:
        raise ValueError("print_every must be greater than 0")
    if args.mcs < 0:
        raise ValueError("mcs must be non-negative")

    nic_id = args.nic_id or resolve_nic_id(args.phy)
    channel_hz = parse_channel_tuple(args.channel)
    print(f'Using phy={args.phy} nic_id={nic_id} channel="{args.channel}" active_probe_rate={args.probe_rate:g}Hz', flush=True)

    if args.prepare:
        print("Preparing NIC for PicoScenes...", flush=True)
        prepare_nic(nic_id, args.channel)
        status = get_nic_status(nic_id)
        if status and status.get("channel_mhz") and not channel_matches_status(status, channel_hz):
            raise RuntimeError(
                "array_prepare_for_picoscenes completed, but array_status does not show "
                f'channel "{args.channel}" for nic_id={nic_id}.'
            )
    elif not args.runtime_retune:
        status = get_nic_status(nic_id)
        if status and status.get("channel_mhz") and not channel_matches_status(status, channel_hz):
            raise RuntimeError(
                f'nic_id={nic_id} is not tuned to "{args.channel}". '
                "Run with --prepare, or use --runtime-retune if your NIC supports live retuning."
            )
        if not status or not status.get("channel_mhz"):
            print("Warning: channel was not visible in array_status; assuming the NIC is already prepared.", flush=True)

    quiet_picoscenes = not args.picoscenes_logs
    with suppress_native_output(quiet_picoscenes):
        from PyPicoScenes.PyPicoScenes import (
            ChannelBandwidthEnum,
            ChannelCodingEnum,
            EchoProbeInjectionContent,
            EchoProbePacketFrameType,
            EchoProbeParameters,
            GuardIntervalEnum,
            MagicIntel123456,
            PacketFormatEnum,
            setTxParameters,
            std,
            getNic,
            picoscenes_start,
            picoscenes_stop,
            picoscenes_wait,
        )
        from PyPicoScenes.buildFrames import buildBasicFrame

    detector = FrameDeltaMotionDetector(
        calibration_frames=args.calibration_frames,
        threshold=args.threshold,
        threshold_sigma=args.threshold_sigma,
        hit_frames=args.hit_frames,
        hold_seconds=args.hold_seconds,
        feature_bins=args.feature_bins,
    )

    lock = threading.Lock()
    stop_event = threading.Event()
    latest_result = MotionResult(False, False, None, None, 0)
    last_csi_time: float | None = None
    rx_csi = 0
    ignored_frames = 0
    tx_frames = 0
    tx_errors = 0
    tx_active = False
    tx_start_error: str | None = None
    last_tx_error: str | None = None
    nic = None
    probe_thread: threading.Thread | None = None
    platform_started = False
    start_time = time.monotonic()

    def handle_frame(frame):
        nonlocal ignored_frames, last_csi_time, latest_result, rx_csi
        now = time.monotonic()
        vector = csi_magnitude_vector_from_frame(frame)
        with lock:
            if vector is None:
                ignored_frames += 1
                return True

            rx_csi += 1
            latest_result = detector.process_vector(vector, now=now)
            last_csi_time = now
            if latest_result.ignored_reason:
                ignored_frames += 1

        return True

    def probe_loop():
        nonlocal last_tx_error, tx_errors, tx_frames
        parameters = EchoProbeParameters()
        parameters.randomMAC = False
        parameters.injectorContent = EchoProbeInjectionContent.Full
        parameters.tx_delay_us = int(1_000_000 / args.probe_rate)
        interval = 1.0 / args.probe_rate

        while not stop_event.is_set():
            loop_started = time.monotonic()
            try:
                task_id = random.randint(9999, 65535)
                tx_frame = buildBasicFrame(
                    task_id,
                    EchoProbePacketFrameType.SimpleInjectionFrameType,
                    nic,
                    parameters,
                )
                nic.transmitPicoScenesFrameSync(tx_frame)
                with lock:
                    tx_frames += 1
                    last_tx_error = None
            except Exception as exc:
                with lock:
                    tx_errors += 1
                    last_tx_error = str(exc)

            elapsed = time.monotonic() - loop_started
            stop_event.wait(max(0.0, interval - elapsed))

    try:
        with suppress_native_output(quiet_picoscenes):
            picoscenes_start()
        platform_started = True

        with suppress_native_output(quiet_picoscenes):
            nic = getNic(str(nic_id))

        if args.runtime_retune:
            control_hz, bandwidth_hz, center_hz = channel_hz
            with suppress_native_output(quiet_picoscenes):
                retune_status = nic.getFrontEnd().setChannelAndBandwidth(control_hz, bandwidth_hz, center_hz)
            if int(retune_status) != 0:
                raise RuntimeError(f"setChannelAndBandwidth failed with status {retune_status}")

        with suppress_native_output(quiet_picoscenes):
            set_magic_destination_filter(nic, std, MagicIntel123456)
            nic.startRxService()
            nic.registerGeneralHandler(DEFAULT_HANDLER_NAME, handle_frame)

        if args.rx_only:
            print("Listening for CSI in RX-only mode. Press Ctrl+C to stop.", flush=True)
        else:
            try:
                with suppress_native_output(quiet_picoscenes):
                    configure_probe_tx(
                        nic,
                        args.mcs,
                        PacketFormatEnum,
                        GuardIntervalEnum,
                        ChannelBandwidthEnum,
                        ChannelCodingEnum,
                        setTxParameters,
                    )
                    nic.startTxService()
                tx_active = True
            except Exception as exc:
                tx_start_error = str(exc)
                try:
                    with suppress_native_output(quiet_picoscenes):
                        nic.stopTxService()
                except Exception:
                    pass
                if args.require_tx:
                    raise RuntimeError(
                        "TX injection failed. This NIC/driver may not support PicoScenes packet injection "
                        "on phy0; retry without --require-tx to listen RX-only, or use a separate supported TX device."
                    ) from exc
                print(
                    "Warning: TX injection is unavailable on this NIC/driver; continuing RX-only. "
                    "For active CSI probes, use a separate supported transmitter or a NIC with PicoScenes TX injection support.",
                    flush=True,
                )

            if tx_active:
                probe_thread = threading.Thread(target=probe_loop, name="csi-probe-tx", daemon=True)
                probe_thread.start()
                print("Listening for CSI while transmitting probes. Press Ctrl+C to stop.", flush=True)
            else:
                print("Listening for CSI in RX-only mode. Press Ctrl+C to stop.", flush=True)

        while True:
            time.sleep(args.print_every)
            now = time.monotonic()
            with lock:
                no_csi_for = (now - start_time) if last_csi_time is None else (now - last_csi_time)
                display_result = MotionResult(
                    calibrated=latest_result.calibrated,
                    motion=detector.current_motion(now) if latest_result.calibrated else False,
                    score=latest_result.score,
                    threshold=latest_result.threshold,
                    calibration_count=latest_result.calibration_count,
                    ignored_reason=latest_result.ignored_reason,
                )
                print(
                    format_status(
                        display_result,
                        rx_csi,
                        ignored_frames,
                        tx_frames,
                        tx_errors,
                        tx_active,
                        tx_start_error,
                        no_csi_for,
                        last_tx_error,
                        args.no_csi_warn_seconds,
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopping CSI motion detector v2...", flush=True)
        return 0
    finally:
        stop_event.set()
        if probe_thread is not None:
            probe_thread.join(timeout=2.0)

        if nic is not None:
            for cleanup in (
                lambda: nic.unregisterGeneralHandler(DEFAULT_HANDLER_NAME),
                nic.stopRxService,
                nic.stopTxService,
            ):
                try:
                    with suppress_native_output(quiet_picoscenes):
                        cleanup()
                except Exception:
                    pass

        if platform_started:
            try:
                with suppress_native_output(quiet_picoscenes):
                    picoscenes_stop()
            except Exception:
                pass
            try:
                with suppress_native_output(quiet_picoscenes):
                    picoscenes_wait()
            except Exception:
                pass


def run_self_test() -> int:
    rng = np.random.default_rng(7)
    detector = FrameDeltaMotionDetector(
        calibration_frames=20,
        threshold=None,
        threshold_sigma=6.0,
        hit_frames=3,
        hold_seconds=0.6,
        feature_bins=128,
    )

    x = np.linspace(0.0, 2.0 * np.pi, 256)
    base = np.exp(0.18 * np.sin(x) + 0.08 * np.cos(3.0 * x))
    now = 0.0

    result = None
    for _ in range(20):
        vector = base * np.exp(rng.normal(0.0, 0.006, base.size))
        result = detector.process_vector(vector, now=now)
        now += 0.05

    assert result is not None and result.calibrated and not result.motion

    calm = detector.process_vector(base * np.exp(rng.normal(0.0, 0.006, base.size)), now=now)
    assert calm.calibrated and not calm.motion
    now += 0.05

    motion_result = None
    for step in range(3):
        moving = base * np.exp(0.55 * np.sin(2.3 * x + step) + 0.25 * np.cos(5.1 * x - 0.7 * step))
        motion_result = detector.process_vector(moving * np.exp(rng.normal(0.0, 0.006, moving.size)), now=now)
        now += 0.05

    assert motion_result is not None and motion_result.motion

    now += 0.8
    final = None
    for _ in range(5):
        final = detector.process_vector(base * np.exp(rng.normal(0.0, 0.006, base.size)), now=now)
        now += 0.2

    assert final is not None and final.calibrated and not final.motion
    print("self-test ok: False -> True -> False")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Active-probe CSI motion detector for PicoScenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phy", default=DEFAULT_PHY, help="Linux PHY name to resolve via array_status.")
    parser.add_argument("--nic-id", default=None, help="PicoScenes PhyPath ID override.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL, help='Channel tuple: "control bandwidth center".')
    parser.add_argument(
        "--prepare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run array_prepare_for_picoscenes before starting PicoScenes.",
    )
    parser.add_argument("--runtime-retune", action="store_true", help="Call setChannelAndBandwidth after PicoScenes starts.")
    parser.add_argument("--probe-rate", type=float, default=20.0, help="Probe transmit rate in frames per second.")
    parser.add_argument("--mcs", type=int, default=0, help="HE MCS index for transmitted 160 MHz probes.")
    parser.add_argument("--rx-only", action="store_true", help="Skip TX injection and only listen for received CSI.")
    parser.add_argument("--require-tx", action="store_true", help="Exit if TX injection cannot be started.")
    parser.add_argument("--calibration-frames", type=int, default=80, help="Valid CSI frames used to learn no-motion noise.")
    parser.add_argument("--threshold", type=float, default=None, help="Manual motion threshold. Auto threshold is used when omitted.")
    parser.add_argument("--threshold-sigma", type=float, default=6.0, help="MAD multiplier for auto threshold.")
    parser.add_argument("--hit-frames", type=int, default=3, help="Consecutive threshold hits required for motion=True.")
    parser.add_argument("--hold-seconds", type=float, default=1.0, help="Keep motion=True for this long after the last confirmed hit.")
    parser.add_argument("--feature-bins", type=int, default=128, help="Resampled CSI feature length.")
    parser.add_argument("--print-every", type=float, default=0.5, help="Seconds between status lines.")
    parser.add_argument("--no-csi-warn-seconds", type=float, default=3.0, help="Seconds before printing the same-NIC no-CSI warning.")
    parser.add_argument("--picoscenes-logs", action="store_true", help="Show native PicoScenes/cppyy startup and shutdown logs.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic detector test without PicoScenes or NIC access.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            return run_self_test()
        return run_live(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
