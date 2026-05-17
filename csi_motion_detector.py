from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np


DEFAULT_PHY = "phy0"
DEFAULT_CHANNEL = "auto"
DEFAULT_HANDLER_NAME = "csi_motion_detector"
DEFAULT_STATUS_LOG = "csi_motion_status.log"
MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}$")
IW_CHANNEL_RE = re.compile(
    r"channel\s+\d+\s+\((?P<control>[0-9.]+)\s*MHz\),\s*"
    r"width:\s*(?P<width>[0-9.]+)\s*MHz"
    r"(?:,\s*center1:\s*(?P<center>[0-9.]+)\s*MHz)?"
)
NO_CSI_AGE = "never"


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


@contextmanager
def native_output_session(enabled: bool = True):
    if not enabled:
        yield sys.stdout
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    user_output = os.fdopen(os.dup(1), "w", buffering=1)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield user_output
    finally:
        user_output.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        user_output.close()


@dataclass
class MotionResult:
    calibrated: bool
    motion: bool
    score: float | None
    threshold: float | None
    calibration_count: int
    ignored_reason: str | None = None


class CSIMotionDetector:
    def __init__(
        self,
        calibration_frames: int = 80,
        threshold: float | None = None,
        min_frames: int = 3,
        hold_seconds: float = 1.0,
        baseline_alpha: float = 0.02,
        threshold_sigma: float = 6.0,
        feature_bins: int = 128,
    ) -> None:
        if calibration_frames < 3:
            raise ValueError("calibration_frames must be at least 3")
        if min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        if hold_seconds < 0:
            raise ValueError("hold_seconds must be non-negative")
        if not 0 <= baseline_alpha <= 1:
            raise ValueError("baseline_alpha must be between 0 and 1")
        if threshold_sigma < 0:
            raise ValueError("threshold_sigma must be non-negative")
        if feature_bins < 8:
            raise ValueError("feature_bins must be at least 8")

        self.calibration_frames = calibration_frames
        self.threshold_override = threshold
        self.threshold = threshold
        self.min_frames = min_frames
        self.hold_seconds = hold_seconds
        self.baseline_alpha = baseline_alpha
        self.threshold_sigma = threshold_sigma
        self.feature_bins = feature_bins

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
            if self._consecutive_hits >= self.min_frames:
                self._last_motion_time = now
        else:
            self._consecutive_hits = 0

        return self._result(self.current_motion(now), score, None)

    def _finish_calibration(self) -> None:
        scores = np.asarray(self._noise_scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if self.threshold_override is not None:
            self.threshold = self.threshold_override
            return
        if scores.size == 0:
            self.threshold = 0.08
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


def csi_magnitude_vector_from_segment(csi_segment) -> np.ndarray | None:
    try:
        if not csi_segment:
            return None

        csi = csi_segment.getCSI()
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


def get_csi_segment(frame):
    try:
        return getattr(frame, "csiSegment", None)
    except Exception:
        return None


def csi_magnitude_vector_from_frame(frame) -> np.ndarray | None:
    return csi_magnitude_vector_from_segment(get_csi_segment(frame))


def parse_channel_tuple(channel: str) -> tuple[float, float, float]:
    parts = channel.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError('channel must look like "5280 80 5290"')

    values = [float(part) for part in parts]
    return tuple(value * 1e6 if abs(value) < 1e6 else value for value in values)


def format_channel_tuple(channel_hz: tuple[float, float, float]) -> str:
    return " ".join(format_mhz(value) for value in channel_hz)


def parse_iw_dev_channel(output: str, phy: str) -> tuple[float, float, float] | None:
    wanted_phy = phy[3:] if phy.startswith("phy") else phy
    in_wanted_phy = False

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("phy#"):
            in_wanted_phy = line[4:] == wanted_phy
            continue
        if not in_wanted_phy:
            continue

        match = IW_CHANNEL_RE.search(line)
        if not match:
            continue

        control_mhz = float(match.group("control"))
        width_mhz = float(match.group("width"))
        center_mhz = float(match.group("center") or control_mhz)
        return control_mhz * 1e6, width_mhz * 1e6, center_mhz * 1e6

    return None


def detect_current_channel(phy: str) -> tuple[float, float, float]:
    try:
        proc = subprocess.run(["iw", "dev"], check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError('iw was not found; pass an explicit --channel like "5280 80 5290"') from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        raise RuntimeError(f"iw dev failed:\n{output.strip()}") from exc

    channel_hz = parse_iw_dev_channel(proc.stdout, phy)
    if channel_hz is None:
        raise RuntimeError(
            f'Could not auto-detect a channel for {phy!r}; pass an explicit --channel like "5280 80 5290".'
        )
    return channel_hz


def format_mhz(value_hz: float) -> str:
    return f"{value_hz / 1e6:g}"


def array_prepare_channel_config(channel_hz: tuple[float, float, float]) -> str:
    control_hz, bandwidth_hz, center_hz = channel_hz
    bandwidth_mhz = bandwidth_hz / 1e6

    if abs(bandwidth_mhz - 20.0) < 0.5 and abs(control_hz - center_hz) < 0.5e6:
        return f"{format_mhz(control_hz)} HT20"

    return f"{format_mhz(control_hz)} {format_mhz(bandwidth_hz)} {format_mhz(center_hz)}"


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


def get_default_gateway() -> tuple[str, str | None] | None:
    try:
        proc = subprocess.run(["ip", "route", "show", "default"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts or parts[0] != "default":
            continue
        try:
            gateway = parts[parts.index("via") + 1]
        except (ValueError, IndexError):
            continue
        interface = None
        if "dev" in parts:
            try:
                interface = parts[parts.index("dev") + 1]
            except IndexError:
                pass
        return gateway, interface

    return None


def resolve_ping_target(target: str | None) -> tuple[str, str | None] | None:
    if not target:
        return None
    if target.lower() != "auto":
        return target, None

    gateway = get_default_gateway()
    if gateway is None:
        raise RuntimeError("Could not detect a default gateway for --ping-target auto")
    return gateway


def start_ping_helper(target: str, interval: float, interface: str | None = None) -> subprocess.Popen:
    if interval < 0.2:
        raise ValueError("ping_interval must be at least 0.2 seconds for normal user ping")

    cmd = ["ping", "-n", "-i", f"{interval:g}"]
    if interface:
        cmd.extend(["-I", interface])
    cmd.append(target)

    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise RuntimeError("ping was not found") from exc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)


def format_csi_age(age_seconds: float | None) -> str:
    if age_seconds is None:
        return NO_CSI_AGE
    return f"{age_seconds:.1f}s"


def format_result(
    result: MotionResult,
    valid_frames: int,
    ignored_frames: int,
    skipped_frames: int,
    csi_age_seconds: float | None = None,
) -> str:
    csi_age = format_csi_age(csi_age_seconds)
    if not result.calibrated:
        return (
            f"calibrating={result.calibration_count} "
            f"motion=False valid_frames={valid_frames} ignored_frames={ignored_frames} "
            f"skipped_frames={skipped_frames} csi_age={csi_age}"
        )

    score = "n/a" if result.score is None else f"{result.score:.5f}"
    threshold = "n/a" if result.threshold is None else f"{result.threshold:.5f}"
    return (
        f"motion={result.motion} score={score} threshold={threshold} "
        f"valid_frames={valid_frames} ignored_frames={ignored_frames} "
        f"skipped_frames={skipped_frames} csi_age={csi_age}"
    )


def emit(message: str = "", output=None) -> None:
    print(message, file=output or sys.stdout, flush=True)


def run_live(args: argparse.Namespace) -> int:
    if args.print_every <= 0:
        raise ValueError("print_every must be greater than 0")
    if args.ping_target and args.ping_interval < 0.2:
        raise ValueError("ping_interval must be at least 0.2 seconds")
    if args.max_csi_rate < 0:
        raise ValueError("max_csi_rate must be non-negative")

    nic_id = args.nic_id or resolve_nic_id(args.phy)
    if args.channel.lower() == "auto":
        channel_hz = detect_current_channel(args.phy)
        channel_label = format_channel_tuple(channel_hz)
    else:
        channel_hz = parse_channel_tuple(args.channel)
        channel_label = args.channel
    control_hz, bandwidth_hz, center_hz = channel_hz
    ping_target_info = None
    if args.ping_target:
        try:
            ping_target_info = resolve_ping_target(args.ping_target)
        except RuntimeError as exc:
            print(f"Ping helper disabled: {exc}", flush=True)

    prepare_channel = array_prepare_channel_config(channel_hz)

    print(f"Using phy={args.phy} nic_id={nic_id} channel=\"{channel_label}\"", flush=True)
    if args.prepare:
        print(f"Preparing NIC for PicoScenes with \"{prepare_channel}\"...")
        prepare_nic(nic_id, prepare_channel)

    status = get_nic_status(nic_id)
    if not args.runtime_retune:
        if channel_matches_status(status, channel_hz):
            print("NIC is already tuned by array_prepare_for_picoscenes; skipping runtime retune.")
        elif args.prepare:
            raise RuntimeError(
                "array_prepare_for_picoscenes completed, but array_status does not show "
                f"channel \"{channel_label}\" for nic_id={nic_id}."
            )
        else:
            raise RuntimeError(
                f"nic_id={nic_id} is not tuned to \"{channel_label}\". "
                "Run without --no-prepare, or use --runtime-retune if your NIC supports live retuning."
            )

    quiet_picoscenes = not args.picoscenes_logs
    with suppress_native_output(quiet_picoscenes):
        from PyPicoScenes.PyPicoScenes import getNic, picoscenes_start, picoscenes_stop, picoscenes_wait

    detector = CSIMotionDetector(
        calibration_frames=args.calibration_frames,
        threshold=args.threshold,
        min_frames=args.min_frames,
        hold_seconds=args.hold_seconds,
        baseline_alpha=args.baseline_alpha,
        threshold_sigma=args.threshold_sigma,
        feature_bins=args.feature_bins,
    )

    lock = threading.Lock()
    latest_result = MotionResult(
        calibrated=False,
        motion=False,
        score=None,
        threshold=None,
        calibration_count=0,
    )
    last_csi_time = None
    last_processed_csi_time: float | None = None
    valid_frames = 0
    ignored_frames = 0
    skipped_frames = 0
    nic = None
    ping_proc = None
    ping_exit_reported = False
    platform_started = False
    min_process_interval = 1.0 / args.max_csi_rate if args.max_csi_rate > 0 else 0.0
    status_log = None

    if args.status_log:
        status_log_path = os.path.abspath(os.path.expanduser(args.status_log))
        os.makedirs(os.path.dirname(status_log_path), exist_ok=True)
        status_log = open(status_log_path, "a", encoding="utf-8", buffering=1)
        print(f"Writing motion status lines to {status_log_path}", flush=True)

    def handle_frame(frame):
        nonlocal ignored_frames, last_csi_time, last_processed_csi_time, latest_result, skipped_frames, valid_frames
        now = time.monotonic()
        csi_segment = get_csi_segment(frame)
        if not csi_segment:
            with lock:
                ignored_frames += 1
            return True

        with lock:
            if (
                min_process_interval > 0.0
                and last_processed_csi_time is not None
                and now - last_processed_csi_time < min_process_interval
            ):
                skipped_frames += 1
                return True
            last_processed_csi_time = now

        vector = csi_magnitude_vector_from_segment(csi_segment)
        with lock:
            if vector is None:
                ignored_frames += 1
                return True
            valid_frames += 1
            result = detector.process_vector(vector, now=now)
            latest_result = result
            last_csi_time = now
            if result.ignored_reason:
                ignored_frames += 1

        return True

    try:
        with suppress_native_output(quiet_picoscenes):
            picoscenes_start()
        platform_started = True
        with suppress_native_output(quiet_picoscenes):
            nic = getNic(str(nic_id))

        if args.runtime_retune:
            with suppress_native_output(quiet_picoscenes):
                status = nic.getFrontEnd().setChannelAndBandwidth(control_hz, bandwidth_hz, center_hz)
            if int(status) != 0:
                raise RuntimeError(f"setChannelAndBandwidth failed with status {status}")

        with suppress_native_output(quiet_picoscenes):
            nic.startRxService()
            nic.registerGeneralHandler(DEFAULT_HANDLER_NAME, handle_frame)
        print("Listening for CSI frames. Press Ctrl+C to stop.")
        if ping_target_info:
            ping_target, detected_interface = ping_target_info
            ping_interface = args.ping_interface or detected_interface
            ping_proc = start_ping_helper(ping_target, args.ping_interval, ping_interface)
            interface_message = f" via {ping_interface}" if ping_interface else ""
            print(
                f"Pinging {ping_target}{interface_message} every {args.ping_interval:g}s "
                "to encourage RX packets.",
                flush=True,
            )

        while True:
            time.sleep(args.print_every)
            now = time.monotonic()
            if ping_proc is not None and ping_proc.poll() is not None and not ping_exit_reported:
                print(
                    "Ping helper exited; if valid_frames stays low, generate Wi-Fi traffic from another device.",
                    flush=True,
                )
                ping_exit_reported = True

            with lock:
                csi_age = None if last_csi_time is None else now - last_csi_time
                display_result = MotionResult(
                    calibrated=latest_result.calibrated,
                    motion=detector.current_motion(now) if latest_result.calibrated else False,
                    score=latest_result.score,
                    threshold=latest_result.threshold,
                    calibration_count=latest_result.calibration_count,
                    ignored_reason=latest_result.ignored_reason,
                )
                status_line = format_result(display_result, valid_frames, ignored_frames, skipped_frames, csi_age)
                print(status_line, flush=True)
                if status_log is not None:
                    print(status_line, file=status_log, flush=True)
    except KeyboardInterrupt:
        print("\nStopping CSI motion detector...")
        return 0
    finally:
        if status_log is not None:
            status_log.close()
        stop_process(ping_proc)
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
    sample_iw = """
phy#0
    Interface wlo1
        channel 56 (5280 MHz), width: 80 MHz, center1: 5290 MHz
"""
    assert parse_iw_dev_channel(sample_iw, "phy0") == (5280e6, 80e6, 5290e6)

    rng = np.random.default_rng(7)
    detector = CSIMotionDetector(
        calibration_frames=20,
        min_frames=3,
        hold_seconds=0.6,
        feature_bins=128,
    )
    x = np.linspace(0.0, 2.0 * np.pi, 256)
    base = np.exp(0.18 * np.sin(x) + 0.08 * np.cos(3.0 * x))
    now = 0.0

    result = None
    for _ in range(20):
        result = detector.process_vector(base * np.exp(rng.normal(0.0, 0.006, base.size)), now=now)
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
    static_changed = base * np.exp(0.40 * np.sin(2.3 * x + 2.2) + 0.15 * np.cos(5.1 * x))
    final = None
    for _ in range(5):
        final = detector.process_vector(static_changed * np.exp(rng.normal(0.0, 0.006, base.size)), now=now)
        now += 0.2

    assert final is not None and not final.motion
    print("self-test ok: False -> True -> False")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple RX-only CSI motion detector for PicoScenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phy", default=DEFAULT_PHY, help="Linux PHY name to resolve via array_status.")
    parser.add_argument("--nic-id", default=None, help="PicoScenes PhyPath ID override.")
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        help='Channel tuple: "control bandwidth center", or "auto" to use the current iw dev channel.',
    )
    parser.add_argument(
        "--prepare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run array_prepare_for_picoscenes.",
    )
    parser.add_argument("--calibration-frames", type=int, default=80, help="Valid CSI frames used to learn no-motion noise.")
    parser.add_argument("--threshold", type=float, default=None, help="Manual motion threshold. Auto threshold is used when omitted.")
    parser.add_argument("--threshold-sigma", type=float, default=6.0, help="MAD multiplier for auto threshold.")
    parser.add_argument("--min-frames", type=int, default=3, help="Consecutive threshold hits required for motion=True.")
    parser.add_argument("--hold-seconds", type=float, default=1.0, help="Keep motion=True for this long after the last confirmed hit.")
    parser.add_argument("--feature-bins", type=int, default=128, help="Resampled CSI feature length.")
    parser.add_argument("--baseline-alpha", type=float, default=0.02, help="Deprecated compatibility option; frame-delta mode does not use a baseline.")
    parser.add_argument("--print-every", type=float, default=0.2, help="Seconds between status lines.")
    parser.add_argument(
        "--status-log",
        default=DEFAULT_STATUS_LOG,
        help="Tail-friendly log file for motion status lines.",
    )
    parser.add_argument(
        "--no-status-log",
        dest="status_log",
        action="store_const",
        const=None,
        help="Disable the separate motion status log file.",
    )
    parser.add_argument(
        "--max-csi-rate",
        type=float,
        default=30.0,
        help="Maximum CSI frames per second to fully process. Use 0 to process every CSI frame.",
    )
    parser.add_argument("--ping-target", default="auto", help='Target to ping for extra traffic. Use "auto" for the default gateway.')
    parser.add_argument("--no-ping", dest="ping_target", action="store_const", const=None, help="Disable the background ping helper.")
    parser.add_argument("--ping-interval", type=float, default=0.2, help="Seconds between ping packets.")
    parser.add_argument("--ping-interface", default=None, help="Optional interface for ping -I. Defaults to the gateway route interface.")
    parser.add_argument("--runtime-retune", action="store_true", help="Call setChannelAndBandwidth after PicoScenes starts.")
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
