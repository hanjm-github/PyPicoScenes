import os
import subprocess
import sys
import sysconfig
import warnings

_CPPYY_REAL_API_INCLUDE_DIR = None


def _configure_cppyy_api_path():
    global _CPPYY_REAL_API_INCLUDE_DIR

    candidate_dirs = [
        sysconfig.get_path("include"),
        os.path.join(
            sys.prefix,
            "include",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        ),
    ]

    for include_dir in candidate_dirs:
        if not include_dir:
            continue

        upper_api = os.path.join(include_dir, "CPyCppyy", "API.h")
        lower_api = os.path.join(include_dir, "CpyCppyy", "API.h")
        if os.path.exists(upper_api):
            _CPPYY_REAL_API_INCLUDE_DIR = include_dir
            return include_dir
        if os.path.exists(lower_api):
            _CPPYY_REAL_API_INCLUDE_DIR = include_dir
            return os.path.join(os.path.dirname(__file__), "_cppyy_api_compat")

    return None


if "CPPYY_API_PATH" not in os.environ:
    _cppyy_api_path = _configure_cppyy_api_path()
    if _cppyy_api_path:
        os.environ["CPPYY_API_PATH"] = _cppyy_api_path
        warnings.filterwarnings(
            "ignore",
            message="CPyCppyy API not found.*",
            category=UserWarning,
            module="cppyy",
        )

import cppyy
import cppyy.ll


def _picoscenes_paths():
    if sys.platform.startswith("linux"):
        root = os.environ.get("PICOSCENES_ROOT", "/usr/local/PicoScenes")
        lib_dir = os.environ.get("PICOSCENES_LIB_DIR", os.path.join(root, "pslib"))
        if not os.path.isdir(lib_dir):
            lib_dir = os.path.join(root, "lib")
    elif sys.platform.startswith("win32"):
        root = os.environ.get("PICOSCENES_ROOT", "C:\\Program Files\\PicoScenes")
        lib_dir = os.environ.get("PICOSCENES_LIB_DIR", os.path.join(root, "lib"))
    else:
        raise RuntimeError("Please add PicoScenes lib/include paths here.")

    include_dir = os.environ.get("PICOSCENES_INCLUDE_DIR", os.path.join(root, "include"))
    return lib_dir, include_dir


_PICOSCENES_LIB_DIR, _PICOSCENES_INCLUDE_DIR = _picoscenes_paths()
cppyy.add_library_path(_PICOSCENES_LIB_DIR)
cppyy.add_include_path(_PICOSCENES_INCLUDE_DIR)
if _CPPYY_REAL_API_INCLUDE_DIR:
    cppyy.add_include_path(_CPPYY_REAL_API_INCLUDE_DIR)

for _library in (
    "libServer",
    "libmac80211Injection",
    "libDSP",
    "libFrontEnd",
    "libIntrinsics",
    "libLicense",
    "libNICHAL",
    "librxs_parsing",
    "libSDRBaseband",
    "libSodiumWrapper",
    "libSystemTools",
):
    cppyy.load_library(_library)

for _header in ("cstdint", "exception", "queue", "atomic", "condition_variable"):
    cppyy.include(_header)

# Conda's Boost.Asio headers default to header-only mode, which makes Cling
# parse implementation details that it cannot legally access. PicoScenes ships
# the compiled Boost symbols, so keep Asio declarations separate for cppyy.
cppyy.cppdef("""
#ifndef BOOST_ASIO_SEPARATE_COMPILATION
#define BOOST_ASIO_SEPARATE_COMPILATION 1
#endif
""")

# CSI-file parsing only needs this frame definition. Keep it independent from
# the full runtime headers so parse_frame.py works even when Boost headers from
# the active Python environment are incompatible with cppyy/Cling.
cppyy.include("PicoScenes/ModularPicoScenesFrame.hxx")

std = cppyy.gbl.std
uint16_t = cppyy.gbl.uint16_t
ModularPicoScenesRxFrame = cppyy.gbl.ModularPicoScenesRxFrame


def _export_if_available(name):
    try:
        globals()[name] = getattr(cppyy.gbl, name)
    except AttributeError:
        pass


for _name in (
    "ChannelBandwidthEnum",
    "PacketFormatEnum",
    "GuardIntervalEnum",
    "PicoScenesFrameTxParameters",
    "MagicIntel123456",
    "TxPrecodingParameters",
    "ExtraInfoSegment",
    "PicoScenesDeviceType",
    "ChannelCodingEnum",
    "PayloadDataType",
    "PayloadSegment",
    "isSDR",
):
    _export_if_available(_name)


_FULL_BINDINGS_ERROR = None
_FULL_BINDINGS_LOADED = False


def _load_full_bindings():
    cppyy.include("PicoScenes/PyPicoScenes.hxx")

    required_headers = (
        "PicoScenes/SDRExtraSegment.hxx",
        "PicoScenes/PicoScenesFrameTxParameters.hxx",
        "PicoScenes/MVMExtraSegment.hxx",
        "PicoScenes/UDPService.hxx",
        "PicoScenes/LicenseModel.hxx",
        "PicoScenes/LoggingService.hxx",
        "PicoScenes/PayloadSegment.hxx",
        "PicoScenes/AbstractPicoScenesFrameSegment.hxx",
        "PicoScenes/LicenseService.hxx",
        "PicoScenes/RxSBasicSegment.hxx",
        "PicoScenes/IntelRateNFlag.hxx",
        "PicoScenes/ExtraInfoSegment.hxx",
        "PicoScenes/SignalMatrix.hxx",
        "PicoScenes/PicoScenesCommons.hxx",
        "PicoScenes/FrontEndModePreset.hxx",
        "PicoScenes/FrameDumper.hxx",
        "PicoScenes/TaggedThreadPool.hxx",
        "PicoScenes/SDRFrontEndConfigurations.hxx",
        "PicoScenes/CSISegment.hxx",
        "PicoScenes/BasebandSignalSegment.hxx",
        "PicoScenes/FrontEndConfigurations.hxx",
        "PicoScenes/Singleton.hxx",
        "PicoScenes/CargoSegment.hxx",
        "PicoScenes/DynamicFieldInterpretation.hxx",
        "PicoScenes/FIFOWaitBlocker.hxx",
        "PicoScenes/SDRHardwareInformation.hxx",
        "PicoScenes/Intrinsics.hxx",
        "PicoScenes/RXSExtraInfo.hxx",
        "PicoScenes/BBSignalsFileWriter.hxx",
        "PicoScenes/DSPRateTracker.hxx",
        "PicoScenes/AbstractSDRFrontEnd.hxx",
        "PicoScenes/MAC80211CSIExtractableFrontEnd.hxx",
    )
    optional_headers = (
        "PicoScenes/CSILivePlotter.hxx",
        "PicoScenes/SoapySDRUtils.hxx",
    )

    for header in required_headers:
        cppyy.include(header)

    for header in optional_headers:
        try:
            cppyy.include(header)
        except ImportError:
            pass

    for name in (
        "ChannelBandwidthEnum",
        "LoggingService",
        "AbstractSDRFrontEnd",
        "FrontEndModePreset",
        "PacketFormatEnum",
        "GuardIntervalEnum",
        "PicoScenesFrameTxParameters",
        "MagicIntel123456",
        "TxPrecodingParameters",
        "ExtraInfoSegment",
        "PicoScenesDeviceType",
        "isIntelMVMTypeNIC",
        "ChannelCodingEnum",
        "PayloadDataType",
        "PayloadSegment",
        "isSDR",
        "MAC80211CSIExtractableFrontEnd",
        "PicoScenesStart",
        "PicoScenesWait",
        "PicoScenesStop",
        "getNIC",
    ):
        globals()[name] = getattr(cppyy.gbl, name)

    _export_if_available("CSILivePlotter")

    cppyy.cppdef("""
#include <cstdint>
#include <optional>
#include <PicoScenes/ModularPicoScenesFrame.hxx>
#include <PicoScenes/PicoScenesCommons.hxx>
#include <PicoScenes/AbstractNIC.hxx>
#include <PicoScenes/PicoScenesFrameTxParameters.hxx>


enum class EchoProbeWorkingMode : uint8_t {
    Standby = 14,
    Injector,
    Logger,
    EchoProbeInitiator,
    EchoProbeResponder,
    Radar
};

enum class EchoProbeInjectionContent: uint8_t {
    NDP = 20,
    Header,
    Full,
};

enum class EchoProbePacketFrameType : uint8_t {
    SimpleInjectionFrameType = 10,
    EchoProbeRequestFrameType,
    EchoProbeReplyFrameType,
    EchoProbeFreqChangeRequestFrameType,
    EchoProbeFreqChangeACKFrameType,
};

enum class EchoProbeReplyStrategy : uint8_t {
    ReplyOnlyHeader = 0,
    ReplyWithExtraInfo,
    ReplyWithCSI,
    ReplyWithFullPayload,
};

class EchoProbeParameters {
public:
    EchoProbeWorkingMode workingMode = EchoProbeWorkingMode::Standby;
    std::optional<std::array<uint8_t, 6>> inj_target_mac_address;
    std::optional<bool> inj_for_intel5300;
    uint32_t tx_delay_us{500000};
    std::optional<uint32_t> delayed_start_seconds;
    bool useBatchAPI{false};
    uint32_t batchLength;

    std::optional<std::string> outputFileName;
    bool randomMAC;
    EchoProbeInjectionContent injectorContent{EchoProbeInjectionContent::Full};
    std::optional<uint32_t> randomPayloadLength;

    std::optional<double> cf_begin;
    std::optional<double> cf_end;
    std::optional<double> cf_step;
    std::optional<uint32_t> cf_repeat;
    std::optional<uint32_t> round_repeat;

    std::optional<double> sf_begin;
    std::optional<double> sf_end;
    std::optional<double> sf_step;

    uint32_t tx_max_retry{100};
    EchoProbeReplyStrategy replyStrategy{EchoProbeReplyStrategy::ReplyWithFullPayload};

    std::optional<PacketFormatEnum> ack_format;
    std::optional<uint32_t> ack_cbw;
    std::optional<uint32_t> ack_mcs;
    std::optional<uint32_t> ack_numSTS;
    std::optional<uint32_t> ack_guardInterval;

    std::optional<uint32_t> timeout_ms{150};
    std::optional<uint32_t> delay_after_cf_change_ms{5};
    std::optional<uint32_t> numOfPacketsPerDotDisplay{10};
};

void setTxParameters(AbstractNIC* nic, PicoScenesFrameTxParameters parameters){
    nic->getUserSpecifiedTxParameters() = parameters;
}
""")

    for name in (
        "EchoProbeWorkingMode",
        "EchoProbeInjectionContent",
        "EchoProbePacketFrameType",
        "EchoProbeReplyStrategy",
        "EchoProbeParameters",
        "setTxParameters",
    ):
        globals()[name] = getattr(cppyy.gbl, name)


def _require_full_bindings():
    global _FULL_BINDINGS_ERROR, _FULL_BINDINGS_LOADED
    if _FULL_BINDINGS_LOADED:
        return

    try:
        _load_full_bindings()
        _FULL_BINDINGS_LOADED = True
        _FULL_BINDINGS_ERROR = None
        return
    except Exception as exc:
        _FULL_BINDINGS_ERROR = exc

    if _FULL_BINDINGS_ERROR is not None:
        raise RuntimeError(
            "Full PicoScenes runtime bindings could not be loaded. "
            "CSI file parsing is available, but NIC/USRP runtime APIs require "
            "PicoScenes headers and Boost headers that cppyy can parse. "
            f"Original error: {_FULL_BINDINGS_ERROR}"
        ) from _FULL_BINDINGS_ERROR


def picoscenes_start(commandString: str = None):
    _require_full_bindings()
    proc = subprocess.Popen(
        ["PicoScenes", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    PicoScenesStart()


def picoscenes_wait():
    _require_full_bindings()
    PicoScenesWait()


def picoscenes_stop():
    _require_full_bindings()
    PicoScenesStop()


def getNic(nicName):
    _require_full_bindings()
    return getNIC(nicName)


if os.environ.get("PYPICOSCENES_CORE_ONLY", "").lower() not in ("1", "true", "yes"):
    _require_full_bindings()


if __name__ == "__main__":
    pass
