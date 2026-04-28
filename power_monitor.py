"""
power_monitor.py
Background-thread power sampler with auto-detecting backends.

Backends tried in order:
  1. pynvml      — NVIDIA GPU via NVML (no subprocess, fast)
  2. nvidia-smi  — NVIDIA GPU via subprocess (fallback)
  3. jetson      — Jetson INA3221 sysfs (board-level, several path variants)
  4. tegrastats  — Jetson tegrastats streaming process (fallback if no sysfs)
  5. macos       — Apple Silicon/Intel Mac via ioreg AppleSmartBattery
                   NOTE: only produces readings when running on battery
                   (unplugged). Returns None samples when plugged in.
  6. None        — unsupported platform

Usage:
    mon = PowerMonitor()
    if mon.available:
        mon.start()
        # ... do work (duration_ms) ...
        stats = mon.stop(duration_ms)
        print(f"{stats['avg_mW']:.0f} mW  {stats['energy_mJ']:.1f} mJ")
"""

import glob
import threading


# Labels tegrastats uses for total board power, in preference order.
# Values may appear as "3924mW/3924mW" (Orin) or "3924/3924" (older).
_TSTAT_LABELS = [
    "VIN_SYS_5V0",      # Orin AGX — total system input power
    "VDD_IN",           # Xavier / older Orin
    "POM_5V_IN",        # Nano / TX2
    "VDD_GPU_SOC",      # fallback: GPU+SOC only
    "VDD_CPU_GPU_CV",   # fallback: CPU+GPU
]


class PowerMonitor:
    def __init__(self, interval_ms: int | None = None):
        self._tstat_proc = None   # only used for tegrastats backend
        self._reader, self.backend = self._detect()
        if interval_ms is None:
            if self.backend in ("nvidia-smi (GPU)",) or self.backend.startswith("macos"):
                interval_ms = 500
            else:
                interval_ms = 100
        self._interval = interval_ms / 1000.0
        self._samples: list[float] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self._reader is not None or self.backend.startswith("tegrastats")

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------
    def _detect(self):
        # pynvml — fast NVIDIA path
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
            return self._read_pynvml, "pynvml (GPU)"
        except Exception:
            pass

        # nvidia-smi subprocess fallback
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                float(r.stdout.strip().split("\n")[0])
                return self._read_nvidia_smi, "nvidia-smi (GPU)"
        except Exception:
            pass

        # Jetson INA3221 sysfs — several path variants across JetPack versions
        _JETSON_GLOBS = [
            "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power*_input",
            "/sys/bus/platform/drivers/ina3221/*/hwmon/hwmon*/power*_input",
            "/sys/bus/i2c/devices/*/hwmon/hwmon*/power*_input",
            "/sys/class/hwmon/hwmon*/power*_input",
        ]
        for pattern in _JETSON_GLOBS:
            paths = sorted(glob.glob(pattern))
            if paths:
                readable = []
                for p in paths:
                    try:
                        if int(open(p).read().strip()) > 0:
                            readable.append(p)
                    except OSError:
                        pass
                if readable:
                    self._jetson_paths = readable
                    return self._read_jetson, f"jetson-ina3221 ({len(readable)} ch)"

        # Jetson tegrastats — streaming subprocess (works when sysfs is absent)
        # Requires sudo on most Jetson configurations.
        try:
            import subprocess, re
            proc = subprocess.Popen(
                ["sudo", "tegrastats", "--interval", "200"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            line = proc.stdout.readline()
            proc.terminate()
            proc.wait(timeout=2)
            if line and self._parse_tegrastats(line) is not None:
                return None, "tegrastats (Jetson board)"
        except Exception:
            pass

        # macOS — AppleSmartBattery via ioreg (no sudo, battery mode only)
        import sys
        if sys.platform == "darwin":
            import subprocess
            _CANDIDATES = [
                ["/usr/sbin/ioreg", "-r", "-c", "AppleSmartBattery", "-d", "1"],
                ["ioreg", "-r", "-c", "AppleSmartBattery", "-d", "1"],
                ["/usr/sbin/ioreg", "-r", "-c", "AppleSmartBatteryManager", "-d", "2"],
                ["ioreg", "-r", "-c", "AppleSmartBatteryManager", "-d", "2"],
                ["/usr/sbin/ioreg", "-rn", "AppleSmartBattery"],
            ]
            for cmd in _CANDIDATES:
                try:
                    out = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=3,
                    ).stdout
                    if '"Voltage"' in out:
                        self._ioreg_cmd = cmd
                        return self._read_macos_battery, "macos-ioreg (battery, on-battery only)"
                except Exception:
                    continue

        return None, "unavailable"

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------
    def _read_pynvml(self) -> float | None:
        try:
            import pynvml
            return float(pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle))
        except Exception:
            return None

    def _read_nvidia_smi(self) -> float | None:
        import subprocess
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                return float(r.stdout.strip().split("\n")[0]) * 1000.0  # W → mW
        except Exception:
            pass
        return None

    def _read_jetson(self) -> float | None:
        total_uw = 0
        valid = 0
        for p in self._jetson_paths:
            try:
                total_uw += int(open(p).read().strip())
                valid += 1
            except OSError:
                pass
        return total_uw / 1000.0 if valid else None  # µW → mW

    def _read_macos_battery(self) -> float | None:
        import subprocess, re
        try:
            out = subprocess.run(
                self._ioreg_cmd, capture_output=True, text=True, timeout=2,
            ).stdout
            amps_m = (re.search(r'"InstantAmperage"\s*=\s*(\d+)', out)
                      or re.search(r'"Amperage"\s*=\s*(-?\d+)', out)
                      or re.search(r'"Current"\s*=\s*(-?\d+)', out))
            volts_m = re.search(r'"Voltage"\s*=\s*(\d+)', out)
            if not (amps_m and volts_m):
                return None
            raw = int(amps_m.group(1))
            amps_ma = raw - (1 << 64) if raw >= (1 << 63) else raw
            volts_mv = int(volts_m.group(1))
            if amps_ma >= 0:
                return None  # charging or on AC
            power_mw = abs(amps_ma) * volts_mv / 1000.0
            if 500 < power_mw < 60_000:
                return power_mw
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_tegrastats(line: str) -> float | None:
        import re
        for label in _TSTAT_LABELS:
            # Matches both "3924/3924" (older) and "3924mW/3924mW" (Orin)
            m = re.search(rf'{label}\s+(\d+)(?:mW)?/\d+', line)
            if m:
                return float(m.group(1))  # already in mW
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        if not self.available:
            return
        self._samples.clear()
        self._stop_evt.clear()
        if self.backend.startswith("tegrastats"):
            import subprocess
            interval_ms = max(100, int(self._interval * 1000))
            self._tstat_proc = subprocess.Popen(
                ["sudo", "tegrastats", "--interval", str(interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self._thread = threading.Thread(target=self._poll_tegrastats, daemon=True)
        else:
            self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self, duration_ms: float | None = None) -> dict | None:
        """Stop polling. Returns stats dict or None if no samples collected."""
        self._stop_evt.set()
        if self._tstat_proc is not None:
            try:
                self._tstat_proc.terminate()
                self._tstat_proc.wait(timeout=2)
            except Exception:
                pass
            self._tstat_proc = None
        if self._thread:
            self._thread.join(timeout=3)
        if not self._samples:
            return None
        avg = sum(self._samples) / len(self._samples)
        stats = {
            "avg_mW":    round(avg, 1),
            "peak_mW":   round(max(self._samples), 1),
            "n_samples": len(self._samples),
            "backend":   self.backend,
        }
        if duration_ms is not None:
            stats["energy_mJ"] = round(avg * duration_ms / 1000.0, 1)
        return stats

    # ------------------------------------------------------------------
    # Poll loops
    # ------------------------------------------------------------------
    def _poll(self):
        while True:
            val = self._reader()
            if val is not None:
                self._samples.append(val)
            if self._stop_evt.wait(self._interval):
                break

    def _poll_tegrastats(self):
        """Read power from a live tegrastats subprocess line by line."""
        try:
            while not self._stop_evt.is_set():
                line = self._tstat_proc.stdout.readline()
                if not line:
                    break
                val = self._parse_tegrastats(line)
                if val is not None:
                    self._samples.append(val)
        except Exception:
            pass
