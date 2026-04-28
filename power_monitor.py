"""
power_monitor.py
Background-thread power sampler with auto-detecting backends.

Backends tried in order:
  1. pynvml      — NVIDIA GPU via NVML (no subprocess, fast)
  2. nvidia-smi  — NVIDIA GPU via subprocess (fallback)
  3. jetson      — Jetson Orin/Xavier/Nano INA3221 sysfs (board-level)
  4. macos       — Apple Silicon/Intel Mac via ioreg AppleSmartBattery
                   NOTE: only produces readings when running on battery
                   (unplugged). Returns None samples when plugged in.
  5. None        — unsupported platform

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


class PowerMonitor:
    def __init__(self, interval_ms: int | None = None):
        self._reader, self.backend = self._detect()
        if interval_ms is None:
            # subprocess-based backends are slower to poll
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
        return self._reader is not None

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------
    def _detect(self):
        # pynvml — fast NVIDIA path (comes with nvidia-ml-py3 or torch)
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)  # smoke test
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

        # Jetson INA3221 sysfs (board-level power)
        paths = sorted(glob.glob(
            "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power*_input"
        ))
        if paths:
            self._jetson_paths = paths
            return self._read_jetson, f"jetson-ina3221 ({len(paths)} ch)"

        # macOS — AppleSmartBattery via ioreg (no sudo, battery mode only)
        import sys
        if sys.platform == "darwin":
            try:
                import subprocess
                out = subprocess.run(
                    ["ioreg", "-rn", "AppleSmartBattery", "-d", "1"],
                    capture_output=True, text=True, timeout=3,
                ).stdout
                if '"Voltage"' in out and '"InstantAmperage"' in out:
                    return self._read_macos_battery, "macos-ioreg (battery, on-battery only)"
            except Exception:
                pass

        return None, "unavailable"

    # ------------------------------------------------------------------
    # Readers — each returns instantaneous power in mW, or None on error
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

    def _read_macos_battery(self) -> float | None:
        import subprocess, re
        try:
            out = subprocess.run(
                ["ioreg", "-rn", "AppleSmartBattery", "-d", "1"],
                capture_output=True, text=True, timeout=2,
            ).stdout
            amps_m = re.search(r'"InstantAmperage"\s*=\s*(\d+)', out)
            volts_m = re.search(r'"Voltage"\s*=\s*(\d+)', out)
            if not (amps_m and volts_m):
                return None
            # Apple Silicon stores signed mA as unsigned 64-bit (two's complement)
            raw = int(amps_m.group(1))
            amps_ma = raw - (1 << 64) if raw >= (1 << 63) else raw
            volts_mv = int(volts_m.group(1))
            # Negative = discharging (drawing from battery)
            if amps_ma >= 0:
                return None  # charging or idle on AC — reading not meaningful
            power_mw = abs(amps_ma) * volts_mv / 1000.0  # mA × mV → mW
            # Sanity check: MacBook Air M1 TDP is ~10–30W
            if 500 < power_mw < 60_000:
                return power_mw
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        self._samples.clear()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self, duration_ms: float | None = None) -> dict | None:
        """
        Stop polling. Returns stats dict or None if no samples collected.
        Pass duration_ms to get an energy estimate (avg_mW × duration).
        """
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
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

    def _poll(self):
        while not self._stop_evt.wait(self._interval):
            val = self._reader()
            if val is not None:
                self._samples.append(val)
