"""
render.py
Applies a Room Impulse Response to a dry audio signal via convolution,
producing a reverberant output wav file.
"""

import numpy as np
import scipy.signal
import soundfile as sf


def apply_rir(input_path, rir, output_path):
    """
    Convolve a dry audio file with a RIR and write the result.

    Args:
        input_path:  path to dry mono .wav file
        rir:         np.ndarray — room impulse response from acoustics.py
        output_path: path to write reverberant .wav file
    """
    audio, fs = sf.read(input_path)

    # If stereo, mix down to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    reverbed = scipy.signal.fftconvolve(audio, rir)

    # Normalize to prevent clipping
    reverbed = reverbed / np.max(np.abs(reverbed))

    sf.write(output_path, reverbed, fs)
    return reverbed, fs


if __name__ == "__main__":
    import time
    from acoustics import compute_rir

    print("=== render.py sanity check ===\n")

    dims = [6, 4, 3]
    materials = {
        "walls":   "brickwork",
        "floor":   "carpet_cotton",
        "ceiling": "ceiling_fibre_absorber",
    }

    print("Computing RIR...")
    rir, t60 = compute_rir(dims, materials)
    print(f"  T60: {t60:.3f}s")

    print("Applying RIR to audio...")
    t_start = time.perf_counter()
    reverbed, fs = apply_rir("audio/test.wav", rir, "audio/output.wav")
    elapsed = (time.perf_counter() - t_start) * 1000

    print(f"  Input length:  {sf.info('audio/test.wav').duration:.2f}s")
    print(f"  Output length: {len(reverbed)/fs:.2f}s")
    print(f"  Render time:   {elapsed:.1f}ms")
    print(f"\nOutput written to audio/output.wav")
