# Offloaded Spatial Audio Rendering for XR

CS 534 IC research project (Abhi Ramakrishnan, Sethu Eapen) studying whether
scene-aware spatial audio for XR headsets can be partially offloaded to a
nearby server without breaking latency or power budgets.

## The problem

Spatial audio needs to match the room a user is standing in, or the mismatch
is immediately noticeable. Doing this properly means computing a Room Impulse
Response (RIR) from the room's actual geometry and surface materials, and
recomputing it as the user moves. SAMOSA (Xu et al., UIST 2025) does this
fully on-device on a Snapdragon XR2+ Gen 2 chip, hitting 58ms per RIR update,
but that number is for one chip, and on-device inference has a real power
cost on a headset.

The obvious alternative is offloading the compute to a nearby server. That
works for some XR workloads, but audio has a tighter constraint than most:
Yeregui et al. (2024) measured 185ms of latency when fully offloading an XR
rendering pipeline over Ethernet, above the roughly 160ms threshold where
audio and video noticeably fall out of sync. Full offload is probably a
non-starter for audio. This project measures whether a partial split, where
only the heavy ML inference leaves the device, can stay inside that budget
while still saving power.

## Pipeline

The pipeline mirrors SAMOSA's architecture: three perception stages that can
run in parallel, then two sequential stages to actually produce sound.

1. **Shoebox estimation** (`shoebox.py`) turns room geometry into a
   width x length x height box. Two input paths exist: a ScanNet scene
   directory (depth frames + camera poses + calibration.txt), which is
   back-projected into a world-space point cloud per frame and reduced to an
   axis-aligned bounding box, or a raw PLY mesh for synthetic test scenes
   built in Blender. Floor and ceiling are separated by Z-percentile
   clustering since depth cameras rarely look straight up.
2. **Material segmentation** (`segmentation.py`) classifies wall, floor,
   and ceiling materials from an RGB frame using SegFormer-b0 pretrained on
   ADE20K, run through a fixed table mapping ADE20K classes onto
   pyroomacoustics material names. Produces both a single best-guess
   material per surface and a full per-surface confidence distribution
   (e.g. floor is 60% wood, 40% carpet), which acoustics.py later blends
   into a weighted absorption coefficient rather than snapping to one
   material.
3. **Scene classification** (`scene_classifier.py`) is a rule-based
   classifier (bedroom / living room / office / conference / outdoor) driven
   by room volume and a softness score computed from the material mix. Each
   scene type maps to a preset reflection order and target RT60. This stage
   is a stand-in for a Places365-style learned classifier, kept rule-based
   for now since it isn't the bottleneck.
4. **RIR synthesis** (`acoustics.py`) feeds geometry, materials, and the
   scene preset into pyroomacoustics' image-source method (ShoeBox model) to
   compute the actual impulse response and its T60.
5. **Audio rendering** (`render.py`) convolves the dry input audio with the
   RIR (FFT convolution via scipy) to produce the reverberant output.

Stages 1 and 2 run concurrently via `ThreadPoolExecutor` since they're
independent; stage 3 needs both of their outputs; stages 4 and 5 are strictly
sequential. `profiler.py` wraps every stage in a timer so every run produces
a per-stage latency breakdown, and `power_monitor.py` runs alongside on a
background thread sampling power draw (NVML/nvidia-smi on NVIDIA GPUs,
INA3221 sysfs or tegrastats on Jetson boards, powermetrics/ioreg on macOS),
so every run also gets an average/peak power and energy figure.

There's a second segmentation mode, `--samosa-mode`, that swaps SegFormer for
a DeepLabv3+/MobileNetV3 forward pass. MobileNetV3 is trained on PASCAL VOC
and its output isn't usable for material classification, so in this mode the
inference only exists to produce a realistic timing number for a
SAMOSA-class lightweight model on our hardware; the actual materials come
from cached SegFormer results (`--init-segmentation` generates the cache
once ahead of time). Audio quality numbers are only meaningful in normal
mode, since SAMOSA mode intentionally throws away its segmentation output.

## Offloading configurations

The headset (a MacBook, standing in for XR headset compute) and the server
(a Jetson Orin AGX) talk over a raw TCP socket (`net_utils.py` handles
message framing and numpy array serialization). `server.py` accepts a
contiguous range of pipeline stages, runs them, and returns the outputs plus
its own timing and power numbers. Binaural rendering (stage 5, in the actual
XR deployment) always stays on-device, since it needs continuous access to
head pose and can't tolerate round-trip latency.

| Config | Runs on device | Runs on server | Why |
|---|---|---|---|
| A (baseline) | geometry + materials + RIR | nothing | reproduces the SAMOSA on-device numbers |
| B (partial) | geometry + RIR | materials | offload just the expensive ML inference |
| C (heavy) | geometry | materials + RIR | offload everything except geometry |
| D (full) | nothing | geometry + materials + RIR | full offload, expected to blow the latency budget |

Config D can only run against a PLY mesh, not a ScanNet scene, since a
ScanNet scene is a whole directory of depth frames and poses, not something
you can hand to a server call.

`pipeline.py` implements both the local path (stages run in-process) and the
offload path (stages before the split run locally, the split range gets
shipped to the server, remaining stages run locally on the response) behind
the same `run()` function, selected by whether `--server`/`--offload` are
passed.

## Experiments

`run_experiments.py` sweeps every scene (both the hand-built PLY+Blender test
rooms in `testing/` and the ScanNet scenes in `scannet/`) across every config
and both segmentation modes, with warmup runs discarded, and writes one row
per run to `results/metrics.csv`: per-stage latency, round-trip and network
overhead for offloaded stages, client and server power/energy, and T60. The
ScanNet scenes only run configs A-C, since config D needs a transferable PLY.
`results/` holds the rendered output audio for every scene/config/mode/run
combination, plus per-scene segmentation caches and several metrics CSVs from
different hardware/network conditions (wired vs wireless MacBook, Jetson,
with and without background load).

The eventual comparison point is a reference RIR computed by running
pyroomacoustics directly on a ScanNet scene's ground-truth mesh, so pipeline
output can be scored against it on T60 and normalized echo density rather
than just eyeballed.

## Layout

```
shoebox.py, segmentation.py, scene_classifier.py, acoustics.py, render.py
    the five pipeline stages, each usable standalone
pipeline.py       orchestrates local and offloaded runs
server.py         offload target, runs a stage range on request
net_utils.py       socket framing + numpy (de)serialization for client/server messages
profiler.py, power_monitor.py
    cross-cutting timing and power instrumentation
run_experiments.py
    sweeps scenes x configs x modes, writes results/metrics.csv
testing/          synthetic PLY + rendered image pairs (Blender)
scannet/           ScanNet scene directories (depth, RGB, poses, calibration)
results/           output audio, segmentation caches, metrics CSVs
progress_report_*.tex
    write-up of motivation, related work, and results
```

## Running it

```
pip install -r requirements.txt

# local, all on-device (Config A)
python pipeline.py --scene scannet/scene0005 --input audio/test.wav --output audio/out.wav

# offload materials + RIR to a server (Config C)
python server.py --port 9000 --device cuda   # on the server
python pipeline.py --scene scannet/scene0005 --server 192.168.1.100:9000 --offload 2 4

# SAMOSA emulation mode (needs a cache first)
python pipeline.py --scene scannet/scene0005 --init-segmentation
python pipeline.py --scene scannet/scene0005 --samosa-mode

# full sweep across scenes/configs/modes
python run_experiments.py --server 192.168.1.100:9000 --scannet-dir scannet
```
