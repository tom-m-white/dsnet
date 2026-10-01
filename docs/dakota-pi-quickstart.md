# Dakota's Raspberry Pi benchmark guide

Prepared September 30, 2026. You do not need programming experience.

**Goal:** run two programs on the Raspberry Pi and send me (tom) the resulting files.
The first program checks that the setup works. The second measures the trained
V5 model's speed and memory use. No training or video dataset is needed.

## 1. Your setup: ROSOrin Starter Edition + Windows laptop

You already have the robot's connection sheet and have connected before. Keep
using that working setup. No new laptop-to-robot cable is needed for this guide.
Power the assembled robot using its existing setup and follow your sheet to
reconnect. Do not unplug its internal cables, reimage storage, or reinstall ROS.

**First distinguish a remote robot desktop from a laptop virtual machine (VM).**
A remote desktop shows the robot's actual computer. A VM is another operating
system running on your laptop. They can look almost identical, but a benchmark
run directly in the laptop VM measures the laptop, not the Pi.

## 2. Open a terminal on the ACTUAL robot

1. Follow the connection sheet exactly as you have before: power the robot,
   join the specified network, and open the connection application.
2. If the application shows a Linux desktop, click inside that desktop and
   press **Ctrl + Alt + T**. Alternatively, find **Terminal** in its applications
   menu. A terminal is a window for typing commands.
3. If your sheet already connects you through SSH, use that connected terminal.
4. Run these two commands, one at a time:

```bash
uname -m
cat /proc/device-tree/model
```

Read the output before proceeding:

| What it prints | What it means / next step |
|---|---|
| `aarch64` and a model containing `Raspberry Pi 5` | You reached the Pi. This is the PI TERMINAL. Continue to Section 3. |
| `x86_64`, or a missing model file | You have not verified the robot. A laptop VM commonly gives these results. Follow the SSH step below; do not benchmark here. |
| A model containing `Jetson` or `NVIDIA` | This is a Jetson version. Send me the exact output and stop before the Pi installation steps; it needs a JetPack-compatible setup. |
| Anything else | Send me both outputs before proceeding. |

### If that window is the laptop VM: connect from it to the robot

Keep the VM running. In its terminal, type the following, replacing BOTH
capitalized words with the ROBOT login username and ROBOT address from the sheet:

```text
ssh USERNAME@ROBOT_ADDRESS
```

Use the robot credentials, which may differ from the VM login. On a first
connection, confirm the target is your robot before accepting the host prompt
with `yes`. Enter the robot login password when asked. **No characters or dots
appear while typing a Linux password; this is normal.** Press Enter.

Now repeat:

```bash
uname -m
cat /proc/device-tree/model
```

Continue only when the model confirms **Raspberry Pi 5**. Leave this connected
terminal open. All benchmark/install commands must run here. If the connection
fails, send me the exact error; do not guess IP addresses or reset the network.

### Record the robot username and address for file transfer

In the verified PI TERMINAL, run:

```bash
whoami
hostname -I
```

Keep the username and the reachable robot address from your sheet handy.
The later file-transfer instructions use Windows PowerShell as a second window.
If Windows cannot reach the robot but the VM can, use the VM transfer option in
Section 5. [SSH reference](https://www.raspberrypi.com/documentation/computers/remote-access.html)

## 3. How to enter commands

For every command box below:

1. Copy the text INSIDE the box, not the surrounding explanation or backticks.
2. Paste into the PI terminal. On the Pi desktop, use **Ctrl + Shift + V**.
   In Windows PowerShell, use **Ctrl + V**; in Mac Terminal, use **Command + V**.
3. Press Enter. For a box with several commands, run them one line at a time.
4. Wait until the prompt returns before continuing. Downloads may take minutes.
5. If a command reports an error, stop there and send me the command and error.

`sudo` may request the Pi login password. Again, invisible password typing is normal.
Keep capitalization and punctuation exactly as shown.

## 4. Check that the Pi can run this software

Run ON THE PI:

```bash
uname -m
getconf LONG_BIT
python3 --version
```

Expected: `aarch64`, then `64`, then a Python version. Send me all three lines
if the first two differ. The PyTorch Pi installation requires a 64-bit OS.
[PyTorch's Pi guide](https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html)

## 5. Put Tom!!'s benchmark ZIP on the Pi

Ask me for **pi_benchmark_20260930.zip**. I got it in a folder called handoff_packages and can send it to you. Its not in the repo by default. It is about 3.3 MB and contains the
benchmark code and the V5 checkpoint. The checkpoints-only research Drive folder
does not contain the code needed to run this benchmark.

### Copy the ZIP from your Windows laptop to the Pi

1. Download the ZIP into your LAPTOP's **Downloads** folder. Do not extract it yet.
2. In the verified PI TERMINAL from Section 2, run:

```bash
mkdir -p ~/Downloads
```

3. On Windows, open PowerShell using Windows key -> type PowerShell -> open.
   Do not connect this second window with SSH. It is the **LAPTOP TERMINAL**.
4. Run these commands in that SECOND window, replacing `USERNAME` and `PI_ADDRESS`
   with the ROBOT username and reachable address recorded in Section 2:

```text
cd ~/Downloads
scp pi_benchmark_20260930.zip USERNAME@PI_ADDRESS:Downloads/
```

5. Enter the Pi password if requested. Wait for transfer completion (usually 100%).
6. Return to the verified PI TERMINAL from Section 2 (VNC/SSH), not an
   unconnected terminal on the laptop or VM.

**If only the VM can reach the robot:** download the ZIP inside the VM's browser
into the VM's Downloads folder. Open a NEW VM terminal (not the one connected
through SSH), and run the same `cd ~/Downloads` and `scp ...` commands above.
This transfers the ZIP from VM to robot. Keep using the original verified PI
TERMINAL for installation and benchmarking. A ZIP sitting in the VM alone is
not yet on the robot.

### Extract the ZIP on the Pi

Run ON THE PI:

```bash
sudo apt update
sudo apt install -y unzip python3-venv
unzip ~/Downloads/pi_benchmark_20260930.zip -d ~/
cd ~/pi_benchmark
ls
```

Expected files/folders: `benchmark.py`, `benchmark_config.yaml`,
`requirements-pi.txt`, `dstg`, and `models`.
If unzip asks to replace existing files, stop and ask me: you may already have
a setup or results that should be preserved.

## 6. Install the software once

Run ON THE PI:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy PyYAML psutil threadpoolctl
```

This installation needs internet access ON THE PI, not just on the laptop.
If downloads fail to connect, send me the error before changing networks.

After activation, the prompt should begin with `(.venv)`. This is a separate
software environment for the benchmark. The Torch command requests a ready-made
CPU package; it will fail rather than spend hours compiling Torch on the Pi.
If it says `No matching distribution found`, send me the error and Section 4's
outputs. Do not try random installation commands from forums.

Check the installation:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected: a Torch version and **CUDA: False**. False is correct for this Pi CPU run.

## 7. Run the small setup check

Run ON THE PI:

```bash
python benchmark.py --variant dummy --dataset summe --split 0 --seed 0
```

Expected: a block of results containing `latency_mean_seconds`,
`frames_per_second`, and `parameter_count`. The dummy parameter count is **1025**.
Its output file is:

```text
results/benchmarks/dummy_summe_0_0_cpu.json
```

This checks the measurement program. It is not the trained research model.

## 8. Install the graph package

Run ON THE PI:

```bash
python -m pip install torch_geometric
python -c "from torch_geometric.nn import GATConv, SAGEConv; print('Graph layers imported OK')"
```

Expected: **Graph layers imported OK**. Start with this minimal installation;
PyG's extra compiled extensions are optional for basic use.
[PyG installation documentation](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html)

Do not start building extra graph packages from source to fix a warning. Send
me any error. If installation is blocked for an hour, tell Dr. Wang as requested.

## 9. Record the hardware and run V5

IMPORTANT INFO DAKOTA: (this will potentially mess up test results)
Keep the robot stationary with its existing cooling running. Use the same power
source throughout; do not run updates or other benchmarks at the same time. Do not kill robot
services yourself. If ROS remains active, report that: it can affect timings.

Run ON THE PI:

```bash
mkdir -p results/benchmarks
cat /proc/device-tree/model > results/benchmarks/pi-model.txt
free -h > results/benchmarks/pi-memory.txt
cat /etc/os-release > results/benchmarks/pi-os.txt
uname -a > results/benchmarks/pi-kernel.txt
python -m pip freeze > results/benchmarks/pi-packages.txt
vcgencmd measure_temp > results/benchmarks/temperature-before.txt
vcgencmd get_throttled > results/benchmarks/throttling-before.txt
python benchmark.py --variant V5 --dataset summe --split 0 --seed 0
vcgencmd measure_temp > results/benchmarks/temperature-after.txt
vcgencmd get_throttled > results/benchmarks/throttling-after.txt
```

The commands using `>` save information to files and may print nothing. That is
normal. If `vcgencmd` is unavailable on this OS, tell me; the benchmark itself
can still run, but those hardware checks remain unrecorded.

The V5 command may take longer than the dummy. Wait for the prompt. It measures
1,000 synthetic feature rows, using four CPU threads, ten inference passes with
the first two as warmup, plus a separate memory-measurement pass. These settings
are already in the bundle. Do not change them to get a faster number.

Expected output file:

```text
results/benchmarks/V5_summe_0_0_cpu.json
```

For this bundled model, `parameter_count` should be **888577**, `device` should
be **cpu**, `measured_repeats` should be **8**, and `discarded_warmup` should be **2**.
I should check these before treating the result as a valid comparison.

## 10. Send me (TOM!) the results

Run ON THE PI:

```bash
tar -czf ~/pi-benchmark-results.tar.gz -C ~/pi_benchmark results/benchmarks
```

In Windows PowerShell (the LAPTOP TERMINAL), run the following,
replacing the same username and address as before:

```text
cd ~/Downloads
scp USERNAME@PI_ADDRESS:pi-benchmark-results.tar.gz .
```

The final dot is required; it means save into the current folder. Then send the
file from your laptop's Downloads folder to me. If you used the VM for file
transfer, run this command in a NEW, unconnected VM terminal instead; the file
will be in the VM's Downloads folder, and you can send it from the VM's browser.

In the same message, include:

- Pi model and advertised RAM capacity from its box or lab inventory.
- Power supply brand/rating and whether a fan or heatsink was installed.
- Whether the Pi had other programs running, and any warnings/errors.
- Date/time and whether a USB power meter was attached.

The memory command reports usable system memory; it is not a substitute for
recording the advertised RAM configuration.

## 11. What these results do and do not measure

These files measure model latency, throughput, parameter count, and sampled
process memory. The timed path starts with synthetic features and includes graph
construction. It does not time video decoding or feature extraction.

**Power and energy are not measured yet.** `energy_joules_per_video: null` means
not measured, not zero. Keep it that way for this setup check. Before collecting
paper energy results, coordinate the inline meter and a timestamped measurement
procedure with me. An unsynchronized meter reading during the entire program
would include setup, warmup, and extra memory passes, not just measured inference.

## 12. Coming back later, reruns, and shutdown

After reopening a Pi terminal or reconnecting by SSH:

```bash
cd ~/pi_benchmark
source .venv/bin/activate
```

The benchmark refuses to overwrite an existing result. If you see
`Refusing to overwrite benchmark`, the earlier file is still there. Send it to
me; ask me to arrange a separate named repeat rather than deleting it.

Use the robot manufacturer's shutdown procedure. a
normal Linux shutdown is the following (not sure if it will work here tho), run ON THE PI:

```bash
sudo shutdown -h now
```

The SSH connection will close if using a laptop. Wait for shutdown to finish
before disconnecting power. Do not pull the power during a run or installation.

## Quick troubleshooting

| What you see | What to do |
|---|---|
| No Pi desktop appears on the laptop | Expected. This guide uses a text-only SSH connection. |
| Laptop terminal says `sudo` or `source` is unknown | You are probably running a Pi command on Windows. Connect with SSH first. |
| `No such file or directory` for the ZIP | It may be on the laptop instead of the Pi, or downloaded under a different name. Repeat Section 5. |
| `ModuleNotFoundError` | Check `(.venv)` is visible, activate it as in Section 12, then repeat the relevant install step. |
| `externally-managed-environment` | Activate `.venv`; do not use `sudo pip` or override the protection. |
| `Killed`, freezing, or low-power/temperature warning | Stop and tell me. Include the Pi model, RAM, power supply, and error. |
| `throttled=0x0` | No throttling flags are set. Send the file anyway. Other values need interpretation by me. |
| Benchmark finishes but energy is `null` | Expected; power measurement is a separate task. |

When asking for help, send the exact command and its complete error text, plus
whether the window was ON THE PI or ON THE LAPTOP. Never send your password.
