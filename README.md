# nc-jax-demo

This repo contains a minimal implementation of a jax training loop for testing 8x MI nodes.

Requirements are handled by uv.

## Jobs

[job.yaml](./job.yaml) configures a job. It uses the `rocm/dev-ubuntu-24.04:7.2.4-complete` image and sets [LLVM_PATH="/opt/rocm/llvm"](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.4/install/3rd-party/jax-install.html) and `JAX_PLATFORM="rocm"` then:
1. Clones this repo
2. Installs uv
3. Syncs this repo's requirements
4. runs `main.py`

## Code

[main.py](./main.py) contains a minimal distributed training loop.

### Model

A deep MLP with scanned layers.

### Loss

MSE between model output and target.

### Data

Input is random gaussian `[batch, feature]` shaped arrays.

Output target is `[batch, feature]` shaped arrays of ones.

### Distributed

2D mesh with axes `["data", "fsdp"]` of shape `[1, NUM_DEVICES]`.

Parameters and optimizer states are fully replicated.

Data is partitioned along the batch dimension across the `data` and `fsdp` axes. The feature dimension is unsharded.

### Experiments
1. Scanned or unscanned layers -> both okay
2. Explicit or jit compilation -> both okay
3. Nested scans -> okay
4. Initialize params on single device then broadcast -> okay

## Example Log
```
Cloning into 'nc-jax-demo'...
2b5b95b7cdea018644ec9265a799308a734647a7
Collecting uv==0.11.21
  Downloading uv-0.11.21-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Downloading uv-0.11.21-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 25.1/25.1 MB 145.2 MB/s  0:00:00
Installing collected packages: uv
Successfully installed uv-0.11.21

[notice] A new release of pip is available: 26.1.1 -> 26.2.1
[notice] To update, run: pip install --upgrade pip
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment at: .venv
Resolved 26 packages in 0.65ms
Downloading numpy (15.9MiB)
Downloading scipy (33.7MiB)
Downloading jaxlib (83.8MiB)
Downloading jax (3.1MiB)
Downloading jax-rocm7-pjrt (131.0MiB)
Downloading jax-rocm7-plugin (4.1MiB)
 Downloaded jax-rocm7-plugin
 Downloaded jax
 Downloaded numpy
 Downloaded scipy
 Downloaded jaxlib
 Downloaded jax-rocm7-pjrt
Prepared 10 packages in 1.76s
Installed 10 packages in 35ms
 + absl-py==2.5.0
 + jax==0.11.1
 + jax-rocm7-pjrt==0.11.1
 + jax-rocm7-plugin==0.11.1
 + jaxlib==0.11.1
 + ml-dtypes==0.6.0
 + numpy==2.5.2
 + opt-einsum==3.4.0
 + optax==0.2.8
 + scipy==1.18.0
W0915 03:14:35.254544     390 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0002:00:01.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:35.254588     390 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 0 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:35.254986     384 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0003:00:03.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:35.255011     384 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 6 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:35.255538     387 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0002:00:04.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:35.255564     387 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 3 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:35.255843     386 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0002:00:03.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:35.255869     386 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 2 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:35.256185     391 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0003:00:02.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:35.256201     391 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 5 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:35.256547     385 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0003:00:04.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:35.256559     385 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 7 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:35.256807     388 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0003:00:01.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:35.256823     388 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 4 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:35.257200     389 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0002:00:02.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:35.257219     389 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 1 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:36.162998     315 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0002:00:01.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:36.163018     315 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 0 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:36.163242     315 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0002:00:02.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:36.163245     315 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 1 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:36.163590     315 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0002:00:03.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:36.163593     315 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 2 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:36.163770     315 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0002:00:04.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:36.163773     315 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 3 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:36.163959     315 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0003:00:01.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:36.163961     315 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 4 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:36.164240     315 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0003:00:02.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:36.164242     315 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 5 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:36.164585     315 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0003:00:03.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:36.164587     315 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 6 via rocm_smi. Assuming PCIe Gen4 x16.
W0915 03:14:36.164783     315 rocm_pcie_bandwidth.cc:75] rsmi_dev_gpu_metrics_info_get failed for 0003:00:04.0: RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from a file) or provided to function is not what was expected
W0915 03:14:36.164785     315 rocm_executor.cc:1176] Could not determine PCIe bandwidth for device 7 via rocm_smi. Assuming PCIe Gen4 x16.
2026-09-15 03:14:36,175 node=alert-mallard
2026-09-15 03:14:36,175 jax=0.11.1
2026-09-15 03:14:36,175 backend=gpu device_types=['AMD Radeon Graphics']
2026-09-15 03:14:36,175 devices=[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3), RocmDevice(id=4), RocmDevice(id=5), RocmDevice(id=6), RocmDevice(id=7)] mesh=Mesh('data': 1, 'fsdp': 8, axis_types=(Auto, Auto))
2026-09-15 03:14:44,711 initialization complete
E0915 03:15:02.160012     395 rendezvous.cc:108] [id=6] This thread has been waiting for `[1] [rank=1] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` for 10 seconds and may be stuck. All 8 threads joined the rendezvous, however the leader has not marked the rendezvous as completed. Leader can be deadlocked inside the rendezvous callback.
E0915 03:15:02.160014     410 rendezvous.cc:108] [id=4] This thread has been waiting for `[6] [rank=6] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` for 10 seconds and may be stuck. All 8 threads joined the rendezvous, however the leader has not marked the rendezvous as completed. Leader can be deadlocked inside the rendezvous callback.
E0915 03:15:02.160029     404 rendezvous.cc:108] [id=0] This thread has been waiting for `[4] [rank=4] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` for 10 seconds and may be stuck. All 8 threads joined the rendezvous, however the leader has not marked the rendezvous as completed. Leader can be deadlocked inside the rendezvous callback.
E0915 03:15:02.160002     392 rendezvous.cc:108] [id=2] This thread has been waiting for `[0] [rank=0] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` for 10 seconds and may be stuck. All 8 threads joined the rendezvous, however the leader has not marked the rendezvous as completed. Leader can be deadlocked inside the rendezvous callback.
E0915 03:15:02.160017     398 rendezvous.cc:108] [id=1] This thread has been waiting for `[2] [rank=2] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` for 10 seconds and may be stuck. All 8 threads joined the rendezvous, however the leader has not marked the rendezvous as completed. Leader can be deadlocked inside the rendezvous callback.
E0915 03:15:02.160014     401 rendezvous.cc:108] [id=5] This thread has been waiting for `[3] [rank=3] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` for 10 seconds and may be stuck. All 8 threads joined the rendezvous, however the leader has not marked the rendezvous as completed. Leader can be deadlocked inside the rendezvous callback.
E0915 03:15:02.159988     407 rendezvous.cc:108] [id=3] This thread has been waiting for `[5] [rank=5] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` for 10 seconds and may be stuck. All 8 threads joined the rendezvous, however the leader has not marked the rendezvous as completed. Leader can be deadlocked inside the rendezvous callback.
E0915 03:15:25.101290     395 rendezvous.cc:130] [id=6] This thread is unstuck waiting for `[1] [rank=1] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` after 32.941385531s. All threads joined the rendezvous on time however the leader took long time to complete it. Warning above was a false-positive. Perhaps the timeout is too short.
E0915 03:15:25.101312     392 rendezvous.cc:130] [id=2] This thread is unstuck waiting for `[0] [rank=0] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` after 32.941433841s. All threads joined the rendezvous on time however the leader took long time to complete it. Warning above was a false-positive. Perhaps the timeout is too short.
E0915 03:15:25.101306     410 rendezvous.cc:130] [id=4] This thread is unstuck waiting for `[6] [rank=6] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` after 32.941422271s. All threads joined the rendezvous on time however the leader took long time to complete it. Warning above was a false-positive. Perhaps the timeout is too short.
E0915 03:15:25.101319     401 rendezvous.cc:130] [id=5] This thread is unstuck waiting for `[3] [rank=3] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` after 32.941432791s. All threads joined the rendezvous on time however the leader took long time to complete it. Warning above was a false-positive. Perhaps the timeout is too short.
E0915 03:15:25.101319     407 rendezvous.cc:130] [id=3] This thread is unstuck waiting for `[5] [rank=5] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` after 32.941439191s. All threads joined the rendezvous on time however the leader took long time to complete it. Warning above was a false-positive. Perhaps the timeout is too short.
E0915 03:15:25.101315     404 rendezvous.cc:130] [id=0] This thread is unstuck waiting for `[4] [rank=4] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` after 32.941458101s. All threads joined the rendezvous on time however the leader took long time to complete it. Warning above was a false-positive. Perhaps the timeout is too short.
E0915 03:15:25.101316     398 rendezvous.cc:130] [id=1] This thread is unstuck waiting for `[2] [rank=2] [run_id=2012504188] Initialize clique: devices=8:[0,1,2,3,4,5,6,7]; local_participants=8; communication_id=0; incarnations=[]` after 32.941451251s. All threads joined the rendezvous on time however the leader took long time to complete it. Warning above was a false-positive. Perhaps the timeout is too short.
2026-09-15 03:15:25,197 step=1 loss=1.475494 seconds=40.485538
2026-09-15 03:15:25,311 step=2 loss=1.403528 seconds=0.114049
2026-09-15 03:15:25,314 step=3 loss=1.336362 seconds=0.002889
2026-09-15 03:15:25,317 step=4 loss=1.279190 seconds=0.002744
2026-09-15 03:15:25,320 step=5 loss=1.211341 seconds=0.002979
2026-09-15 03:15:25,323 step=6 loss=1.160329 seconds=0.002712
2026-09-15 03:15:25,325 step=7 loss=1.103712 seconds=0.002761
2026-09-15 03:15:25,328 step=8 loss=1.045209 seconds=0.002486
2026-09-15 03:15:25,331 step=9 loss=1.001450 seconds=0.002665
2026-09-15 03:15:25,333 step=10 loss=0.958071 seconds=0.002542
...
```