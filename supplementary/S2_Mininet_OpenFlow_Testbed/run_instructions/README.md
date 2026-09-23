# Reproducing the live control-plane experiments

Two separate testbeds are reported in the article, and they answer different
questions. Run them in the order below.

| | Testbed | Question it answers | Article location |
|---|---|---|---|
| 1 | Controller-only (`E4`, `E7`) | What does the controller cost, and at what offered rate does one instance saturate? | Sec. VII-R, Table 17 |
| 2 | Mininet + Open vSwitch (`E4b`) | Does the mitigation the detector installs actually protect traffic? | Sec. VII-P, Table 15 |
| 3 | Flow-table occupancy (`E8b`) | Which aggregation policy keeps a bounded TCAM from overflowing? | Sec. VII-S, Table 18 |

The first is a control plane without a data plane: it exchanges genuine OpenFlow
1.3 messages with switch agents over TCP, but no packet traverses a switch. The
second adds a real data plane. Neither is a network of operational scale, and
Sec. IX-D records that limit.

---

## 0. Prerequisites

Mininet does not run on Windows. Use Linux, or WSL2 with a kernel that supports
network namespaces.

```bash
sudo apt-get install -y mininet openvswitch-switch python3-pip
python3 -m pip install -r ../../S6_Environment/requirements.txt
```

Verify the versions the article reports:

```bash
mn --version                 # expect 2.3.0
ovs-vsctl --version          # expect 3.7.1
```

Any other versions will still run, but the absolute numbers will differ.

---

## 1. Set up Open vSwitch

```bash
sudo bash ../ovs/setup_mininet.sh
```

This starts `ovs-vswitchd` and `ovsdb-server` and clears any bridge left behind
by a previous run. That last step matters: a controller killed without cleanup
keeps holding its TCP port, and a new run's switch agents will attach to the
orphan while the fresh process is sampled at zero CPU. Each session in the
harness binds its own ephemeral port for the same reason.

---

## 2. Controller cost and the offered-rate sweep

```bash
sudo python3 ../controller/e4_controller_testbed.py
```

Writes `E4_controller_testbed.json`, `E4_timeseries_detection_only.csv`,
`E4_timeseries_with_explanations.csv` and `E7_rate_sweep.csv` into
`../rate_sweeps/`.

Read the sweep with the article's caveat in hand. The sweep is saturated at
every offered rate, so it bounds goodput rather than locating a knee, and the
range across session configurations (34 to 114 events/s) is wider than any
variation the sweep attributes to offered rate. Sec. VII-R states the range
rather than a single ceiling.

---

## 3. Live Mininet deployment

```bash
sudo python3 ../controller/e4b_mininet_testbed.py
```

Builds the topology in `../topology/run_topology.py`: one Open vSwitch datapath
speaking OpenFlow 1.3 to the detector running inside an os-ken controller
process, with three hosts on 100 Mbit links at 1 ms delay (an attacker, a benign
client and a server).

The run proceeds through a baseline measurement, a SYN flood with randomized
sources, and the detector's mitigation. It writes `E4b_mininet_testbed.json` and
the console capture in `../evidence/`.

Expect Linux `tc`/HTB warnings of the form `sch_htb: quantum of class ... is
big` while the links are configured. They are unrelated to the measurement.

---

## 4. Flow-table occupancy

```bash
python3 ../tcam_evaluation/e8b_tcam_policy.py
```

Evaluates six aggregation policies at three table capacities and writes
`E8b_occupancy.csv`, `E8b_tcam_grid.csv` and `E8b_tcam_policy.json`.

Note what this experiment measures on this corpus. 99.08% of the SYN partition's
flows carry a single source address, so collapsing the source field changes
nothing: the aggregation that bounds the table is the one that wildcards the
port. Source-field pressure is exercised in the emulated network of step 3
instead, where sources are randomized at the packet level.

---

## 5. Recording the demonstration

```bash
python3 ../traffic_generation/record_demo.py
```

Produces the narrated walkthrough. By default it writes to `~/Downloads`, not to
the project tree.

---

## Expected output

From the live deployment on the hardware in Sec. VI-C:

| Quantity | Value |
|---|---|
| Benign throughput retained under attack, with detection | 100% |
| Benign throughput retained under attack, no controller | 27% |
| Added packet loss, all phases | 0% |
| Flow entries / drop rules at peak | 999 / 997 |
| Controller decision latency, median | 10.84 ms |
| Controller CPU | ~101% of one core |

Run-to-run variation on different hardware is expected. The qualitative result
that should reproduce is that benign traffic stays at 0% added loss across the
baseline, flood and mitigation phases, and that throughput collapses without the
detector.
