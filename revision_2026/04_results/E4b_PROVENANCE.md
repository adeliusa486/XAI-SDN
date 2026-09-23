# E4b provenance note

`E4b_mininet_testbed.json` holds the run of 2026-09-21T13:46. The numbers in it
are the ones quoted in Section VII-P, Table 15, Fig. 5 and the response to
Reviewer 4, Concern 2.

On 2026-09-23 two defects were fixed in the surrounding code **without re-running
the experiment**, because neither defect can change a measurement:

1. `e4b_mininet_testbed.py` captured the switch flow table and the OpenFlow
   connection state by shelling into WSL *after* the Mininet topology had been
   torn down. Those captures therefore recorded
   `ovs-ofctl: s1 is not a bridge or a socket` instead of a dump. The reported
   entry and drop-rule counts never came from that path. They come from
   `run_topology.py`, which dumps the table with `s1.cmd(...)` from inside the
   live namespace while the network is still up. The capture now reuses that
   live dump.

2. `run_topology.py` stored `net.pingAll()` under the name
   `reachability_before`. That call returns the percentage of pings *dropped*,
   so the stored `0.0` meant full reachability, not none. The field is now
   `pingall_loss_percent`, with `reachability_percent` derived beside it.

The stored result predates both fixes, so it still contains the old field name
and the failed captures. Its measurements stand. The result file's timestamp was
refreshed on 2026-09-23 so that `run_queue.py` does not re-run E4b and overwrite
numbers that are already published in the manuscript.

A deliberate re-run is safe to do, but it will produce new throughput, latency
and packet-in figures, and Table 15, Fig. 5 and the R4.2 response must all be
updated together if that happens.
