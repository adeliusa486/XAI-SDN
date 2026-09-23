# Known issues in the captured evidence

Six captures in this directory record a failed command rather than a
measurement. They were taken by the evidence harness after the Mininet
namespace had already been torn down, so `ovs-ofctl` could no longer reach
the bridge and returned:

```
ovs-ofctl: s1 is not a bridge or a socket
```

Affected files:

- `detect__flow_table.txt`
- `detect__ofctl_show.txt`
- `detect_explain__flow_table.txt`
- `detect_explain__ofctl_show.txt`
- `no_controller__flow_table.txt`
- `no_controller__ofctl_show.txt`

They are included unaltered for completeness. They are **not** evidence of
anything and no manuscript value depends on them. The flow-table counts
reported in Sec. VII-P come from `E4b_mininet_testbed.json` and from the
live-demo transcript in this directory, both captured while the topology was
running.
