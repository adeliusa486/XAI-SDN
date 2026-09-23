#!/usr/bin/env bash
# Live demonstration of XAI-SDN on Mininet, narrated for a video recording.
#
# Six steps: bring up the topology, connect the switch to the XAI-SDN
# controller, measure benign traffic with the network idle, start a
# randomised-source SYN flood, let the detector install drop rules, then show
# the resulting flow table and the controller's own summary.
#
# Everything printed here is produced by the running system. Nothing is staged.
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
D=/opt/xaisdn
say() { echo ""; echo "=============================================================="; echo "$1"; echo "=============================================================="; }
cmd() { echo "\$ $*"; eval "$@" 2>&1; }

mn -c >/dev/null 2>&1
pkill -f run_controller.py >/dev/null 2>&1
sleep 1

say "STEP 1  Environment"
cmd "mn --version 2>&1"
cmd "ovs-vsctl --version | head -1"
cmd "/opt/xaisdn/bin/python -c 'import os_ken, sklearn; print(\"os-ken\", os_ken.__version__, \"| scikit-learn\", sklearn.__version__)'"
echo "Detector: $(ls -lh $D/model.joblib 2>/dev/null | awk '{print $5}') Random Forest, 200 trees, running inside the controller process"

say "STEP 2  Starting the XAI-SDN controller (OpenFlow 1.3, port 6653)"
rm -f $D/demo_ctl.json $D/demo_ctl.log
XAISDN_DETECT=1 XAISDN_EXPLAIN=0 XAISDN_OUT=$D/demo_ctl.json \
  nohup $D/bin/python $D/run_controller.py $D/xaisdn_controller.py 6653 \
  > $D/demo_ctl.log 2>&1 &
CTL=$!
echo "controller process started, pid $CTL"
for i in $(seq 1 20); do
  if grep -q "listening on" $D/demo_ctl.log 2>/dev/null; then break; fi
  sleep 1
done
cmd "grep -v Warning $D/demo_ctl.log | tail -3"
cmd "ss -lntp 2>/dev/null | grep 6653 || netstat -lntp 2>/dev/null | grep 6653"

say "STEP 3  Building the topology and measuring baseline traffic"
cat > $D/demo_topo.py <<'PYEOF'
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel
import sys, time

setLogLevel('warning')          # keep the narration readable

net = Mininet(switch=OVSSwitch, link=TCLink, autoSetMacs=True, cleanup=True,
              controller=None)
# The controller has to be added explicitly: building a topology with
# addSwitch/addHost/addLink does not create one implicitly, and a switch with no
# controller drops every packet, which is what an earlier run of this script did.
c0 = net.addController('c0', controller=RemoteController,
                       ip='127.0.0.1', port=6653)
s1 = net.addSwitch('s1', protocols='OpenFlow13')
h1 = net.addHost('h1', ip='10.0.0.1/8')   # attacker
h2 = net.addHost('h2', ip='10.0.0.2/8')   # benign client
h3 = net.addHost('h3', ip='10.0.0.3/8')   # server
for h in (h1, h2, h3):
    net.addLink(h, s1, bw=100, delay='1ms')
net.start()

ok = False
for _ in range(20):
    probe = s1.cmd('ovs-vsctl -f table -- --columns=target,is_connected '
                   'list controller 2>/dev/null')
    if 'true' in probe.lower():
        ok = True
        break
    time.sleep(1)
print('switch s1 controller :', s1.cmd('ovs-vsctl get-controller s1').strip())
print('openflow connected   :', ok)
sys.stdout.flush()

print('')
print('--- BASELINE: benign client pings the server, no attack ---')
print(h2.cmd('ping -c 8 -i 0.3 10.0.0.3 | tail -3'))
sys.stdout.flush()

print('--- STEP 4  attacker starts a SYN flood with randomised sources ---')
h1.cmd('timeout 22 hping3 --flood --rand-source -S -p 80 10.0.0.3 '
       '> /dev/null 2>&1 &')
time.sleep(4)
print('flood running from h1 -> h3')
print(h2.cmd('ping -c 8 -i 0.3 10.0.0.3 | tail -3'))
sys.stdout.flush()

print('--- STEP 5  detector reacting: rules installed by XAI-SDN ---')
time.sleep(8)
out = s1.cmd('ovs-ofctl -O OpenFlow13 dump-flows s1')
rows = [l for l in out.strip().splitlines() if 'cookie=' in l]
print('flow table entries                   :', len(rows))
print('drop rules installed by the detector :', sum('actions=drop' in l for l in rows))
print(h2.cmd('ping -c 8 -i 0.3 10.0.0.3 | tail -3'))
sys.stdout.flush()

print('--- STEP 6  flow table after mitigation ---')
print(s1.cmd('ovs-ofctl -O OpenFlow13 dump-flows s1 | head -8'))
h1.cmd('pkill hping3 2>/dev/null')
net.stop()
sys.stdout.flush()
PYEOF
cmd "python3 -u $D/demo_topo.py"

say "STEP 6  Controller summary"
kill -TERM $CTL 2>/dev/null
sleep 4
cmd "cat $D/demo_ctl.json 2>/dev/null | head -30"
cmd "grep -viE 'warning|warn\(' $D/demo_ctl.log | tail -6"
mn -c >/dev/null 2>&1
echo ""
echo "Demonstration complete. Every line above was produced by the running system."
