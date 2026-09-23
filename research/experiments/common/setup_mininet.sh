#!/usr/bin/env bash
# Install Mininet, Open vSwitch and a Ryu-compatible controller inside the
# existing Ubuntu WSL distro, then verify with a smoke test.
#
# Run from Windows after the Virtual Machine Platform feature is enabled and the
# machine has rebooted:
#
#     bash 03_experiments/common/setup_mininet.sh
#
# Idempotent. Safe to re-run. Do not run while E4 is running, because both bind
# the OpenFlow port.
set -uo pipefail

echo "=== checking WSL ==="
if ! wsl.exe --status >/dev/null 2>&1; then
  echo "WSL is not responding. Enable the Virtual Machine Platform feature and reboot:"
  echo "    (Administrator PowerShell)  wsl.exe --install --no-distribution"
  exit 1
fi
if wsl.exe --status 2>&1 | tr -d '\0' | grep -qi "virtualization is not enabled"; then
  echo "WSL2 still reports virtualization disabled. Reboot has not taken effect."
  exit 1
fi
echo "WSL is available."

# A login shell inherits the Windows PATH, and entries such as
# 'C:/Program Files/...' break word splitting inside WSL. Use a clean
# POSIX PATH instead.
WSLPATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
run() { wsl.exe -d Ubuntu -u root -- bash -c "export PATH=$WSLPATH; $1"; }

echo
echo "=== distro ==="
run 'cat /etc/os-release | head -2' || { echo "Ubuntu distro unreachable"; exit 1; }

echo
echo "=== installing packages (this takes a few minutes) ==="
run 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq' || true
run 'export DEBIAN_FRONTEND=noninteractive; apt-get install -y -qq \
       mininet openvswitch-switch openvswitch-common \
       python3-pip python3-venv iperf3 net-tools iproute2 tcpdump' || true

echo
echo "=== starting Open vSwitch ==="
run 'service openvswitch-switch start || /usr/share/openvswitch/scripts/ovs-ctl start' || true
run 'ovs-vsctl show >/dev/null 2>&1 && echo "ovs responding" || echo "ovs NOT responding"'

echo
echo "=== controller stack ==="
run 'test -d /opt/xaisdn || python3 -m venv /opt/xaisdn'
run '/opt/xaisdn/bin/pip install -q --upgrade pip'
run '/opt/xaisdn/bin/pip install -q --only-binary=:all: numpy scipy scikit-learn joblib psutil os-ken shap' || true
run '/opt/xaisdn/bin/python -c "import os_ken, sklearn, shap, joblib; print(\"controller stack ok\")"' || true

echo
echo "=== versions ==="
run 'mn --version 2>/dev/null || echo "mininet: NOT INSTALLED"'
run 'ovs-vsctl --version 2>/dev/null | head -1 || echo "ovs: NOT INSTALLED"'

echo
echo "=== smoke test: 2 hosts, 1 switch, remote controller absent ==="
run 'mn --test pingall --topo single,2 2>&1 | tail -5' || \
  echo "smoke test failed; Mininet may need a running controller or additional privileges"

echo
echo "=== done ==="
echo "If the smoke test reported '0% dropped', Mininet is working."
echo "Next: python 03_experiments/e4b_mininet_testbed.py"
