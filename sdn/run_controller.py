#!/usr/bin/env python3
"""Launch the XAI-SDN os-ken application without the osken-manager console script.

The os-ken 4.2.2 wheel installed here does not ship the os_ken.cmd package, so the
osken-manager entry point is absent. Everything the entry point does is available
from the library: instantiate the application manager, load the app, and start the
OpenFlow listener. Doing it directly also lets the demonstration set the listen
port and the detector's environment without a configuration file.

    python run_controller.py xaisdn_controller.py [port]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from os_ken import cfg  # noqa: E402
from os_ken.base import app_manager  # noqa: E402
from os_ken.controller import controller as of_controller  # noqa: E402
from os_ken.lib import hub  # noqa: E402


def main() -> int:
    app_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
        HERE / "xaisdn_controller.py"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 6653

    sys.path.insert(0, str(app_path.parent))
    module = app_path.stem

    # Importing os_ken.controller.controller registers ofp_tcp_listen_port, so
    # the value is set directly rather than through a command line the console
    # script would normally parse.
    CONF = cfg.CONF
    CONF(args=[], project="os_ken", default_config_files=[])
    CONF.set_override("ofp_tcp_listen_port", port)

    mgr = app_manager.AppManager.get_instance()
    mgr.load_apps([module])
    contexts = mgr.create_contexts()
    services = list(mgr.instantiate_apps(**contexts))

    ctl = of_controller.OpenFlowController()
    services.append(hub.spawn(ctl))
    print(f"XAI-SDN controller listening on 0.0.0.0:{port} (app {module})",
          flush=True)
    try:
        hub.joinall(services)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("XAISDN_MODEL", str(HERE / "model.joblib"))
    sys.exit(main())
