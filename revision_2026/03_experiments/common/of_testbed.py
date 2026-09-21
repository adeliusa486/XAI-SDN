"""Real OpenFlow 1.3 control-plane testbed primitives.

Mininet and ONOS require hardware virtualization, which is disabled in firmware
on the evaluation machine, so a full data-plane emulator is not available. What
IS available, and is what the reviewer's question actually turns on, is a real
control plane: genuine OpenFlow 1.3 messages, encoded and decoded by the os-ken
reference protocol stack, carried over real TCP sockets between a controller
process and switch agent processes.

Every message on the wire here is a valid OpenFlow 1.3 PDU. Switch-to-controller
messages are hand-encoded against the specification and then parsed by os-ken,
so the encoding is validated by an independent implementation rather than merely
asserted. Controller-to-switch messages are serialised by os-ken directly.

This is described in the manuscript as a controller-in-the-loop testbed with
emulated switch agents. It is never described as Mininet.
"""
from __future__ import annotations

import heapq
import socket
import struct
from collections import OrderedDict
from dataclasses import dataclass, field

from os_ken.ofproto import ofproto_v1_3 as ofp
from os_ken.ofproto import ofproto_v1_3_parser as ofparser
from os_ken.ofproto import ofproto_parser

OFP_VERSION = ofp.OFP_VERSION            # 4
OFP_HEADER_LEN = 8


class DatapathStub:
    """Minimal datapath shim so os-ken message classes can serialise."""

    ofproto = ofp
    ofproto_parser = ofparser

    def __init__(self, dpid: int = 1):
        self.id = dpid
        self.xid = 0

    def set_xid(self, msg):
        self.xid += 1
        msg.set_xid(self.xid)
        return self.xid

    def send_msg(self, msg):          # not used; agents write to sockets directly
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #

def encode_header(msg_type: int, length: int, xid: int) -> bytes:
    return struct.pack("!BBHI", OFP_VERSION, msg_type, length, xid)


def encode_hello(xid: int = 0) -> bytes:
    return encode_header(ofp.OFPT_HELLO, OFP_HEADER_LEN, xid)


def encode_echo_reply(xid: int, data: bytes = b"") -> bytes:
    return encode_header(ofp.OFPT_ECHO_REPLY, OFP_HEADER_LEN + len(data), xid) + data


def encode_features_reply(xid: int, dpid: int, n_buffers: int = 256,
                          n_tables: int = 1) -> bytes:
    """OFPT_FEATURES_REPLY: header + datapath_id, n_buffers, n_tables,
    auxiliary_id, pad[2], capabilities, reserved."""
    body = struct.pack("!QIBB2xII", dpid, n_buffers, n_tables, 0,
                       ofp.OFPC_FLOW_STATS | ofp.OFPC_TABLE_STATS, 0)
    return encode_header(ofp.OFPT_FEATURES_REPLY,
                         OFP_HEADER_LEN + len(body), xid) + body


def encode_packet_in(xid: int, buffer_id: int, reason: int, table_id: int,
                     cookie: int, data: bytes) -> bytes:
    """OFPT_PACKET_IN with an empty OXM match, per the OpenFlow 1.3 layout.

    header | buffer_id(4) total_len(2) reason(1) table_id(1) cookie(8)
           | ofp_match (OXM, header-only, padded to 8) | pad(2) | data
    """
    match = struct.pack("!HH4x", ofp.OFPMT_OXM, 4)     # 8 bytes, no OXM fields
    body = struct.pack("!IHBBQ", buffer_id, len(data), reason, table_id, cookie)
    total = OFP_HEADER_LEN + len(body) + len(match) + 2 + len(data)
    return (encode_header(ofp.OFPT_PACKET_IN, total, xid) + body + match +
            b"\x00\x00" + data)


def encode_flow_mod(dp: DatapathStub, xid: int, priority: int, cookie: int,
                    idle_timeout: int, hard_timeout: int,
                    match_kwargs: dict, out_port_drop: bool = True) -> bytes:
    """Controller-to-switch FLOW_MOD, serialised by os-ken itself."""
    match = ofparser.OFPMatch(**match_kwargs)
    inst = [] if out_port_drop else [
        ofparser.OFPInstructionActions(
            ofp.OFPIT_APPLY_ACTIONS,
            [ofparser.OFPActionOutput(ofp.OFPP_NORMAL)])]
    msg = ofparser.OFPFlowMod(
        datapath=dp, cookie=cookie, cookie_mask=0, table_id=0,
        command=ofp.OFPFC_ADD, idle_timeout=idle_timeout,
        hard_timeout=hard_timeout, priority=priority,
        buffer_id=ofp.OFP_NO_BUFFER, out_port=ofp.OFPP_ANY,
        out_group=ofp.OFPG_ANY, flags=ofp.OFPFF_SEND_FLOW_REM,
        match=match, instructions=inst)
    msg.set_xid(xid)
    msg.serialize()
    return bytes(msg.buf)


def encode_packet_out(dp: DatapathStub, xid: int, buffer_id: int,
                      in_port: int, data: bytes = b"") -> bytes:
    msg = ofparser.OFPPacketOut(
        datapath=dp, buffer_id=buffer_id, in_port=in_port,
        actions=[ofparser.OFPActionOutput(ofp.OFPP_NORMAL)], data=data)
    msg.set_xid(xid)
    msg.serialize()
    return bytes(msg.buf)


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #

def read_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (ConnectionResetError, OSError):
            return None
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_message(sock: socket.socket) -> tuple[int, int, int, bytes] | None:
    """Return (version, type, xid, whole_message_bytes) or None on close."""
    head = read_exact(sock, OFP_HEADER_LEN)
    if head is None:
        return None
    version, mtype, length, xid = ofproto_parser.header(head)
    rest = read_exact(sock, length - OFP_HEADER_LEN) if length > OFP_HEADER_LEN else b""
    if rest is None:
        return None
    return version, mtype, xid, head + rest


def validate_with_oskan(raw: bytes) -> str:
    """Parse a PDU with os-ken so the hand-encoded bytes are checked independently.

    os-ken registers parsers only for messages a controller receives, so the
    controller-to-switch direction (FLOW_MOD, PACKET_OUT) is validated by its
    own serialiser instead and reported as 'header-valid'.
    """
    version, mtype, mlen, xid = ofproto_parser.header(raw)
    if version != OFP_VERSION or mlen != len(raw):
        return f"INVALID(version={version}, declared_len={mlen}, actual={len(raw)})"
    dp = DatapathStub()
    try:
        msg = ofproto_parser.msg(dp, version, mtype, mlen, xid, raw)
    except Exception:
        return f"header-valid(type={mtype})"
    if msg is None:
        return f"header-valid(type={mtype})"
    return type(msg).__name__


# --------------------------------------------------------------------------- #
# TCAM model (E8)
# --------------------------------------------------------------------------- #

@dataclass
class TcamModel:
    """Bounded flow table with a selectable aggregation and eviction policy.

    aggregation policies, in increasing order of how much of the match they
    wildcard:

      exact           source address, destination address, destination port,
                      protocol: one entry per distinct flow
      prefix24        as exact, with the source collapsed to its /24
      prefix16        as exact, with the source collapsed to its /16
      noport          source address, destination address, protocol: the
                      destination port is wildcarded
      victim          destination address, destination port, protocol: the
                      source is wildcarded entirely
      victim_service  destination address and protocol only

    Which of these bounds the table depends on which field the flood varies, so
    the diversity of each field is reported alongside the occupancy results
    rather than assumed.

    eviction: 'idle' | 'hard' | 'lru' | 'lfu'
    """
    capacity: int = 4096
    aggregation: str = "exact"
    eviction: str = "lru"
    idle_timeout: float = 10.0
    hard_timeout: float = 60.0

    rules: dict = field(default_factory=dict)   # key -> dict(installed, last, hits)
    _touch: "OrderedDict" = field(default_factory=OrderedDict)  # key -> None, LRU order
    _heap: list = field(default_factory=list)   # lazy (hits, seq, key) heap for LFU
    _seq: int = 0
    installs: int = 0
    hits: int = 0
    evictions: int = 0
    overflows: int = 0
    timeouts: int = 0
    peak_occupancy: int = 0

    def key_for(self, src_ip: str, dst_ip: str, dport: int, proto: int):
        if self.aggregation == "exact":
            return (src_ip, dst_ip, dport, proto)
        if self.aggregation == "noport":
            return (src_ip, dst_ip, proto)
        if self.aggregation == "victim":
            return ("*", dst_ip, dport, proto)
        if self.aggregation == "victim_service":
            return ("*", dst_ip, proto)
        parts = src_ip.split(".")
        if len(parts) != 4:
            return (src_ip, dst_ip, dport, proto)
        if self.aggregation == "prefix24":
            return (".".join(parts[:3]) + ".0/24", dst_ip, dport, proto)
        return (".".join(parts[:2]) + ".0.0/16", dst_ip, dport, proto)

    # Expiry and eviction were originally written as linear scans over the whole
    # table. That is O(capacity) per admitted flow, which at a 16,384-entry
    # capacity turns a 40,000-flow replay into hundreds of millions of dictionary
    # probes. The structures below give the same answers in amortized constant
    # time: `rules` is never re-inserted, so its iteration order is installation
    # order, and `_touch` tracks recency separately.

    def _drop(self, k) -> None:
        del self.rules[k]
        self._touch.pop(k, None)

    def _expire(self, now: float) -> None:
        # Hard timeout applies under every policy, and installation order is
        # monotone, so the expired entries are a prefix of `rules`.
        while self.rules:
            k = next(iter(self.rules))
            if now - self.rules[k]["installed"] <= self.hard_timeout:
                break
            self._drop(k)
            self.timeouts += 1
        # The idle timeout applies only to the recency-driven policies.
        if self.eviction in ("idle", "lru"):
            while self._touch:
                k = next(iter(self._touch))
                if k not in self.rules:
                    self._touch.pop(k, None)
                    continue
                if now - self.rules[k]["last"] <= self.idle_timeout:
                    break
                self._drop(k)
                self.timeouts += 1

    def _lfu_victim(self):
        """Least-frequently-used key, via a heap that is corrected lazily.

        A hit pushes the key again rather than reordering the heap, so entries
        may be stale; a stale entry is discarded when it reaches the top and its
        recorded count disagrees with the live one.
        """
        while self._heap:
            hits, _, k = heapq.heappop(self._heap)
            r = self.rules.get(k)
            if r is not None and r["hits"] == hits:
                return k
        return next(iter(self.rules))

    def _make_room(self) -> None:
        if len(self.rules) < self.capacity:
            return
        self.overflows += 1
        if self.eviction == "lfu":
            victim = self._lfu_victim()
        elif self.eviction in ("lru", "idle"):
            victim = next(iter(self._touch))
        else:                                   # 'hard': oldest installation
            victim = next(iter(self.rules))
        self._drop(victim)
        self.evictions += 1

    def admit(self, src_ip: str, dst_ip: str, dport: int, proto: int,
              now: float) -> bool:
        """Return True if a new rule had to be installed (a control-plane event)."""
        self._expire(now)
        k = self.key_for(src_ip, dst_ip, dport, proto)
        r = self.rules.get(k)
        if r is not None:
            r["last"] = now
            r["hits"] += 1
            self.hits += 1
            self._touch.pop(k, None)
            self._touch[k] = None
            if self.eviction == "lfu":
                self._seq += 1
                heapq.heappush(self._heap, (r["hits"], self._seq, k))
            return False
        self._make_room()
        self.rules[k] = {"installed": now, "last": now, "hits": 1}
        self._touch[k] = None
        if self.eviction == "lfu":
            self._seq += 1
            heapq.heappush(self._heap, (1, self._seq, k))
        self.installs += 1
        self.peak_occupancy = max(self.peak_occupancy, len(self.rules))
        return True

    def snapshot(self) -> dict:
        return {"aggregation": self.aggregation, "eviction": self.eviction,
                "capacity": self.capacity, "occupancy": len(self.rules),
                "peak_occupancy": self.peak_occupancy, "installs": self.installs,
                "hits": self.hits, "evictions": self.evictions,
                "overflows": self.overflows, "timeouts": self.timeouts,
                "hit_rate": round(self.hits / max(self.hits + self.installs, 1), 6)}
