"""Count real network connections while the app's models load, with and without HF_HUB_OFFLINE.

    python -m evaluation.offline_check offline    # as the app runs: expect 0 connections
    python -m evaluation.offline_check online     # without the setting: the Hub is asked for newer revisions
"""
import os
import socket
import sys
import time

if sys.argv[1:] == ["offline"]:
    os.environ.setdefault(
        "HF_HUB_OFFLINE",
        "1")  # before huggingface_hub is imported, as in app/ui.py
elif sys.argv[1:] != ["online"]:
    sys.exit(__doc__)

connections = []
_connect = socket.socket.connect


def logged_connect(self, address):
    connections.append(address)
    return _connect(self, address)


socket.socket.connect = logged_connect

import huggingface_hub.constants  # noqa: E402

from medguiderag.retrieval.hybrid import HybridRetriever  # noqa: E402

start = time.time()
results = HybridRetriever("BAAI/bge-m3").search(
    "Как се лекува херпесният енцефалит?", k=5)
outside = [a for a in connections if not str(a[0]).startswith(("127.", "::1"))]
print(
    f"HF_HUB_OFFLINE={huggingface_hub.constants.HF_HUB_OFFLINE} | {len(results)} chunks | {time.time() - start:.1f} s | "
    f"connections leaving the machine: {len(outside)} {sorted({str(a[0]) for a in outside})}"
)
