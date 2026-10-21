# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The closer can read the migrated contracts without a change-control checkout."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

_CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
# Git blob identities at change-control commit
# 49cfc14c21a2c50f9964c1aeed4b66001ce279a3; no sibling checkout is needed to run.
_OMN_17390_BLOB_ID = "ab72a43d7e4e6b521142debe3ef25848ce7c2296"
_OMN_20868_BLOB_ID = "520d4e6b79f88cf32727fde521e59791050af3ca"
_SOURCE_BLOBS = {
    "OMN-17390": _OMN_17390_BLOB_ID,
    "OMN-20868": _OMN_20868_BLOB_ID,
}


class TestMigratedContracts(unittest.TestCase):
    """Each ticket resolves locally to exactly the original contract bytes."""

    def test_contracts_preserve_source_bytes(self) -> None:
        for ticket, expected in _SOURCE_BLOBS.items():
            with self.subTest(ticket=ticket):
                data = (_CONTRACTS / f"{ticket}.yaml").read_bytes()
                git_blob = f"blob {len(data)}\0".encode() + data
                assert (
                    hashlib.sha1(git_blob, usedforsecurity=False).hexdigest()
                    == expected
                )


if __name__ == "__main__":
    unittest.main()
