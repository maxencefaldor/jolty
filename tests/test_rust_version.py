# Copyright 2026 Maxence Faldor
"""Verification that every crate declares the MSRV pinned in `rust-toolchain.toml`.

The pin tracks the latest stable Rust release, and workspace-inherited `rust-version`
declares the minimum toolchain required by downstream users. Claiming an older Rust
version would advertise an untested toolchain.
"""

import tomllib

from workspace import ROOT, crates


def test_every_crate_claims_the_pinned_toolchain() -> None:
    with (ROOT / "rust-toolchain.toml").open("rb") as pin:
        channel = tomllib.load(pin)["toolchain"]["channel"]
    claims = {
        crate["name"]: crate["rust_version"] for crate in crates(dependencies=False)
    }
    assert claims == dict.fromkeys(claims, channel)
