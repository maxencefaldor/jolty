# Copyright 2026 Maxence Faldor
"""Every crate claims the Rust it is built with, the one `rust-toolchain.toml` pins.

The pin is Rust's latest stable release, and `rust-version` in each crate's manifest,
inherited from the workspace's, says what the crate's users need: a claim of an older
Rust would be one that nothing builds or tests jolty with.
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
