from unittest.mock import patch

from legend_data_monitor.monitoring import get_tier_keyresult


def test_get_tier_keyresult_hit_branch():
    # tiers[1] does not exist, returns default hit+ecal
    with (
        patch("os.path.isdir", return_value=False),
        patch("os.listdir", return_value=[]),
    ):
        tier, key_result = get_tier_keyresult(["tier0", "tier1"])
        assert tier == "hit"
        assert key_result == "ecal"

    # tiers[1] exists but is empty, returns default hit+ecal
    with (
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=[]),
    ):
        tier, key_result = get_tier_keyresult(["tier0", "tier1"])
        assert tier == "hit"
        assert key_result == "ecal"


def test_get_tier_keyresult_pht_branch():
    # tiers[1] exists and is not empty, returns pht+partition_ecal
    with (
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=["some_file"]),
    ):
        tier, key_result = get_tier_keyresult(["tier0", "tier1"])
        assert tier == "pht"
        assert key_result == "partition_ecal"
