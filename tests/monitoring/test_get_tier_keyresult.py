"""Partitioned data selects the pht tier; the test uses a real directory.

It used to patch os.path.isdir/os.listdir, which pinned the implementation to
one filesystem API rather than to the behaviour.
"""

from legend_data_monitor.monitoring import get_tier_keyresult


def test_hit_branch_when_the_partition_dir_is_absent(tmp_path):
    tier, key_result = get_tier_keyresult([str(tmp_path), str(tmp_path / "missing")])
    assert (tier, key_result) == ("hit", "ecal")


def test_hit_branch_when_the_partition_dir_is_empty(tmp_path):
    empty = tmp_path / "pht"
    empty.mkdir()
    tier, key_result = get_tier_keyresult([str(tmp_path), str(empty)])
    assert (tier, key_result) == ("hit", "ecal")


def test_pht_branch_when_the_partition_dir_has_content(tmp_path):
    populated = tmp_path / "pht"
    populated.mkdir()
    (populated / "some_file").touch()
    tier, key_result = get_tier_keyresult([str(tmp_path), str(populated)])
    assert (tier, key_result) == ("pht", "partition_ecal")
