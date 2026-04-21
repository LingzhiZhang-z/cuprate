import pytest

from cuprate.io import parse_main_cli_args


def test_parse_fixed_sz_s2_quantum_numbers():
    params = parse_main_cli_args(
        ["cuprate.main", "N=4", "MODE=fixed_sz_s2", "twoSz=0", "twoS=0"]
    )

    assert params.mode == "fixed_sz_s2"
    assert params.twoSz == 0
    assert params.twoS == 0


def test_parse_workflow_and_select_are_normalized():
    params = parse_main_cli_args(
        ["cuprate.main", "N=4", "MODE=full", "WORKFLOW=MULTI", "SELECT=BLOCK"]
    )

    assert params.workflow == "multi"
    assert params.select == "block"


def test_parse_match_spin_sectors_flag():
    params = parse_main_cli_args(
        ["cuprate.main", "N=4", "MODE=full", "MATCH_SPIN_SECTORS=true"]
    )

    assert params.match_spin_sectors is True


def test_parse_rejects_legacy_sector_keys():
    with pytest.raises(ValueError, match="does not accept SZ_IDX"):
        parse_main_cli_args(
            ["cuprate.main", "N=4", "MODE=fixed_sz_s2", "SZ_IDX=0", "S_IDX=0"]
        )
