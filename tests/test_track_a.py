from mlops.track_a.data import EXPECTED_MINIMUM_ROWS, _row_count


def test_dataset_guard_rejects_small_csv(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_text("a\n1\n")
    assert _row_count(path) < EXPECTED_MINIMUM_ROWS
