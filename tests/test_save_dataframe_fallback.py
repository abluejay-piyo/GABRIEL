from pathlib import Path

import pandas as pd

from gabriel.utils.file_utils import save_dataframe_with_fallback


def test_save_dataframe_with_fallback_splits_large_files(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame({"value": range(200_005)})
    target = tmp_path / "final.csv"

    original_to_csv = pd.DataFrame.to_csv

    def flaky_to_csv(self, path_or_buf=None, *args, **kwargs):
        if path_or_buf is not None and str(path_or_buf) == str(target):
            raise OSError("simulated save failure")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", flaky_to_csv)

    saved_paths = save_dataframe_with_fallback(df, str(target), index=False, label="Test")

    expected = [
        str(tmp_path / "final_1.csv"),
        str(tmp_path / "final_2.csv"),
        str(tmp_path / "final_3.csv"),
    ]
    assert saved_paths == expected
    for part in expected:
        assert Path(part).exists()


def test_save_dataframe_with_fallback_returns_empty_when_all_writes_fail(
    tmp_path: Path, monkeypatch
) -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})

    def always_fail(self, path_or_buf=None, *args, **kwargs):
        raise OSError("all writes fail")

    monkeypatch.setattr(pd.DataFrame, "to_csv", always_fail)

    saved_paths = save_dataframe_with_fallback(
        df,
        str(tmp_path / "never.csv"),
        index=False,
        label="Test",
    )

    assert saved_paths == []
