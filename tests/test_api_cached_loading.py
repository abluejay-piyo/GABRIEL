from pathlib import Path

import pandas as pd

from gabriel.api import _load_cached_dataframe
from gabriel.utils.file_utils import save_dataframe_with_fallback


def test_load_cached_dataframe_uses_split_fallback_when_primary_missing(tmp_path: Path) -> None:
    (tmp_path / "ratings_cleaned_2.csv").write_text("row,val\n2,b\n")
    (tmp_path / "ratings_cleaned_10.csv").write_text("row,val\n10,j\n")
    (tmp_path / "ratings_cleaned_1.csv").write_text("row,val\n1,a\n")

    loaded = _load_cached_dataframe(
        str(tmp_path / "ratings_cleaned.csv"),
        task_name="Rate",
    )

    assert loaded.to_dict(orient="records") == [
        {"row": 1, "val": "a"},
        {"row": 2, "val": "b"},
        {"row": 10, "val": "j"},
    ]


def test_load_cached_dataframe_prefers_primary_csv_over_split_files(tmp_path: Path) -> None:
    (tmp_path / "ratings_cleaned.csv").write_text("row,val\n0,primary\n")
    (tmp_path / "ratings_cleaned_1.csv").write_text("row,val\n1,split\n")

    loaded = _load_cached_dataframe(
        str(tmp_path / "ratings_cleaned.csv"),
        task_name="Rate",
    )

    assert loaded.to_dict(orient="records") == [{"row": 0, "val": "primary"}]


def test_load_cached_dataframe_after_dynamic_split_fallback(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame({"row": range(20_005), "val": ["x"] * 20_005})
    target = tmp_path / "ratings_cleaned.csv"

    original_to_csv = pd.DataFrame.to_csv

    def size_sensitive_to_csv(self, path_or_buf=None, *args, **kwargs):
        path_text = str(path_or_buf) if path_or_buf is not None else ""
        if path_or_buf is not None and path_text == str(target):
            raise OSError("simulated full save failure")
        if path_text.endswith("_1.csv") and len(self) > 10_000:
            raise OSError("simulated 100k fallback failure")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", size_sensitive_to_csv)

    saved_paths = save_dataframe_with_fallback(df, str(target), label="Rate")
    assert saved_paths == [
        str(tmp_path / "ratings_cleaned_1.csv"),
        str(tmp_path / "ratings_cleaned_2.csv"),
        str(tmp_path / "ratings_cleaned_3.csv"),
    ]

    loaded = _load_cached_dataframe(str(target), task_name="Rate")
    assert len(loaded) == len(df)
    assert loaded.iloc[0]["row"] == 0
    assert loaded.iloc[-1]["row"] == 20_004
