from pathlib import Path


from gabriel.api import _load_cached_dataframe


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
