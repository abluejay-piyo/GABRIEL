import asyncio

import pytest

import gabriel.utils.openai_utils as openai_utils
from gabriel.utils.openai_utils import get_all_embeddings


def test_get_all_embeddings_uses_cache(tmp_path, capsys):
    save_path = tmp_path / "emb.pkl"
    texts = ["a", "b"]

    asyncio.run(
        get_all_embeddings(
            texts=texts,
            identifiers=texts,
            save_path=str(save_path),
            use_dummy=True,
            verbose=False,
        )
    )

    capsys.readouterr()

    asyncio.run(
        get_all_embeddings(
            texts=texts,
            identifiers=texts,
            save_path=str(save_path),
            use_dummy=True,
            verbose=False,
        )
    )

    captured = capsys.readouterr()
    out = captured.out
    assert "Loaded 2 existing embeddings" in out
    assert "Using cached embeddings" in out


def test_get_all_embeddings_dummy_default(tmp_path):
    save_path = tmp_path / "emb.pkl"
    texts = ["hello", "world"]
    result = asyncio.run(
        get_all_embeddings(
            texts=texts,
            identifiers=["1", "2"],
            save_path=str(save_path),
            use_dummy=True,
            verbose=False,
        )
    )
    assert set(result.keys()) == {"1", "2"}
    assert all(isinstance(vec, list) and vec for vec in result.values())


def test_get_all_embeddings_dummy_override(tmp_path):
    save_path = tmp_path / "emb.pkl"
    overrides = {
        "first": [0.1, 0.2, 0.3],
        "*": [9.0, 8.0, 7.0],
    }
    result = asyncio.run(
        get_all_embeddings(
            texts=["a", "b"],
            identifiers=["first", "second"],
            save_path=str(save_path),
            use_dummy=True,
            dummy_embeddings=overrides,
            verbose=False,
        )
    )
    assert result["first"] == [0.1, 0.2, 0.3]
    assert result["second"] == [9.0, 8.0, 7.0]


def test_get_all_embeddings_custom_callable(tmp_path):
    calls = []

    async def custom(text: str, *, model: str, timeout: float):
        calls.append((text, model, timeout))
        return [float(len(text)), 1.0]

    result = asyncio.run(
        get_all_embeddings(
            texts=["aa", "bbbb"],
            identifiers=["first", "second"],
            save_path=str(tmp_path / "custom.pkl"),
            timeout=12.0,
            embedding_fn=custom,
            verbose=False,
        )
    )

    assert calls == [
        ("aa", "text-embedding-3-small", 12.0),
        ("bbbb", "text-embedding-3-small", 12.0),
    ]
    assert result["first"] == [2.0, 1.0]
    assert result["second"] == [4.0, 1.0]


def test_get_all_embeddings_custom_callable_requires_text(tmp_path):
    async def missing_text():
        return [1.0]

    with pytest.raises(TypeError, match="must accept a `text` argument"):
        asyncio.run(
            get_all_embeddings(
                texts=["hello"],
                identifiers=["row-1"],
                save_path=str(tmp_path / "missing_text.pkl"),
                embedding_fn=missing_text,  # type: ignore[arg-type]
                verbose=False,
            )
        )


def test_get_all_embeddings_custom_driver_receives_kwargs(tmp_path):
    calls = []

    async def custom_driver(texts, identifiers, model=None, extra=None, **kwargs):
        calls.append(
            {
                "texts": texts,
                "identifiers": identifiers,
                "model": model,
                "extra": extra,
                "kwargs": kwargs,
            }
        )
        return {ident: [float(idx)] for idx, ident in enumerate(identifiers)}

    save_path = str(tmp_path / "custom_driver.pkl")
    result = asyncio.run(
        get_all_embeddings(
            texts=["alpha", "beta"],
            identifiers=None,
            model="embedding-model",
            extra="value",
            get_all_embeddings_fn=custom_driver,
            save_path=save_path,
            verbose=False,
        )
    )

    assert calls and calls[0]["texts"] == ["alpha", "beta"]
    assert calls[0]["identifiers"] == ["alpha", "beta"]
    assert calls[0]["model"] == "embedding-model"
    assert calls[0]["extra"] == "value"
    assert calls[0]["kwargs"]["save_path"] == save_path
    assert result == {"alpha": [0.0], "beta": [1.0]}


def test_get_all_embeddings_custom_driver_requires_identifiers(tmp_path):
    async def missing_identifiers(texts):
        return {text: [1.0] for text in texts}

    with pytest.raises(TypeError, match="identifiers"):
        asyncio.run(
            get_all_embeddings(
                texts=["hello"],
                identifiers=["row-1"],
                get_all_embeddings_fn=missing_identifiers,  # type: ignore[arg-type]
                save_path=str(tmp_path / "missing_identifiers.pkl"),
                verbose=False,
            )
        )


def test_get_all_embeddings_split_checkpoint_fallback_on_save_failure(tmp_path, monkeypatch):
    save_path = tmp_path / "emb.pkl"
    original_dump = openai_utils.pickle.dump

    def flaky_dump(obj, file_obj, *args, **kwargs):
        if getattr(file_obj, "name", None) == str(save_path):
            raise OSError("simulated primary pickle failure")
        return original_dump(obj, file_obj, *args, **kwargs)

    monkeypatch.setattr(openai_utils.pickle, "dump", flaky_dump)

    result = asyncio.run(
        get_all_embeddings(
            texts=["alpha", "beta"],
            identifiers=["a", "b"],
            save_path=str(save_path),
            use_dummy=True,
            verbose=False,
        )
    )

    assert result["a"]
    split_parts = sorted(tmp_path.glob("emb_*.pkl"))
    assert split_parts


def test_get_all_embeddings_loads_split_checkpoint_when_primary_missing(tmp_path, capsys):
    save_path = tmp_path / "emb.pkl"
    split_one = tmp_path / "emb_1.pkl"
    split_two = tmp_path / "emb_2.pkl"

    with open(split_one, "wb") as f:
        openai_utils.pickle.dump({"x": [1.0]}, f)
    with open(split_two, "wb") as f:
        openai_utils.pickle.dump({"y": [2.0]}, f)

    result = asyncio.run(
        get_all_embeddings(
            texts=["x", "y"],
            identifiers=["x", "y"],
            save_path=str(save_path),
            use_dummy=True,
            verbose=False,
        )
    )

    captured = capsys.readouterr().out
    assert "Loaded 2 existing embeddings from split files" in captured
    assert result == {"x": [1.0], "y": [2.0]}
