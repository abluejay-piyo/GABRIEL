import asyncio
import pandas as pd

import gabriel.tasks.deduplicate as deduplicate
from gabriel.tasks.deduplicate import Deduplicate, DeduplicateConfig


async def _run_dedup(tmp_path, n_runs=1):
    cfg = DeduplicateConfig(save_dir=str(tmp_path), use_dummy=True, use_embeddings=False, n_runs=n_runs)
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["apple", "Apple", "banana", "BANANA", "pear"]})
    return await task.run(df, column_name="term")


def test_deduplicate_dummy(tmp_path):
    result = asyncio.run(_run_dedup(tmp_path, n_runs=1))
    assert "mapped_term" in result.columns
    assert result["mapped_term"].tolist() == ["apple", "apple", "banana", "banana", "pear"]


def test_deduplicate_multiple_runs(tmp_path):
    result = asyncio.run(_run_dedup(tmp_path, n_runs=2))
    assert "mapped_term_run1" in result.columns
    assert "mapped_term_final" in result.columns
    assert "mapped_term" in result.columns
    assert result["mapped_term"].tolist() == ["apple", "apple", "banana", "banana", "pear"]


def test_response_files_unique(tmp_path, monkeypatch):
    captured_paths = []

    async def fake_get_all_responses(*, save_path, **kwargs):
        captured_paths.append(save_path)
        return pd.DataFrame({"Identifier": [], "Response": []})

    monkeypatch.setattr(deduplicate, "get_all_responses", fake_get_all_responses)

    cfg = DeduplicateConfig(save_dir=str(tmp_path), use_dummy=True, use_embeddings=False, n_runs=2)
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["apple", "Apple"]})
    asyncio.run(task.run(df, column_name="term"))

    expected1 = tmp_path / "deduplicate_responses_run1.csv"
    expected2 = tmp_path / "deduplicate_responses_run2.csv"
    assert captured_paths == [str(expected1), str(expected2)]


def test_prompt_contains_terms_with_embeddings(monkeypatch, tmp_path):
    captured = {}

    async def fake_get_all_responses(*, prompts, identifiers, **kwargs):
        captured["prompts"] = prompts
        return pd.DataFrame({"Identifier": identifiers, "Response": ["{}"] * len(identifiers)})

    monkeypatch.setattr(deduplicate, "get_all_responses", fake_get_all_responses)

    cfg = DeduplicateConfig(
        save_dir=str(tmp_path),
        use_dummy=True,
        use_embeddings=True,
        group_size=2,
        n_runs=1,
    )
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["apple", "Apple", "banana", "BANANA", "pear"]})
    asyncio.run(task.run(df, column_name="term"))

    assert "prompts" in captured
    prompt = captured["prompts"][0]
    assert "BEGIN RAW TERMS" in prompt
    assert "END RAW TERMS" in prompt
    body = prompt.split("BEGIN RAW TERMS", 1)[1].split("END RAW TERMS", 1)[0].strip()
    assert body != ""


def test_mapping_dict_parsed(monkeypatch, tmp_path):
    async def fake_get_all_responses(*, prompts, identifiers, **kwargs):
        mapping = '{"red apples": ["red apples", "red apple"]}'
        return pd.DataFrame({"Identifier": identifiers, "Response": [mapping]})

    monkeypatch.setattr(deduplicate, "get_all_responses", fake_get_all_responses)

    cfg = DeduplicateConfig(save_dir=str(tmp_path), use_dummy=True, use_embeddings=False, n_runs=1)
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["red apple", "red apples", "banana"]})
    result = asyncio.run(task.run(df, column_name="term"))
    assert result["mapped_term"].tolist() == ["red apples", "red apples", "banana"]


def test_deduplicate_embedding_overrides_routed(monkeypatch, tmp_path):
    captured = {}

    async def fake_get_all_embeddings(*, identifiers, embedding_fn=None, get_all_embeddings_fn=None, **kwargs):
        captured["embedding_fn"] = embedding_fn
        captured["get_all_embeddings_fn"] = get_all_embeddings_fn
        captured["embedding_kwargs"] = kwargs
        return {
            ident: [float(idx + 1), float((idx + 1) * 2)]
            for idx, ident in enumerate(identifiers)
        }

    async def fake_get_all_responses(*, identifiers, **kwargs):
        captured["response_kwargs"] = kwargs
        return pd.DataFrame({"Identifier": identifiers, "Response": ["{}"] * len(identifiers)})

    monkeypatch.setattr(deduplicate, "get_all_embeddings", fake_get_all_embeddings)
    monkeypatch.setattr(deduplicate, "get_all_responses", fake_get_all_responses)

    async def custom_embedding(text: str):
        return [float(len(text))]

    async def custom_embedding_driver(texts, identifiers):
        return {ident: [float(i)] for i, ident in enumerate(identifiers)}

    cfg = DeduplicateConfig(
        save_dir=str(tmp_path),
        use_dummy=True,
        use_embeddings=True,
        group_size=2,
        n_runs=1,
    )
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["apple", "Apple", "banana", "BANANA", "pear"]})
    asyncio.run(
        task.run(
            df,
            column_name="term",
            embedding_fn=custom_embedding,
            get_all_embeddings_fn=custom_embedding_driver,
            response_fn=custom_embedding,
        )
    )

    assert captured["embedding_fn"] is custom_embedding
    assert captured["get_all_embeddings_fn"] is custom_embedding_driver
    assert "embedding_fn" not in captured["response_kwargs"]
    assert "get_all_embeddings_fn" not in captured["response_kwargs"]
    assert captured["response_kwargs"]["response_fn"] is custom_embedding
