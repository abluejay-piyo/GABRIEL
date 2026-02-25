import asyncio
import pandas as pd
from gabriel.utils import openai_utils


def test_get_all_responses_reset_files(tmp_path):
    save_path = tmp_path / "out.csv"
    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a", "b"],
            identifiers=["1", "2"],
            save_path=str(save_path),
            use_dummy=True,
        )
    )
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["b"],
            identifiers=["2"],
            save_path=str(save_path),
            use_dummy=True,
            reset_files=True,
        )
    )
    assert set(df["Identifier"]) == {"2"}


def test_resume_treats_string_success_values_as_completed(tmp_path):
    save_path = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "Identifier": ["1", "2", "3"],
            "Response": ["[]", "[]", "[]"],
            "Web Search Sources": ["[]", "[]", "[]"],
            "Time Taken": [0.1, 0.1, 0.1],
            "Input Tokens": [1, 1, 1],
            "Reasoning Tokens": [0, 0, 0],
            "Output Tokens": [1, 1, 1],
            "Reasoning Effort": ["default", "default", "default"],
            "Successful": ["True", "true", "1"],
            "Error Log": ["[]", "[]", "[]"],
            "Response IDs": ["[]", "[]", "[]"],
            "Reasoning Summary": ["", "", ""],
        }
    ).to_csv(save_path, index=False)

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a", "b", "c"],
            identifiers=["1", "2", "3"],
            save_path=str(save_path),
            use_dummy=True,
            reset_files=False,
        )
    )

    assert len(df) == 3


def test_resume_skip_tail_fails_returns_checkpoint_without_retry(tmp_path, capsys):
    save_path = tmp_path / "out.csv"
    total_rows = 5_001
    identifiers = [str(i) for i in range(total_rows)]
    successful = [True] * total_rows
    successful[-1] = False
    pd.DataFrame(
        {
            "Identifier": identifiers,
            "Response": [openai_utils._ser(["cached"])] * total_rows,
            "Web Search Sources": [openai_utils._ser([])] * total_rows,
            "Time Taken": [0.1] * total_rows,
            "Input Tokens": [1] * total_rows,
            "Reasoning Tokens": [0] * total_rows,
            "Output Tokens": [1] * total_rows,
            "Reasoning Effort": ["default"] * total_rows,
            "Successful": successful,
            "Error Log": [openai_utils._ser([])] * total_rows,
        }
    ).to_csv(save_path, index=False)

    async def should_not_retry(prompt: str, **kwargs):
        raise AssertionError("Tail failures should have been skipped.")

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["prompt"] * total_rows,
            identifiers=identifiers,
            save_path=str(save_path),
            response_fn=should_not_retry,
            verbose=False,
        )
    )

    output = capsys.readouterr().out
    assert "skip_tail_fails=False" in output
    assert "1/5,001" in output
    assert len(df) == total_rows
    success_mask = df["Successful"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    assert int(success_mask.sum()) == total_rows - 1


def test_resume_skip_tail_fails_false_retries_incomplete_rows(tmp_path):
    save_path = tmp_path / "out.csv"
    total_rows = 5_001
    identifiers = [str(i) for i in range(total_rows)]
    successful = [True] * total_rows
    successful[-1] = False
    pd.DataFrame(
        {
            "Identifier": identifiers,
            "Response": [openai_utils._ser(["cached"])] * total_rows,
            "Web Search Sources": [openai_utils._ser([])] * total_rows,
            "Time Taken": [0.1] * total_rows,
            "Input Tokens": [1] * total_rows,
            "Reasoning Tokens": [0] * total_rows,
            "Output Tokens": [1] * total_rows,
            "Reasoning Effort": ["default"] * total_rows,
            "Successful": successful,
            "Error Log": [openai_utils._ser([])] * total_rows,
        }
    ).to_csv(save_path, index=False)

    calls = []

    async def responder(prompt: str, **kwargs):
        calls.append(prompt)
        return ["retried"]

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["prompt"] * total_rows,
            identifiers=identifiers,
            save_path=str(save_path),
            response_fn=responder,
            skip_tail_fails=False,
            verbose=False,
        )
    )

    assert calls == ["prompt"]
