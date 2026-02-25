import asyncio
import time
from pathlib import Path

import pandas as pd

from gabriel.utils import openai_utils


def test_resume_uses_checkpoint_timing_for_dynamic_timeout(tmp_path: Path) -> None:
    save_path = tmp_path / "responses.csv"
    checkpoint_rows = []
    for idx in range(5):
        checkpoint_rows.append(
            {
                "Identifier": f"id{idx}",
                "Response": "ok",
                "Web Search Sources": "[]",
                "Time Taken": 0.12,
                "Input Tokens": 1,
                "Reasoning Tokens": 0,
                "Output Tokens": 1,
                "Reasoning Effort": "medium",
                "Successful": True,
                "Error Log": "[]",
            }
        )
    checkpoint_rows.append(
        {
            "Identifier": "id5",
            "Response": "",
            "Web Search Sources": "[]",
            "Time Taken": None,
            "Input Tokens": 1,
            "Reasoning Tokens": 0,
            "Output Tokens": 1,
            "Reasoning Effort": "medium",
            "Successful": False,
            "Error Log": "[]",
        }
    )
    pd.DataFrame(checkpoint_rows).to_csv(save_path, index=False)

    async def slow_responder(prompt: str, **_: object):
        await asyncio.sleep(1.5)
        return [f"late-{prompt}"], 1.5, []

    start = time.time()
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"prompt-{idx}" for idx in range(6)],
            identifiers=[f"id{idx}" for idx in range(6)],
            response_fn=slow_responder,
            use_dummy=False,
            save_path=str(save_path),
            reset_files=False,
            max_retries=1,
            dynamic_timeout=True,
            timeout_factor=1.0,
            max_timeout=0.25,
            n_parallels=4,
            ramp_up_seconds=0,
            manage_rate_limits=False,
            status_report_interval=None,
            logging_level="error",
        )
    )
    elapsed = time.time() - start

    pending_row = df.loc[df["Identifier"] == "id5"].iloc[0]
    assert not bool(pending_row["Successful"])
    error_log_text = str(pending_row.get("Error Log", "")).lower()
    assert ("timed out" in error_log_text) or (error_log_text in {"none", "nan", ""})
    assert elapsed < 1.4
