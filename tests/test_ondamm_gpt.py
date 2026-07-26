from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_gpt import OpenAIFrameReviewer, extract_video_frame_data_urls  # noqa: E402


class OndammGptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ondamm-gpt-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_extracts_bounded_jpeg_frames_instead_of_uploading_whole_video(self) -> None:
        import cv2
        import numpy as np

        video = self.temp_dir / "clip.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (96, 64))
        for index in range(8):
            writer.write(np.full((64, 96, 3), 20 + index * 10, dtype=np.uint8))
        writer.release()

        frames = extract_video_frame_data_urls(video, max_frames=3, max_dimension=80)

        self.assertEqual(len(frames), 3)
        self.assertTrue(all(frame.startswith("data:image/jpeg;base64,") for frame in frames))

    def test_openai_reviewer_uses_responses_api_and_safety_prompt(self) -> None:
        captured = {}

        def fake_transport(url, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return {
                "id": "resp-test",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "관찰 가능한 움직임만 요약했습니다."}],
                    }
                ],
            }

        reviewer = OpenAIFrameReviewer(
            api_key="test-key",
            model="gpt-5.6",
            transport=fake_transport,
        )
        result = reviewer.review(
            frame_data_urls=["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"],
            event_metadata={"event_type": "gaze_diverted"},
        )

        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(captured["payload"]["model"], "gpt-5.6")
        content = captured["payload"]["input"][0]["content"]
        self.assertEqual(sum(item["type"] == "input_image" for item in content), 2)
        self.assertIn("감정", content[0]["text"])
        self.assertIn("진단", content[0]["text"])
        self.assertEqual(result["review_text"], "관찰 가능한 움직임만 요약했습니다.")
        self.assertEqual(result["remote_frame_count"], 2)
        self.assertFalse(result["dossier_auto_updated"])


if __name__ == "__main__":
    unittest.main()
