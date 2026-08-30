from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "ui" / "app.js"
INDEX_HTML = ROOT / "ui" / "index.html"


class _CloseButtonParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.close_buttons: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "button" and values.get("aria-label") == "닫기":
            self.close_buttons.append(values)


class OndammUiRegressionTests(unittest.TestCase):
    def test_async_handlers_do_not_dereference_current_target_after_await(self) -> None:
        script = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn("event.currentTarget.reset()", script)
        self.assertNotIn("event.currentTarget.elements", script)
        for form_id in (
            "create-form",
            "session-form",
            "facial-profile-form",
            "local-rag-form",
            "camera-consent-form",
        ):
            pattern = re.compile(
                rf'\$\("#{re.escape(form_id)}"\)\.addEventListener\("submit", async \(event\) => \{{'
                rf'[\s\S]*?const form = event\.currentTarget;',
            )
            self.assertRegex(script, pattern)

    def test_successful_mutations_have_separate_refresh_error_copy(self) -> None:
        script = APP_JS.read_text(encoding="utf-8")
        self.assertIn("refreshAfterSuccessfulMutation", script)
        self.assertIn("변경 사항은 저장됐지만 화면 갱신에 실패했습니다", script)

    def test_dialog_close_icons_are_non_submit_buttons(self) -> None:
        parser = _CloseButtonParser()
        parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(parser.close_buttons), 4)
        for button in parser.close_buttons:
            with self.subTest(button=button):
                self.assertEqual(button.get("type"), "button")
                self.assertIn("data-close-dialog", button)


if __name__ == "__main__":
    unittest.main()
