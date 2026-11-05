"""Synthetic checks for CODEC-3's bank-3 paragraph rule (no text fetched)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec3_token_bank import paragraphs  # noqa: E402

START = "*** START OF THE PROJECT GUTENBERG EBOOK NEW GRUB STREET ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK NEW GRUB STREET ***"


def book(*body: str) -> str:
    return "\r\n".join(["licence header, not the body", START, *body, END, "licence footer"])


CONTENTS_THEN_BODY = book(
    "NEW GRUB STREET",
    "",
    "CONTENTS",
    "",
    "CHAPTER I. A MAN OF HIS DAY",
    "CHAPTER II. HOUSE OF TROUBLE",
    "",
    "VOLUME I",
    "",
    "CHAPTER I. A MAN OF HIS DAY",
    "",
    "First paragraph, line one,",
    "line two.  Two spaces kept.",
    "",
    "* * *",
    "",
    "Second paragraph mentions chapter one in passing.",
    "",
    "CHAPTER II.",
    "",
    "HOUSE OF TROUBLE",
    "",
    "Third paragraph.",
)


class ParagraphRuleTests(unittest.TestCase):
    def test_the_text_starts_at_the_last_chapter_one_heading_not_the_contents(self):
        self.assertEqual(paragraphs(CONTENTS_THEN_BODY), [
            "First paragraph, line one, line two.  Two spaces kept.",
            "Second paragraph mentions chapter one in passing.",
            "Third paragraph.",
        ])

    def test_arabic_numerals_and_case_are_headings_too(self):
        text = book("Chapter 1", "", "Body one.", "", "chapter 2", "", "Body two.")
        self.assertEqual(paragraphs(text), ["Body one.", "Body two."])

    def test_nothing_outside_the_markers_is_body(self):
        text = book("CHAPTER I", "", "Only paragraph.")
        self.assertEqual(paragraphs(text), ["Only paragraph."])

    def test_a_missing_marker_or_chapter_one_refuses(self):
        for text, rule in [
            ("CHAPTER I\n\nBody.\n" + END, "rule 1"),
            (START + "\nCHAPTER I\n\nBody.", "rule 1"),
            (book("CHAPTER II", "", "Body."), "rule 3"),
        ]:
            with self.assertRaises(SystemExit) as refused:
                paragraphs(text)
            self.assertIn(rule, str(refused.exception))


if __name__ == "__main__":
    unittest.main()
