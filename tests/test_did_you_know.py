"""Bounded research tests for the category-driven video formats."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))
did_you_know = importlib.import_module("did_you_know")


class DidYouKnowResearchTests(unittest.TestCase):
    def test_category_selection_rejects_unknown_values(self) -> None:
        self.assertEqual(
            did_you_know.category_selection("science", "space"),
            ("Science", "Space and astronomy", "space astronomy"),
        )
        with self.assertRaises(ValueError):
            did_you_know.category_selection("made-up", "space")
        with self.assertRaises(ValueError):
            did_you_know.category_selection("science", "made-up")
        self.assertIn("human_relations", did_you_know.CATEGORIES)
        self.assertIn("politics", did_you_know.CATEGORIES)
        self.assertIn("romantic_relationships", did_you_know.CATEGORIES)
        self.assertEqual(did_you_know.CATEGORIES["bible_stories"]["subcategories"]["book_of_enoch"]["label"], "The Book of Enoch")
        self.assertEqual(
            did_you_know.CATEGORIES["bible_stories"]["subcategories"]["angels"]["label"],
            "Angels (The Heavenly, the Fallen, and the Holy Ones Among Us)",
        )
        self.assertIn("marriages", did_you_know.CATEGORIES)
        self.assertIn("marriage_relationship_dynamics", did_you_know.CATEGORIES)
        self.assertIn("business", did_you_know.CATEGORIES)

    def test_research_is_limited_to_selected_topic_and_uses_safe_http_settings(self) -> None:
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "query": {"pages": [{
                "title": "Astronomy", "fullurl": "https://en.wikipedia.org/wiki/Astronomy",
                "extract": "Astronomy is a natural science that studies celestial objects and phenomena. " * 5,
            }]}
        }
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False
        session.get.return_value = response
        with patch.object(did_you_know.requests, "Session", return_value=session):
            sources = did_you_know.fetch_research_sources("science", "space")

        self.assertFalse(session.trust_env)
        call = session.get.call_args
        self.assertEqual(call.args, (did_you_know.RESEARCH_API,))
        self.assertEqual(call.kwargs["params"]["gsrsearch"], "space astronomy facts")
        self.assertFalse(call.kwargs["allow_redirects"])
        self.assertEqual(call.kwargs["timeout"], (10, 25))
        self.assertEqual(sources[0]["title"], "Astronomy")
        self.assertEqual(sources[0]["license"], "CC BY-SA 4.0")

    def test_cancelled_research_does_not_make_a_request(self) -> None:
        with patch.object(did_you_know.requests, "Session") as session:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                did_you_know.fetch_research_sources("science", "space", lambda: True)
        session.assert_not_called()
