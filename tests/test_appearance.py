import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent.appearance import AppearanceStore


class AppearanceStoreTests(unittest.TestCase):
    def test_preferences_persist(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "appearance.json"
            store = AppearanceStore(path)
            store.update("theme", "nord")
            store.update("contrast", "on")
            store.update("motion", "reduced")
            store.update("density", "compact")

            restored = AppearanceStore(path)

            self.assertEqual(restored.settings.theme, "nord")
            self.assertTrue(restored.settings.high_contrast)
            self.assertTrue(restored.settings.reduced_motion)
            self.assertEqual(restored.settings.density, "compact")
            self.assertIn("High contrast: on", restored.report())

    def test_invalid_updates_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = AppearanceStore(Path(directory) / "appearance.json")
            for name, value in (
                ("contrast", "maybe"),
                ("motion", "slow"),
                ("density", "tiny"),
                ("unknown", "on"),
            ):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        store.update(name, value)

    def test_invalid_file_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "appearance.json"
            path.write_text(
                json.dumps({"theme": "nord", "high_contrast": "yes"}),
                encoding="utf-8",
            )

            store = AppearanceStore(path)

            self.assertEqual(store.settings.theme, "textual-dark")
            self.assertIsNotNone(store.load_error)


if __name__ == "__main__":
    unittest.main()
