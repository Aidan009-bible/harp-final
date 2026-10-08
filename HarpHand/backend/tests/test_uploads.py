import unittest

from core.uploads import UploadValidationError, safe_upload_name


class UploadNameTests(unittest.TestCase):
    def test_strips_path_components_and_unsafe_characters(self):
        self.assertEqual(
            safe_upload_name(
                "../../performance final?.mp4",
                fallback="video.mp4",
                allowed_suffixes={".mp4"},
            ),
            "performance_final_.mp4",
        )

    def test_rejects_wrong_suffix(self):
        with self.assertRaises(UploadValidationError):
            safe_upload_name(
                "weights.exe",
                fallback="weights.pt",
                allowed_suffixes={".pt"},
            )


if __name__ == "__main__":
    unittest.main()
