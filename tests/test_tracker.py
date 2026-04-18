import unittest
import sys
from pathlib import Path

# Ana dizine yol ekleme
sys.path.append(str(Path(__file__).parent.parent / "model-training"))

from trackers.deepsort_tracker import DeepSortTracker

class TestTracker(unittest.TestCase):
    def test_tracker_initialization(self):
        tracker = DeepSortTracker(max_age=30, max_cosine=0.2, max_spatial_dist=100)
        self.assertIsNotNone(tracker)
        # Sadece geometri odaklı tracking'i test edebiliriz

    def test_tracker_update_empty(self):
        tracker = DeepSortTracker(max_age=30, max_cosine=0.2, max_spatial_dist=100)
        tracks = tracker.update([], features=None, frame_id=1)
        self.assertEqual(len(tracks), 0)

if __name__ == "__main__":
    unittest.main()
