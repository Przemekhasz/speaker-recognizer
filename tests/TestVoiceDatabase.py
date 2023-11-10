import unittest
import numpy as np

from database import VoiceDatabase


class TestVoiceDatabase(unittest.TestCase):
    def setUp(self):
        # reset the voice database for each test
        VoiceDatabase._instances = {}

    def test_add_and_get_voice(self):
        db = VoiceDatabase()

        # dummy voice features
        features = np.array([[0.1, 0.2], [0.3, 0.4]])

        # test add_voice
        db.add_voice("user1", features)
        self.assertIn("user1", db.voice_database)

        # test get_voice
        retrieved_features = db.get_voice("user1")
        np.testing.assert_array_equal(retrieved_features, features)

        # test get_voice for non-existent user
        self.assertIsNone(db.get_voice("non_existent_user"))

    def test_compare_features(self):
        db = VoiceDatabase()

        # voice features
        features1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        features2 = np.array([[0.1, 0.2], [0.3, 0.4]])
        features3 = np.array([[0.9, 0.8], [0.7, 0.6]])

        # calculate and print the correlation for debugging
        correlation = np.correlate(features1.mean(axis=1), features2.mean(axis=1))[0]
        print("Correlation:", correlation)

        # test compare_features for similar features
        self.assertTrue(db.compare_features(features1, features2))

        # test compare_features for different features
        self.assertFalse(db.compare_features(features1, features3))


if __name__ == '__main__':
    unittest.main()
