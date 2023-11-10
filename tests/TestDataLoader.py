import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    @patch('os.listdir')
    @patch('os.path.join')
    @patch('audio_utils.feature_extractor.extract_features')
    def test_load_data_from_directory(self, mock_extract_features, mock_join, mock_listdir):
        mock_listdir.return_value = ['sample1.wav', 'sample2.wav', 'not_audio.txt']
        mock_join.side_effect = lambda *args: '/'.join(arg.decode('utf-8') if isinstance(arg, bytes) else arg for arg in args)

        dummy_mfccs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_extract_features.return_value = dummy_mfccs

        loader = DataLoader()
        directory_path = 'audio_training_data'
        features, labels = loader.load_data_from_directory(directory_path)

        mock_listdir.assert_called_with(directory_path)
        self.assertEqual(mock_join.call_count, 2)  # Only called for .wav files
        mock_extract_features.assert_called()
        self.assertEqual(len(features), 2)
        self.assertEqual(len(labels), 2)
        self.assertListEqual(labels, ['sample1', 'sample2'])
        self.assertEqual(features.shape, (2, dummy_mfccs.shape[0], dummy_mfccs.shape[1]))
        for feature_array, expected_feature in zip(features, [dummy_mfccs, dummy_mfccs]):
            for feature, expected in zip(feature_array, expected_feature):
                self.assertTrue(np.allclose(feature, expected))


if __name__ == '__main__':
    unittest.main()
