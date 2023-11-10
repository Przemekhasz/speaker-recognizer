import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from audio_processor import AudioProcessor


class TestAudioProcessor(unittest.TestCase):
    @patch('librosa.load')
    @patch('audio_utils.feature_extractor.extract_mfcc_features')
    @patch('numpy.corrcoef')
    @patch('soundfile.write')
    def test_find_and_save_sample_in_group(self, mock_write, mock_corrcoef, mock_extract_mfcc, mock_load):
        processor = AudioProcessor()

        # mock the librosa.load function to return dummy data
        mock_load.return_value = (np.array([0.1, 0.2, 0.3]), 44100)

        # feature_extractor to return dummy MFCC features
        dummy_mfcc = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_extract_mfcc.return_value = dummy_mfcc

        # m the numpy.corrcoef function to return dummy correlations
        mock_corrcoef.return_value = np.array([[1, 0.9], [0.9, 1]])

        sample_filename = 'sample.wav'
        group_filename = 'group.wav'
        result_filename = 'result.wav'

        processor.find_and_save_sample_in_group(sample_filename, group_filename, result_filename)

        mock_load.assert_any_call(sample_filename, sr=None)
        mock_load.assert_any_call(group_filename, sr=None)
        mock_extract_mfcc.assert_called()
        mock_corrcoef.assert_called()

        # Check if soundfile.write was called correctly
        args, _ = mock_write.call_args
        self.assertEqual(args[0], result_filename)
        np.testing.assert_array_almost_equal(args[1], np.array([0.1, 0.2, 0.3]))
        self.assertEqual(args[2], 44100)


if __name__ == '__main__':
    unittest.main()
