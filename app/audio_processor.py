import librosa
import numpy as np
import soundfile as sf

from app.audio_utils import feature_extractor


class AudioProcessor:
    def find_and_save_sample_in_group(self, sample_filename, group_filename, result_filename):
        y_sample, sr_sample = librosa.load(sample_filename, sr=None)
        y_group, sr_group = librosa.load(group_filename, sr=None)

        mfcc_sample = feature_extractor.extract_mfcc_features(y_sample, sr_sample)
        mfcc_group = feature_extractor.extract_mfcc_features(y_group, sr_group)

        sample_length = len(mfcc_sample)
        group_length = len(mfcc_group)
        correlation_range = group_length - sample_length + 1

        correlations = [np.corrcoef(mfcc_group[i:i + sample_length].ravel(), mfcc_sample.ravel())[0, 1]
                        for i in range(correlation_range)]

        max_correlation_index = np.argmax(correlations)
        start_point = max_correlation_index * sr_group // sr_sample
        end_point = start_point + len(y_sample)

        result_signal = y_group[start_point:end_point]

        sf.write(result_filename, result_signal, int(sr_group))


audio_processor = AudioProcessor()