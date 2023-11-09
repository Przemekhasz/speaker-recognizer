import os
import numpy as np
from audio_utils import feature_extractor


class DataLoader:
    def load_data_from_directory(self, directory_path):
        features = []
        labels = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".wav"):
                full_path = os.path.join(directory_path, filename)
                label = filename.split('.')[0]
                mfccs = feature_extractor.extract_features(full_path)
                features.append(mfccs)
                labels.append(label)
        return np.array(features), labels


data_loader = DataLoader()
