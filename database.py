import numpy as np
from scipy.spatial.distance import euclidean

from utils import SingletonMeta


class VoiceDatabase(metaclass=SingletonMeta):
    """
        Voice Database for storing and comparing voice features.

        References:
        - MFCC explanation: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
        - Voice feature comparison techniques: https://ieeexplore.ieee.org/document/1163055
    """
    def __init__(self):
        self.voice_database = {}

    def add_voice(self, user, features):
        self.voice_database[user] = features

    def get_voice(self, user):
        return self.voice_database.get(user, None)

    def compare_features(self, mfcc1, mfcc2):
        correlation = np.correlate(mfcc1.mean(axis=1), mfcc2.mean(axis=1))
        distance = euclidean(mfcc1.flatten(), mfcc2.flatten())
        threshold = 0.8
        return distance < threshold


voice_database = VoiceDatabase()