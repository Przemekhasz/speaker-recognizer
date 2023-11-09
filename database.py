import numpy as np

from utils import SingletonMeta


class VoiceDatabase(metaclass=SingletonMeta):
    def __init__(self):
        self.voice_database = {}

    def add_voice(self, user, features):
        self.voice_database[user] = features

    def get_voice(self, user):
        return self.voice_database.get(user, None)

    def compare_features(self, mfcc1, mfcc2):
        correlation = np.correlate(mfcc1.mean(axis=1), mfcc2.mean(axis=1))
        threshold = 0.8
        return correlation[0] > threshold


voice_database = VoiceDatabase()