import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt

from constants import MAX_FRAME_LENGTH, MIC_ID


def add_noise(audio_data):
    # Gauss
    noise_amp = 0.005 * np.random.uniform() * np.amax(audio_data)
    audio_data = audio_data + noise_amp * np.random.normal(size=audio_data.shape)
    return audio_data


class AudioRecorder:
    def __init__(self, duration, samplerate):
        self.duration = duration
        self.samplerate = samplerate

    def record_audio(self):
        print("Rec...")
        audio_data = sd.rec(int(self.samplerate * self.duration), samplerate=self.samplerate, channels=1,
                            dtype='float64', device=MIC_ID)
        sd.wait()
        print("End rec.")

        audio_data_with_noise = add_noise(audio_data)
        return audio_data_with_noise


class FeatureExtractor:
    def extract_features(self, filename, plot=True):
        y, sr = librosa.load(filename)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        combined_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)

        if plot:
            librosa.display.specshow(combined_mfccs, x_axis='time')
            plt.colorbar()
            plt.title('MFCC with Deltas')
            plt.tight_layout()
            plt.show()

        return combined_mfccs

    def pad_mfcc(self, mfcc):
        if mfcc.shape[1] < MAX_FRAME_LENGTH:
            return np.pad(mfcc, ((0, 0), (0, MAX_FRAME_LENGTH - mfcc.shape[1])))
        return mfcc[:, :MAX_FRAME_LENGTH]

    def extract_mfcc_features(self, signal, sr):
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        return self.pad_mfcc(mfccs).T


feature_extractor = FeatureExtractor()
