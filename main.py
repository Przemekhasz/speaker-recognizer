import numpy as np
from scipy.io.wavfile import write
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from utils import find_microphone_id, generate_filename
from audio_utils import AudioRecorder, feature_extractor
from database import voice_database
from data_loader import data_loader
from model import model_creator
from server import server_operation
from audio_processor import audio_processor

if __name__ == '__main__':
    find_microphone_id()

    samplerate = 44100
    duration = 5

    # Record and save single voice sample
    print("Please record voice of the person for the sample:")
    audio_recorder = AudioRecorder(duration * 2, samplerate)
    audio_data_single = audio_recorder.record_audio()
    file_name_single = generate_filename("sample")
    write(f"audio_training_data/{file_name_single}.wav", samplerate, (audio_data_single * 32767).astype(np.int16))
    voice_database.add_voice("user123",
                             feature_extractor.extract_features(f"audio_training_data/{file_name_single}.wav"))

    # Record and save group conversation
    print("Please record the conversation of multiple people:")
    audio_data_group = audio_recorder.record_audio()
    file_name_group = generate_filename("group")
    write(f"audio_training_data/{file_name_group}.wav", samplerate, (audio_data_group * 32767).astype(np.int16))

    # Compare features and save fragment if similar
    user123_mfcc = voice_database.get_voice("user123")
    if user123_mfcc is not None:
        new_mfcc = feature_extractor.extract_features(f"audio_training_data/{file_name_group}.wav")
        is_similar = voice_database.compare_features(user123_mfcc, new_mfcc)
        if is_similar:
            save_filename = f'fragment_{file_name_group}.npy'
            np.save(save_filename, new_mfcc)
            print(f"Successfully saved discussion fragment in {save_filename}")
        else:
            print("Voice features do not match the user!")

    # Extract and save sample in group
    result_filename = generate_filename("result")
    audio_processor.find_and_save_sample_in_group(
        f'audio_training_data/{file_name_single}.wav',
        f'audio_training_data/{file_name_group}.wav',
        f'result_audio/{result_filename}.wav'
    )

    # Model training
    X, y_labels = data_loader.load_data_from_directory('audio_training_data/')
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    y = to_categorical(y_encoded)

    input_shape = (X.shape[1], X.shape[2], 1)
    model = model_creator.create_model(input_shape, len(label_encoder.classes_))

    model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, shuffle=True)

    # Predict using the model
    new_mfcc = feature_extractor.extract_features(f'audio_training_data/{file_name_single}.wav')
    predicted_class = model.predict(np.expand_dims(new_mfcc, axis=0))
    predicted_label = label_encoder.inverse_transform([np.argmax(predicted_class)])
    print(f"The voice belongs to: {predicted_label}")

    server_operation.run_server()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Server stopped by user.")
