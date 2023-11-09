import speech_recognition as sr
from pydub import AudioSegment

from app.constants import RESULT_FILENAME


class AudioAnalyzer:
    async def process_audio(self):
        try:
            recognizer = sr.Recognizer()
            result_filename = RESULT_FILENAME
            audio_file = f'result_audio/{result_filename}.wav'
            audio = AudioSegment.from_file(audio_file, format="wav")
            audio.export("converted.wav", format="wav")

            with sr.AudioFile("converted.wav") as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language='pl-PL')

        except sr.UnknownValueError:
            text = "You can't recognize the speech."
        except sr.RequestError:
            text = "Communication error with Google Speech Recognition."
        except Exception as e:
            text = f"An error occurred: {e}"

        return text


audio_analyzer = AudioAnalyzer()
