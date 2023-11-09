import datetime
import random
import sounddevice as sd

from constants import MIC_ID


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def generate_filename(prefix):
    """Generates a unique filename with the given prefix."""
    now = datetime.datetime.now()
    random_number = random.randint(0, 9999)
    return f"{prefix}_{now.strftime('%Y%m%d%H%M%S')}_{random_number:04d}"


def find_microphone_id():
    """Finds and returns the ID of the device which name contains 'MICROPHONE'."""
    print("Available audio devices:")
    microphone_id = MIC_ID
    for i, device in enumerate(sd.query_devices()):
        print(f"  Device {i}: {device['name']} (sample rates: {device['default_samplerate']} Hz)")
        if 'MICROPHONE' in device['name'].upper():
            microphone_id = i
            print(f"==> Microphone found: Device ID {microphone_id}")

    if microphone_id is not None:
        return microphone_id
    else:
        print("No microphone found.")
        return None
