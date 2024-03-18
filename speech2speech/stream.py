""" To use: install LLM studio (or Ollama), clone OpenVoice, run this script in the OpenVoice directory
    git clone https://github.com/myshell-ai/OpenVoice
    cd OpenVoice
    git clone https://huggingface.co/myshell-ai/OpenVoice
    cp -r OpenVoice/* .
    pip install -r requirements.txt
    pip install whisper pynput pyaudio ollama
"""

import sys
sys.path.append("OpenVoice")

#from openai import OpenAI
import ollama
import time
import threading
import pyaudio
import numpy as np
import torch
import os
import re
from openvoice import se_extractor
import whisper
from pynput import keyboard
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice.utils import split_sentences_latin




SYSTEM_PROMPT = """You are an artist who describes rich visual scenes in exquisite detail."""

PROMPT_TEMPLATE = """Here's a transcript of an ongoing narration. The transcript is rough and sometimes repeated:

---
{transcription}
---

Write a rich visual description of 20-30 words that tries to visualize what is being talked about.
Do not summarize or restate the request or state any pretext, output ONLY the visual description of the scene.
"""

SPEAKER_WAV = None

BUFFER_DURATION    = 9   # How many seconds of audio to keep in the buffer
NUM_TRANSCRIPTIONS = 2
TRANSCRIBE_EVERY   = 4  # 0-5, 4-9
CALL_LLM_EVERY     = 4

tts_en_ckpt_base = "OpenVoice/checkpoints/base_speakers/EN"
tts_ckpt_converter = "OpenVoice/checkpoints/converter"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else  "cpu"

tts_model = BaseSpeakerTTS(f'{tts_en_ckpt_base}/config.json', device=device)
tts_model.load_ckpt(f'{tts_en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{tts_ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{tts_ckpt_converter}/checkpoint.pth')
en_source_default_se = torch.load(f"{tts_en_ckpt_base}/en_default_se.pth").to(device)
target_se, _ = se_extractor.get_se(SPEAKER_WAV, tone_color_converter, target_dir='processed', vad=True) if SPEAKER_WAV else (None, None)
sampling_rate = tts_model.hps.data.sampling_rate
mark = tts_model.language_marks.get("english", None)
asr_model = whisper.load_model("base.en")


class RollingAudioBuffer:
    def __init__(self, buffer_duration=5, sample_rate=16000):
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.buffer_size = buffer_duration * sample_rate
        self.buffer = np.empty(0, dtype=np.int16)
        self.lock = threading.Lock()

    def add_chunk(self, chunk):
        with self.lock:
            self.buffer = np.append(self.buffer, chunk)
            if len(self.buffer) > self.buffer_size:
                self.buffer = self.buffer[-self.buffer_size:]

    def get_buffer(self):
        with self.lock:
            return np.array(self.buffer, dtype=np.int16)

class TranscriptionBuffer:
    def __init__(self, max_transcriptions=2):
        self.max_transcriptions = max_transcriptions
        self.transcriptions = []
        self.lock = threading.Lock()

    def add_transcription(self, transcription):
        with self.lock:
            self.transcriptions.append(transcription)
            if len(self.transcriptions) > self.max_transcriptions:
                self.transcriptions.pop(0)

    def get_transcriptions(self):
        with self.lock:
            return list(self.transcriptions)

def record_audio(rolling_buffer, stream):
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        chunk = np.frombuffer(data, dtype=np.int16)
        rolling_buffer.add_chunk(chunk)

def transcribe_buffer(buffer, model):
    data = np.array(buffer, dtype=np.float32) / 32768.0
    result = model.transcribe(data)['text']
    return result

def transcribe_audio(rolling_buffer, transcription_buffer, model):
    while True:
        time.sleep(TRANSCRIBE_EVERY)  # Transcribe every n seconds
        buffer = rolling_buffer.get_buffer()
        if len(buffer) > 0:
            transcription_result = transcribe_buffer(buffer, model)
            transcription_buffer.add_transcription(transcription_result)
            #print(f"Transcribed: {transcription_result}")
            

def print_transcriptions(transcription_buffer):
    while True:
        time.sleep(CALL_LLM_EVERY)  # Print transcriptions every 4 seconds
        transcriptions = transcription_buffer.get_transcriptions()

        print("-------------------------------")
        print("\n".join(transcriptions))
        print("-------------------------------")
        
        prompt = PROMPT_TEMPLATE.format(transcription="\n".join(transcriptions))

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ]
        output_prompt = ollama.chat(model='gemma:7b-instruct', messages=messages)['message']['content']

        #print("-------")
        #print(output_prompt)
        with open('prompt.txt', 'w') as file:
            file.write(output_prompt)


def main():
    model = whisper.load_model("base.en")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, frames_per_buffer=1024, input=True)

    rolling_buffer = RollingAudioBuffer(buffer_duration=BUFFER_DURATION)
    transcription_buffer = TranscriptionBuffer(max_transcriptions=NUM_TRANSCRIPTIONS)

    # Start recording thread
    recording_thread = threading.Thread(target=record_audio, args=(rolling_buffer, stream))
    recording_thread.start()

    # Start transcription thread
    transcription_thread = threading.Thread(target=transcribe_audio, args=(rolling_buffer, transcription_buffer, model))
    transcription_thread.start()

    # Start thread to print transcriptions
    print_thread = threading.Thread(target=print_transcriptions, args=(transcription_buffer,))
    print_thread.start()

    # Wait for threads to complete (they won't in this case, so this script will run until manually stopped)
    recording_thread.join()
    transcription_thread.join()
    print_thread.join()

if __name__ == "__main__":
   main()