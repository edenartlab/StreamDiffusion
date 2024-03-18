""" To use: install LLM studio (or Ollama), clone OpenVoice, run this script in the OpenVoice directory
    git clone https://github.com/myshell-ai/OpenVoice
    cd OpenVoice
    git clone https://huggingface.co/myshell-ai/OpenVoice
    cp -r OpenVoice/* .
    pip install -r requirements.txt
    pip install whisper pynput pyaudio
"""

#from openai import OpenAI
import ollama
import time
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

SYSTEM_MESSAGE = "You are Bob an AI assistant. KEEP YOUR RESPONSES VERY SHORT AND CONVERSATIONAL."
SPEAKER_WAV = None

#llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

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

def play_audio(text):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sampling_rate, output=True)
    texts = split_sentences_latin(text)
    for t in texts:
        audio_list = []
        t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
        t = f'[{mark}]{t}[{mark}]'
        stn_tst = tts_model.get_text(t, tts_model.hps, False)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(tts_model.device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(tts_model.device)
            sid = torch.LongTensor([tts_model.hps.speakers["default"]]).to(tts_model.device)
            audio = tts_model.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.6)[0][0, 0].data.cpu().float().numpy()
            if target_se is not None:
                audio = tone_color_converter.convert_from_tensor(audio=audio, src_se=en_source_default_se, tgt_se=target_se)
            audio_list.append(audio)            
        data = tts_model.audio_numpy_concat(audio_list, sr=sampling_rate).tobytes()
        stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()


def record_and_transcribe_audio():
    recording = False
    def on_press(key):
        nonlocal recording
        if key == keyboard.Key.shift:
            recording = True

    def on_release(key):
        nonlocal recording
        if key == keyboard.Key.shift:
            recording = False
            return False

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    print('Press shift to record...')
    while not recording:
        time.sleep(0.1)
    print('Start recording...')

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, frames_per_buffer=1024, input=True)
    frames = []
    while recording:
        data = stream.read(1024, exception_on_overflow = False)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print('Finished recording')

    frames = [np.array(frame, dtype=np.float32) for frame in frames]  # Convert each frame to float32
    data = np.hstack(frames) / 32768.0
    #data = np.hstack(frames, dtype=np.float32) / 32768.0 
    result = asr_model.transcribe(data)['text']
    print(result)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return result


def conversation():
    conversation_history = [{'role': 'system', 'content': SYSTEM_MESSAGE}]
    while True:
        user_input = record_and_transcribe_audio()
        conversation_history.append({'role': 'user', 'content': user_input})

        #response = llm_client.chat.completions.create(model="local-model", messages=conversation_history)
        response = ollama.chat(model='llama2', messages=conversation_history)
        #chatbot_response = response.choices[0].message.content
        print(response)
        print(response['message']['content'])
        chatbot_response = response['message']['content']
        conversation_history.append({'role': 'assistant', 'content': chatbot_response})
        print(conversation_history)
        play_audio(chatbot_response)

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

conversation()