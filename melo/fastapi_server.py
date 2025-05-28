from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import io
from melo.api import TTS
from fastapi.responses import StreamingResponse
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import torch

app = FastAPI()

# Legacy local model (not optimized)
device = "auto"
models = {
    "EN": TTS(language="EN", device=device),
    "ES": TTS(language="ES", device=device),
    "FR": TTS(language="FR", device=device),
    "ZH": TTS(language="ZH", device=device),
    "JP": TTS(language="JP", device=device),
    "KR": TTS(language="KR", device=device),
}

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

tts_models = []
for i in range(num_gpus):
    tts_models.append(TTS(language="EN", device=f"cuda:{i}"))

class SynthesizePayload(BaseModel):
    text: str = "Ahoy there matey! There she blows!"
    language: str = "EN"
    speaker: str = "EN-US"
    speed: float = 1.0
    optimize: bool = False

def synthesize_chunk(model, text, speaker, speed):
    bio = io.BytesIO()
    model.tts_to_file(
        text, model.hps.data.spk2id[speaker], bio, speed=speed, format="wav"
    )
    return AudioSegment.from_file(io.BytesIO(bio.getvalue()), format="wav")

def split_text(text, num_parts):
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return [text]
    
    sentences_per_part = len(sentences) // num_parts
    if sentences_per_part == 0:
        sentences_per_part = 1
    
    parts = []
    for i in range(0, len(sentences), sentences_per_part):
        part = " ".join(sentences[i:i + sentences_per_part])
        if part:
            parts.append(part)
    
    while len(parts) < num_parts:
        parts.append(parts[-1])
    
    return parts[:num_parts]

def parallel_tts(text, speaker, speed):
    if not tts_models:
        tts = TTS(language="EN", device="cpu")
        return synthesize_chunk(tts, text, speaker, speed)
    
    text_parts = split_text(text, len(tts_models))
    
    with ThreadPoolExecutor(max_workers=len(tts_models)) as executor:
        futures = []
        for i, part in enumerate(text_parts):
            futures.append(
                executor.submit(synthesize_chunk, tts_models[i], part, speaker, speed)
            )
        audio_segments = [f.result() for f in futures]
    
    final_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        final_audio += segment
    
    return final_audio

@app.post("/stream")
async def synthesize_stream(payload: SynthesizePayload):
    text = payload.text
    speaker = payload.speaker or list(tts_models[0].hps.data.spk2id.keys())[0]
    speed = payload.speed
    optimize = payload.optimize
    language = payload.language

    if optimize:
        audio = parallel_tts(text, speaker, speed)

        def audio_stream():
            out_io = io.BytesIO()
            audio.export(out_io, format="wav")
            yield out_io.getvalue()

        return StreamingResponse(audio_stream(), media_type="audio/wav")
    else:
        def audio_stream():
            bio = io.BytesIO()
            models[language].tts_to_file(
                text,
                models[language].hps.data.spk2id[speaker],
                bio,
                speed=speed,
                format="wav",
            )
            audio_data = bio.getvalue()
            yield audio_data

        return StreamingResponse(audio_stream(), media_type="audio/wav")
