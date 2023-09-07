#!/usr/bin/env python
# export HF_DATASETS_CACHE="~/Desktop/v2v/.cache/huggingface/hub"

import torch
import scipy
from datetime import datetime
from transformers import BarkModel
from transformers import AutoProcessor
from fastapi import FastAPI

app = FastAPI()

import uvicorn

model = BarkModel.from_pretrained("suno/bark")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
processor = AutoProcessor.from_pretrained("suno/bark")

@app.get("/{prompt}")
async def inference(prompt: str):
  prompt = "Hello, world!"
  inputs = processor(prompt)
  sampling_rate = model.generation_config.sample_rate

  # # generate speech
  speech_output = model.generate(**inputs.to(device))

  scipy.io.wavfile.write("voice-{}.wav".format(datetime.now().strftime("%Y%M%D-%H%M")), rate=sampling_rate, data=speech_output[0].cpu().numpy())
  return prompt

if __name__ == "__main__":
  config = uvicorn.Config("server:app", port=8000, log_level="info")
  server = uvicorn.Server(config)
  server.run()
