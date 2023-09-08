#!/usr/bin/env python
# export HF_DATASETS_CACHE="~/Desktop/text2speech-poc/.cache/huggingface/hub"

import os
import torch
import scipy
import uvicorn
from datetime import datetime
from transformers import BarkModel
from transformers import AutoProcessor
from fastapi import FastAPI

export_dir = os.getcwd()

app = FastAPI()
model = BarkModel.from_pretrained("suno/bark")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

processor = AutoProcessor.from_pretrained("suno/bark")

@app.get("/{prompt}")
async def inference(prompt: str):
  inputs = processor(prompt)
  sampling_rate = model.generation_config.sample_rate

  # # generate speech
  speech_output = model.generate(**inputs.to(device))

  scipy.io.wavfile.write(export_dir + "/voice-{}.wav".format(datetime.now()), rate=sampling_rate, data=speech_output[0].cpu().numpy())
  return prompt

if __name__ == "__main__":
  config = uvicorn.Config("server:app", port=8000, log_level="info")
  server = uvicorn.Server(config)
  server.run()
