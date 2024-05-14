# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/google/paligemma-3b-pt-224/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            revision="bfloat16",
        ).to('cuda')
        self.processor = AutoProcessor.from_pretrained(MODEL_CACHE)

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        prompt: str = Input(description="Input prompt", default="caption es"),
    ) -> str:
        """Run a single prediction on the model"""
        img = Image.open(image).convert("RGB")
        model_inputs = self.processor(text=prompt, images=img, return_tensors="pt").to('cuda')
        input_len = model_inputs["input_ids"].shape[-1]

        generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
