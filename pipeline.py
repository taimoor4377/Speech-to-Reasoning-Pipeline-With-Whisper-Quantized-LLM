import os
import time
import librosa
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import whisper


# ----------------- Audio Processing -----------------
class AudioProcessor:
    def validate_audio(self, audio_path: str) -> bool:
        return os.path.exists(audio_path) and audio_path.split(".")[-1].lower() in ["wav", "mp3", "m4a"]

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        audio, sr = librosa.load(audio_path, sr=None)
        return audio

    def get_audio_duration(self, audio_path: str) -> float:
        return librosa.get_duration(filename=audio_path)


# ----------------- Whisper Transcription -----------------
class WhisperTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(audio)
        return result.get("text", "")

    def transcribe_with_timestamps(self, audio: np.ndarray) -> dict:
        return self.model.transcribe(audio, verbose=True)


# ----------------- Quantized LLM Handler -----------------
class QuantizedLLMHandler:
    def __init__(self, model_name: str, quantization_config: dict):
        self.model_name = model_name
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            device_map="auto"
        )

    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        result = generator(prompt, max_new_tokens=max_tokens)
        return result[0]["generated_text"]

    def clear_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------- Speech Reasoning Pipeline -----------------
class SpeechReasoningPipeline:
    def __init__(self, whisper_model: str = "base", llm_model: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
        self.audio_processor = AudioProcessor()
        self.transcriber = WhisperTranscriber(whisper_model)
        self.llm_handler = QuantizedLLMHandler(llm_model, {"quantization": "4bit"})
        self.llm_handler.load_model()

    def process_audio(self, audio_path: str, reasoning_prompt: str = None) -> dict:
        start_time = time.time()
        stats = {}

        if not self.audio_processor.validate_audio(audio_path):
            raise ValueError("Invalid or unsupported audio format.")

        audio = self.audio_processor.preprocess_audio(audio_path)
        transcription = self.transcriber.transcribe(audio)

        if reasoning_prompt:
            transcription = f"{reasoning_prompt}\n{transcription}"

        response = self.llm_handler.generate_response(transcription)

        stats["total_time"] = round(time.time() - start_time, 2)
        return {
            "transcription": transcription,
            "response": response,
            "stats": stats
        }

    def get_pipeline_stats(self) -> dict:
        return {"cuda_mem_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}


# ----------------- Example -----------------
if __name__ == "__main__":
    pipeline = SpeechReasoningPipeline()
    result = pipeline.process_audio("sample.wav", reasoning_prompt="Summarize this query:")
    print(result)
