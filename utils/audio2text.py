import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

class Audio2Text:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.detype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model_id = "openai/whisper-large-v3"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, detype=self.detype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            detype=self.detype,
            device=self.device,
        )
    
    def transcribe(self, audio):
        result = self.pipe(audio)
        return result["text"]
    
    def save_results(self, result, output_file):
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)


def main():
    argparser = argparse.ArgumentParser(description="Audio to Text Transcription")
    argparser.add_argument("--audio_file", type=str, required=True, help="Path to \
        the audio file to transcribe")
    argparser.add_argument("--output_file", type=str, required=True, help="Path to \
        save the transcription results in JSON format")
    args = argparser.parse_args()

    audio2text = Audio2Text()
    transcription = audio2text.transcribe(args.audio_file)
    print("Transcription:", transcription)

    result = {"audio_file": audio_file, "transcription": transcription}
    audio2text.save_results(result, args.output_file)

if __name__ == "__main__":  
    main()