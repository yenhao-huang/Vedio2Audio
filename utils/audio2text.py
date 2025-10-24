import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

class AutomaticSpeechRecognition:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16

        self.model_id = "openai/whisper-large-v3"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, dtype=self.dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            return_timestamps=True,
            dtype=self.dtype,
            device=self.device,
        )
    
    def transcribe(self, audio):
        result = self.pipe(audio)
        return result
    
    def save_results(self, result, output_file):
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)


def main():
    argparser = argparse.ArgumentParser(description="Audio to Text Transcription")
    argparser.add_argument("--audio_file", type=str, required=True, help="Path to \
        the audio file to transcribe")
    argparser.add_argument("--output_dir", type=str, default="results", help="Directory to \
        save the transcription results in JSON format")
    args = argparser.parse_args()

    asr = AutomaticSpeechRecognition()
    transcription = asr.transcribe(args.audio_file)
    print("Transcription:", transcription)

    result = {"audio_file": args.audio_file, "transcription": transcription}
    filename = args.audio_file.split('/')[-1].split('.')[0]
    output_file = f"{args.output_dir}/{filename}_transcription.json"
    asr.save_results(result, output_file)

if __name__ == "__main__":  
    main()