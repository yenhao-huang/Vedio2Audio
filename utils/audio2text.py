import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json
import logging

# Configure logging to output at INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutomaticSpeechRecognition:
    """
    Automatic Speech Recognition using OpenAI Whisper Large v3 model.
    Handles audio transcription and saving transcription results.
    """

    def __init__(self):
        """
        Initialize the ASR system by loading the model and processor,
        and setting the device (GPU if available, else CPU).
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16

        self.model_id = "openai/whisper-large-v3"

        logging.info(f"Loading model {self.model_id} to device {self.device}...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, dtype=self.dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        logging.info("Model loaded successfully.")

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
        """
        Transcribe the provided audio file.

        Args:
            audio (str): Path to the audio file to transcribe.

        Returns:
            dict: Transcription result.
        """
        logging.info(f"Transcribing audio file: {audio}")
        result = self.pipe(audio)
        logging.info("Transcription completed.")
        return result

    def save_results(self, result, output_file):
        """
        Save the transcription result to a JSON file.

        Args:
            result (dict): The transcription result to save.
            output_file (str): Path to the output JSON file.
        """
        logging.info(f"Saving transcription results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        logging.info("Results saved.")

def main():
    """
    Command-line interface for audio transcription.
    Parses arguments, runs transcription, and saves results.
    """
    argparser = argparse.ArgumentParser(description="Audio to Text Transcription")
    argparser.add_argument("--audio_file", type=str, default="data/1000.mp3",
                           help="Path to the audio file to transcribe")
    argparser.add_argument("--output_dir", type=str, default="results",
                           help="Directory to save the transcription results in JSON format")
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
