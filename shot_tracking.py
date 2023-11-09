import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-medium.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)


def transcribe(chunk_length_s=20.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Recording...")
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
        yield item["text"]


def track_shots():
    total_shots = 0
    makes = 0
    while True:
        for transcriptions in transcribe():
            print(transcriptions)
            if ("fucker" in transcriptions.lower() 
                or "fuck" in transcriptions.lower() 
                or "fucking" in transcriptions.lower()):
                makes += 1
            total_shots += 1
            print(f"{makes}/{total_shots}")


track_shots()
