"""
Requirements:
pip install whisper torch pyannote.audio pydub tqdm
"""

import whisper
from pyannote.audio import Pipeline
import torch
import json
from datetime import datetime, timedelta
from pydub import AudioSegment
import os
from tqdm import tqdm
import tempfile

class AudioChunkProcessor:
    def __init__(self, auth_token, chunk_size_mins=10):
        """
        Initialize with chunk size in minutes
        """
        self.chunk_size_mins = chunk_size_mins
        self.auth_token = auth_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models in the main process
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        print("Loading Diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=auth_token
        ).to(self.device)

    def split_audio(self, audio_path):
        """Split audio file into chunks and save them"""
        print("Loading audio file...")
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = self.chunk_size_mins * 60 * 1000
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp()
        chunks = []
        
        print("Splitting audio into chunks...")
        for i in tqdm(range(0, len(audio), chunk_length_ms)):
            chunk = audio[i:i + chunk_length_ms]
            chunk_path = os.path.join(temp_dir, f'chunk_{i}.wav')
            chunk.export(chunk_path, format="wav")
            chunks.append({
                "path": chunk_path,
                "start_time": i / 1000  # Convert to seconds
            })
        return chunks, temp_dir

    def process_chunk(self, chunk_data):
        """Process a single chunk of audio"""
        chunk_path = chunk_data["path"]
        start_time = chunk_data["start_time"]
        
        try:
            # Transcribe
            print(f"\nTranscribing chunk starting at {timedelta(seconds=int(start_time))}...")
            transcription = self.whisper_model.transcribe(chunk_path)
            
            # Get speaker segments
            print(f"Analyzing speakers for chunk starting at {timedelta(seconds=int(start_time))}...")
            diarization = self.diarization_pipeline(chunk_path)
            
            # Create a mapping of unique speakers to numbers
            unique_speakers = sorted(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
            speaker_mapping = {speaker: f"Speaker {i+1}" for i, speaker in enumerate(unique_speakers)}
            
            # Convert diarization to list of segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "speaker": speaker_mapping[speaker],
                    "start": turn.start + start_time,
                    "end": turn.end + start_time
                })

            # Process segments
            final_segments = []
            for segment in transcription["segments"]:
                segment_start = segment["start"] + start_time
                segment_end = segment["end"] + start_time
                
                # Find matching speaker
                current_speaker = None
                max_overlap = 0
                
                for speaker_seg in speaker_segments:
                    overlap_start = max(segment_start, speaker_seg["start"])
                    overlap_end = min(segment_end, speaker_seg["end"])
                    
                    if overlap_end > overlap_start:
                        overlap_duration = overlap_end - overlap_start
                        if overlap_duration > max_overlap:
                            max_overlap = overlap_duration
                            current_speaker = speaker_seg["speaker"]

                final_segments.append({
                    "speaker": current_speaker if current_speaker else "Unknown Speaker",
                    "start": segment_start,
                    "end": segment_end,
                    "text": segment["text"].strip(),
                    "timestamp": f"{timedelta(seconds=int(segment_start))} --> {timedelta(seconds=int(segment_end))}"
                })

            print(f"Finished processing chunk at {timedelta(seconds=int(start_time))}")
            return final_segments

        except Exception as e:
            print(f"Error processing chunk at {timedelta(seconds=int(start_time))}: {str(e)}")
            return []

    def process_audio(self, audio_path, output_path):
        """Process audio file sequentially with progress updates"""
        # Split audio into chunks
        chunks, temp_dir = self.split_audio(audio_path)
        all_segments = []
        
        try:
            # Process chunks sequentially with progress bar
            print(f"\nProcessing {len(chunks)} chunks...")
            for chunk in tqdm(chunks):
                segments = self.process_chunk(chunk)
                all_segments.extend(segments)
                
                # Save intermediate results
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_segments, f, ensure_ascii=False, indent=2)
        
        finally:
            # Clean up temporary files
            print("\nCleaning up temporary files...")
            for chunk in chunks:
                try:
                    os.remove(chunk["path"])
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start"])
        
        # Save final results
        print("\nSaving final results...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_segments, f, ensure_ascii=False, indent=2)
        
        # Print sample of results
        print("\nSample of transcribed text:")
        for segment in all_segments[:5]:
            print(f"\n{segment['timestamp']}")
            print(f"{segment['speaker']}: {segment['text']}")

def main():
    # Replace with your HuggingFace auth token
    AUTH_TOKEN = "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"    
    # Initialize processor
    processor = AudioChunkProcessor(AUTH_TOKEN, chunk_size_mins=10)
    
    # Process audio file
    audio_path = "D:/Vscode/python_projects/transcribe/NPR5211779361.mp3"  # Replace with your audio
    output_path = "transcript.json"
    
    processor.process_audio(audio_path, output_path)

if __name__ == "__main__":
    main()