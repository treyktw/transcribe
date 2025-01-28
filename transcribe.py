"""
Requirements:
pip install whisper torch pyannote.audio spacy
"""

import whisper
from pyannote.audio import Pipeline
import torch
import json
import spacy
from datetime import datetime, timedelta

class AudioProcessor:
    def __init__(self, auth_token):
        """
        Initialize the AudioProcessor with HuggingFace auth token.
        Get your token from: https://hf.co/settings/tokens
        """
        self.whisper_model = whisper.load_model("base")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=auth_token
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Load spaCy for name recognition
        self.nlp = spacy.load("en_core_web_sm")

    def extract_names_from_text(self, text):
        """Extract potential names from text using spaCy"""
        doc = self.nlp(text)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        return names

    def find_speaker_introductions(self, transcription, speaker_segments, intro_duration=30):
        """
        Find speaker introductions in the first few seconds of audio.
        Returns a mapping of speaker IDs to real names.
        """
        speaker_names = {}
        intro_texts = []

        # Collect all text from the introduction period
        for segment in transcription:
            if segment["start"] <= intro_duration:
                intro_texts.append(segment["text"])

        # Join all introduction text
        intro_text = " ".join(intro_texts)
        
        # Look for patterns like "I'm [Name]" or "My name is [Name]"
        doc = self.nlp(intro_text.lower())
        
        # Find speaker segments in intro period
        intro_speakers = [s for s in speaker_segments if s["start"] <= intro_duration]
        
        # Extract names from the introduction text
        names = self.extract_names_from_text(intro_text)
        
        # Map speakers to names based on order of appearance
        for i, speaker in enumerate(set(s["speaker"] for s in intro_speakers)):
            if i < len(names):
                speaker_names[speaker] = names[i]
        
        return speaker_names

    def transcribe_audio(self, audio_path):
        """Transcribe audio file using Whisper"""
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["segments"]

    def get_speaker_segments(self, audio_path):
        """Get speaker diarization segments"""
        print("Analyzing speakers...")
        diarization = self.diarization_pipeline(audio_path)
        
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        return speaker_segments

    def match_transcription_with_speakers(self, transcription, speaker_segments, speaker_names=None):
        """Match transcribed segments with speaker information"""
        final_segments = []
        
        for trans_segment in transcription:
            segment_start = trans_segment["start"]
            segment_end = trans_segment["end"]
            
            # Find overlapping speaker segment
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
            
            # Use real name if available, otherwise use speaker ID
            if speaker_names and current_speaker in speaker_names:
                speaker_label = speaker_names[current_speaker]
            else:
                speaker_label = current_speaker if current_speaker else "Unknown"

            final_segments.append({
                "speaker": speaker_label,
                "start": segment_start,
                "end": segment_end,
                "text": trans_segment["text"].strip(),
                "timestamp": f"{timedelta(seconds=int(segment_start))} --> {timedelta(seconds=int(segment_end))}"
            })
        
        return final_segments

    def process_audio(self, audio_path, output_path):
        """Process audio file and save results"""
        # Get transcription and speaker segments
        transcription = self.transcribe_audio(audio_path)
        speaker_segments = self.get_speaker_segments(audio_path)
        
        # Find speaker names from introductions
        speaker_names = self.find_speaker_introductions(transcription, speaker_segments)
        print("Detected speakers:", speaker_names)
        
        # Match speakers with transcribed text
        final_segments = self.match_transcription_with_speakers(
            transcription, 
            speaker_segments,
            speaker_names
        )
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_segments, f, ensure_ascii=False, indent=2)
        
        # Print results
        for segment in final_segments:
            print(f"\n{segment['timestamp']}")
            print(f"{segment['speaker']}: {segment['text']}")

def main():
    # Replace with your HuggingFace auth token
    AUTH_TOKEN = "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"
    
    # Initialize processor
    processor = AudioProcessor(AUTH_TOKEN)
    
    # Process audio file
    audio_path = "D:/Vscode/python_projects/transcribe/NPR5211779361.mp3"  # Replace with your audio file path
    output_path = "transcript.json"
    
    processor.process_audio(audio_path, output_path)

if __name__ == "__main__":
    main()