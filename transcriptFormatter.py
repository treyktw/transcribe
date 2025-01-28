import json
import time
import os
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import glob

class TranscriptFormatter:
    def __init__(self, directory):
        self.directory = directory
        self.processed_files = {}  # Keep track of last modified times
    
    def get_txt_path(self, json_path):
        """Convert JSON path to corresponding TXT path"""
        base_name = os.path.splitext(json_path)[0]
        return f"{base_name}.txt"
    
    def format_transcript(self, json_path):
        """Read JSON and format it into readable text"""
        try:
            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            txt_path = self.get_txt_path(json_path)
            
            # Format the transcript
            formatted_lines = []
            current_speaker = None
            
            # Add header with timestamp and filename
            formatted_lines.append(f"Transcript: {os.path.basename(json_path)}")
            formatted_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            formatted_lines.append("=" * 80 + "\n")
            
            for segment in data:
                # Add speaker line only when speaker changes
                if segment["speaker"] != current_speaker:
                    formatted_lines.append(f"\n[{segment['speaker']}]")
                    current_speaker = segment["speaker"]
                
                # Add timestamp and text
                formatted_lines.append(f"{segment['timestamp']}")
                formatted_lines.append(f"{segment['text']}\n")
            
            # Write to text file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(formatted_lines))
            
            print(f"Updated transcript: {os.path.basename(txt_path)} at {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"Error formatting transcript {json_path}: {str(e)}")

    def process_directory(self):
        """Process all JSON files in directory"""
        json_files = glob.glob(os.path.join(self.directory, "*.json"))
        
        for json_path in json_files:
            try:
                modified_time = os.path.getmtime(json_path)
                last_processed = self.processed_files.get(json_path, 0)
                
                # Process file if it's new or modified
                if modified_time > last_processed:
                    print(f"Processing {os.path.basename(json_path)}...")
                    self.format_transcript(json_path)
                    self.processed_files[json_path] = modified_time
            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}")

class TranscriptWatcher(FileSystemEventHandler):
    def __init__(self, formatter):
        self.formatter = formatter
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.json'):
            # Add a small delay to ensure file is completely written
            time.sleep(0.5)
            print(f"\nNew JSON file detected: {os.path.basename(event.src_path)}")
            self.formatter.format_transcript(event.src_path)
            self.formatter.processed_files[event.src_path] = os.path.getmtime(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.json'):
            # Add a small delay to ensure file is completely written
            time.sleep(0.5)
            print(f"\nJSON file modified: {os.path.basename(event.src_path)}")
            self.formatter.format_transcript(event.src_path)
            self.formatter.processed_files[event.src_path] = os.path.getmtime(event.src_path)

def main():
    # Directory to watch (current directory by default)
    watch_directory = os.getcwd()  # Change this to your desired directory path
    
    # Initialize formatter
    formatter = TranscriptFormatter(watch_directory)
    
    # Process existing files first
    print("Processing existing JSON files...")
    formatter.process_directory()
    
    # Set up file watcher
    event_handler = TranscriptWatcher(formatter)
    observer = Observer()
    observer.schedule(event_handler, path=watch_directory, recursive=False)
    observer.start()
    
    print(f"\nWatching directory: {watch_directory}")
    print("Waiting for new or modified JSON files...")
    
    try:
        while True:
            # Periodically check for new files that might have been missed
            formatter.process_directory()
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped watching directory")
    
    observer.join()

if __name__ == "__main__":
    main()