from moviepy.editor import VideoFileClip
import whisper
import json

# Extract Audio from Video
def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def prepare_file_for_adding_captions_n_headings_thru_html(input_video_path="input_video.mp4"):
    
    #print("Received add_captions Arguments:", locals())

    audio_path = "audio.wav"

    try:
        extract_audio(input_video_path, audio_path)
        print("Audio extracted successfully!")
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return
    
    model = whisper.load_model("base")
    captions_data = model.transcribe(audio_path, word_timestamps=True)

    word_timestamps = []
    position_index = 0  # Track word positions

    for segment in captions_data["segments"]:
        for word_data in segment.get("words", []):
            word_timestamps.append({
                "word": word_data["word"].lower(),  # Normalize case
                "start": word_data["start"],
                "end": word_data["end"],
                "position": position_index,  # Assign position index
                "matched": False  # Initialize as not matched
            })
            position_index += 1

    with open('temp/word_timestamps.json', 'w') as f:
        json.dump(word_timestamps, f,indent=4)

    print("Extracted captions saved to temp/word_timestamps.json")


if __name__ == "__main__":
    prepare_file_for_adding_captions_n_headings_thru_html("input_video.mp4")
