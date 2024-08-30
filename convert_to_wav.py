from pydub import AudioSegment
import os

def mp3_to_wav(mp3_file_path, wav_file_path):
    try:
        # Load the MP3 file
        audio = AudioSegment.from_mp3(mp3_file_path)
        
        # Export as WAV
        audio.export(wav_file_path, format="wav")
        print(f"Converted {mp3_file_path} to {wav_file_path}")
    except Exception as e:
        print(f"Failed to convert {mp3_file_path}: {e}")

def convert_folder(mp3_folder, wav_folder):
    if not os.path.exists(wav_folder):
        os.makedirs(wav_folder)
    
    for file_name in os.listdir(mp3_folder):
        if file_name.endswith(".mp3"):
            mp3_file_path = os.path.join(mp3_folder, file_name)
            wav_file_name = file_name.replace(".mp3", ".wav")
            wav_file_path = os.path.join(wav_folder, wav_file_name)
            
            # Check if the WAV file already exists
            if not os.path.exists(wav_file_path):
                mp3_to_wav(mp3_file_path, wav_file_path)
            else:
                print(f"{wav_file_path} already exists. Skipping conversion.")

if __name__ == "__main__":
    mp3_folder = "/home/wjb23/MSc-Project/Data/mp3s"  
    wav_folder = "/home/wjb23/MSc-Project/Data/wavs"  
    
    convert_folder(mp3_folder, wav_folder)
