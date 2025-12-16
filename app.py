import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import librosa
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
import subprocess  # To run FFmpeg commands for re-encoding

app = Flask(__name__)

# Set the model folder path
MODEL_FOLDER = 'model/'

# Load model configuration from JSON file
with open(os.path.join(MODEL_FOLDER, 'config.json'), 'r') as json_file:
    model_json = json_file.read()

# Recreate the model from the JSON configuration
model = model_from_json(model_json)

# Load the model weights
model.load_weights(os.path.join(MODEL_FOLDER, 'model.weights.h5'))

# Configure upload folder and allowed file types
UPLOAD_FOLDER = 'static/audio_uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Explicitly set the path for ffmpeg and ffprobe using pydub's 'which'
from pydub.utils import which

# Railway will automatically detect the ffmpeg binary installed
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")

# Set ffmpeg and ffprobe paths explicitly
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

# Check if ffmpeg and ffprobe are correctly set (for debugging purposes)
print("ffmpeg path:", AudioSegment.ffmpeg)
print("ffprobe path:", AudioSegment.ffprobe)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def reencode_mp3(mp3_file_path):
    """
    Re-encode the MP3 file to a new MP3 to resolve encoding issues.
    """
    # Define the output file path
    reencoded_file_path = mp3_file_path.rsplit('.', 1)[0] + '_reencoded.mp3'

    # Run FFmpeg to re-encode the file with the correct codec and settings
    subprocess.run([
        ffmpeg_path, '-y', '-i', mp3_file_path, '-vn', '-acodec', 'libmp3lame', '-ar', '44100', '-ab', '192k', reencoded_file_path
    ], check=True)

    return reencoded_file_path


def convert_mp3_to_wav(mp3_file_path):
    """
    Convert MP3 to WAV using FFmpeg.
    """
    wav_file_path = mp3_file_path.rsplit('.', 1)[0] + '.wav'
    audio = AudioSegment.from_file(mp3_file_path, format="mp3")
    audio.export(wav_file_path, format="wav")
    return wav_file_path


def preprocess_audio(audio_file):
    """
    This function loads an audio file, extracts MFCC features,
    and prepares it for prediction by the model.
    """
    # Load the audio file with librosa
    data, sr = librosa.load(audio_file, sr=22050)  # Load at 22050Hz sample rate

    # Extract MFCCs (40 coefficients as expected by the model)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccs = mfccs.T  # Transpose to shape (frames, features)

    # Pad or truncate to match model input timesteps (228)
    if mfccs.shape[0] < 228:
        pad_width = 228 - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    elif mfccs.shape[0] > 228:
        mfccs = mfccs[:228, :]

    # Reshape to fit the model's input (1, 228, 40, 1)
    mfccs = mfccs.reshape(1, 228, 40, 1)

    return mfccs


def predict_emotion(audio_file):
    """
    This function preprocesses the audio file and passes it through the model
    to predict the emotion.
    """
    # Preprocess the audio file to extract features
    audio_data = preprocess_audio(audio_file)

    # Model prediction
    prediction = model.predict(audio_data)
    emotion_index = np.argmax(prediction)  # Get the emotion with the highest probability

    # Mapping of emotion index to label and emoji
    label_conversion = {
        0: ('neutral', '😐'),
        1: ('calm', '😌'),
        2: ('happy', '😊'),
        3: ('sad', '😢'),
        4: ('angry', '😡'),
        5: ('fear', '😱'),
        6: ('disgust', '🤢'),
        7: ('surprise', '😲')
    }

    # Get the corresponding emotion label and emoji
    emotion_label, emotion_emoji = label_conversion.get(emotion_index, ("unknown", "❓"))
    return emotion_label, emotion_emoji


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # If it's an MP3 file, re-encode it to fix possible header issues
        if filename.endswith('.mp3'):
            try:
                reencoded_file_path = reencode_mp3(file_path)
                wav_file_path = convert_mp3_to_wav(reencoded_file_path)
                emotion_label, emotion_emoji = predict_emotion(wav_file_path)
                return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})


            except subprocess.CalledProcessError as e:
                return jsonify({"error": f"FFmpeg error: {e}"})

        # Process WAV files directly
        emotion_label, emotion_emoji = predict_emotion(file_path)
        return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})

    return jsonify({"error": "Invalid file format"})


if __name__ == '__main__':
    # Ensure to listen on the dynamic port provided by Railway
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)



# # LATEST APP.PY FOR CLOUD DEPLOYMENT
# import os
# import json
# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from pydub import AudioSegment
# import librosa
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import model_from_json
# import subprocess  # To run FFmpeg commands for re-encoding
#
# app = Flask(__name__)
#
# # Set the model folder path
# MODEL_FOLDER = 'model/'
#
# # Load model configuration from JSON file
# with open(os.path.join(MODEL_FOLDER, 'config.json'), 'r') as json_file:
#     model_json = json_file.read()
#
# # Recreate the model from the JSON configuration
# model = model_from_json(model_json)
#
# # Load the model weights
# model.load_weights(os.path.join(MODEL_FOLDER, 'model.weights.h5'))
#
# # Configure upload folder and allowed file types
# UPLOAD_FOLDER = 'static/audio_uploads'
# ALLOWED_EXTENSIONS = {'mp3', 'wav'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# # Explicitly set the path for ffmpeg and ffprobe using pydub's 'which'
# from pydub.utils import which
#
# # Railway will automatically detect the ffmpeg binary installed
# ffmpeg_path = which("ffmpeg")
# ffprobe_path = which("ffprobe")
#
# # Set ffmpeg and ffprobe paths explicitly
# AudioSegment.ffmpeg = ffmpeg_path
# AudioSegment.ffprobe = ffprobe_path
#
# # Check if ffmpeg and ffprobe are correctly set (for debugging purposes)
# print("ffmpeg path:", AudioSegment.ffmpeg)
# print("ffprobe path:", AudioSegment.ffprobe)
#
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# def reencode_mp3(mp3_file_path):
#     """
#     Re-encode the MP3 file to a new MP3 to resolve encoding issues.
#     """
#     # Define the output file path
#     reencoded_file_path = mp3_file_path.rsplit('.', 1)[0] + '_reencoded.mp3'
#
#     # Run FFmpeg to re-encode the file with the correct codec and settings
#     subprocess.run([
#         ffmpeg_path, '-y', '-i', mp3_file_path, '-vn', '-acodec', 'libmp3lame', '-ar', '44100', '-ab', '192k', reencoded_file_path
#     ], check=True)
#
#     return reencoded_file_path
#
#
# def convert_mp3_to_wav(mp3_file_path):
#     """
#     Convert MP3 to WAV using FFmpeg.
#     """
#     wav_file_path = mp3_file_path.rsplit('.', 1)[0] + '.wav'
#     audio = AudioSegment.from_file(mp3_file_path, format="mp3")
#     audio.export(wav_file_path, format="wav")
#     return wav_file_path
#
#
# def preprocess_audio(audio_file):
#     """
#     This function loads an audio file, extracts MFCC features,
#     and prepares it for prediction by the model.
#     """
#     # Load the audio file with librosa
#     data, sr = librosa.load(audio_file, sr=22050)  # Load at 22050Hz sample rate
#
#     # Extract MFCCs (40 coefficients as expected by the model)
#     mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
#     mfccs = mfccs.T  # Transpose to shape (frames, features)
#
#     # Pad or truncate to match model input timesteps (228)
#     if mfccs.shape[0] < 228:
#         pad_width = 228 - mfccs.shape[0]
#         mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
#     elif mfccs.shape[0] > 228:
#         mfccs = mfccs[:228, :]
#
#     # Reshape to fit the model's input (1, 228, 40, 1)
#     mfccs = mfccs.reshape(1, 228, 40, 1)
#
#     return mfccs
#
#
# def predict_emotion(audio_file):
#     """
#     This function preprocesses the audio file and passes it through the model
#     to predict the emotion.
#     """
#     # Preprocess the audio file to extract features
#     audio_data = preprocess_audio(audio_file)
#
#     # Model prediction
#     prediction = model.predict(audio_data)
#     emotion_index = np.argmax(prediction)  # Get the emotion with the highest probability
#
#     # Mapping of emotion index to label and emoji
#     label_conversion = {
#         0: ('neutral', '😐'),
#         1: ('calm', '😌'),
#         2: ('happy', '😊'),
#         3: ('sad', '😢'),
#         4: ('angry', '😡'),
#         5: ('fear', '😱'),
#         6: ('disgust', '🤢'),
#         7: ('surprise', '😲')
#     }
#
#     # Get the corresponding emotion label and emoji
#     emotion_label, emotion_emoji = label_conversion.get(emotion_index, ("unknown", "❓"))
#     return emotion_label, emotion_emoji
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({"error": "No selected file"})
#
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#
#         # If it's an MP3 file, re-encode it to fix possible header issues
#         if filename.endswith('.mp3'):
#             try:
#                 reencoded_file_path = reencode_mp3(file_path)
#                 wav_file_path = convert_mp3_to_wav(reencoded_file_path)
#                 emotion_label, emotion_emoji = predict_emotion(wav_file_path)
#                 return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})
#             except subprocess.CalledProcessError as e:
#                 return jsonify({"error": f"FFmpeg error: {e}"})
#
#         # Process WAV files directly
#         emotion_label, emotion_emoji = predict_emotion(file_path)
#         return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})
#
#     return jsonify({"error": "Invalid file format"})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


# LATEST APP.PY FOR LOCAL DEPLOYMENT
# import os
# import json
# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from pydub import AudioSegment
# import librosa
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import model_from_json
# import subprocess  # To run FFmpeg commands for re-encoding
#
# app = Flask(__name__)
#
# # Set the model folder path
# MODEL_FOLDER = 'model/'
#
# # Load model configuration from JSON file
# with open(os.path.join(MODEL_FOLDER, 'config.json'), 'r') as json_file:
#     model_json = json_file.read()
#
# # Recreate the model from the JSON configuration
# model = model_from_json(model_json)
#
# # Load the model weights
# model.load_weights(os.path.join(MODEL_FOLDER, 'model.weights.h5'))
#
# # Configure upload folder and allowed file types
# UPLOAD_FOLDER = 'static/audio_uploads'
# ALLOWED_EXTENSIONS = {'mp3', 'wav'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# # Explicitly set the path for ffmpeg and ffprobe
# from pydub.utils import which
#
# # Set the full path to your ffmpeg and ffprobe executables
# ffmpeg_path = r"D:\Download2\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
# ffprobe_path = r"D:\Download2\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe"
#
# # Set ffmpeg and ffprobe paths explicitly
# AudioSegment.ffmpeg = ffmpeg_path
# AudioSegment.ffprobe = ffprobe_path
#
# # Check if ffmpeg and ffprobe are correctly set
# print("ffmpeg path:", AudioSegment.ffmpeg)
# print("ffprobe path:", AudioSegment.ffprobe)
#
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# # def reencode_mp3(mp3_file_path):
# #     """
# #     Re-encode the MP3 file to a new MP3 to resolve encoding issues.
# #     """
# #     # Define the output file path
# #     reencoded_file_path = mp3_file_path.rsplit('.', 1)[0] + '_reencoded.mp3'
# #
# #     # Run FFmpeg to re-encode the file with the correct codec and settings
# #     subprocess.run([
# #         ffmpeg_path, '-i', mp3_file_path, '-vn', '-acodec', 'libmp3lame', '-ar', '44100', '-ab', '192k', reencoded_file_path
# #     ], check=True)
# #
# #     return reencoded_file_path
#
# def reencode_mp3(mp3_file_path):
#     """
#     Re-encode the MP3 file to a new MP3 to resolve encoding issues.
#     """
#     # Define the output file path
#     reencoded_file_path = mp3_file_path.rsplit('.', 1)[0] + '_reencoded.mp3'
#
#     # Run FFmpeg to re-encode the file with the correct codec and settings
#     subprocess.run([
#         ffmpeg_path, '-y', '-i', mp3_file_path, '-vn', '-acodec', 'libmp3lame', '-ar', '44100', '-ab', '192k', reencoded_file_path
#     ], check=True)
#
#     return reencoded_file_path
#
#
# def convert_mp3_to_wav(mp3_file_path):
#     """
#     Convert MP3 to WAV using FFmpeg.
#     """
#     wav_file_path = mp3_file_path.rsplit('.', 1)[0] + '.wav'
#     audio = AudioSegment.from_file(mp3_file_path, format="mp3")
#     audio.export(wav_file_path, format="wav")
#     return wav_file_path
#
#
# def preprocess_audio(audio_file):
#     """
#     This function loads an audio file, extracts MFCC features,
#     and prepares it for prediction by the model.
#     """
#     # Load the audio file with librosa
#     data, sr = librosa.load(audio_file, sr=22050)  # Load at 22050Hz sample rate
#
#     # Extract MFCCs (40 coefficients as expected by the model)
#     mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
#     mfccs = mfccs.T  # Transpose to shape (frames, features)
#
#     # Pad or truncate to match model input timesteps (228)
#     if mfccs.shape[0] < 228:
#         pad_width = 228 - mfccs.shape[0]
#         mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
#     elif mfccs.shape[0] > 228:
#         mfccs = mfccs[:228, :]
#
#     # Reshape to fit the model's input (1, 228, 40, 1)
#     mfccs = mfccs.reshape(1, 228, 40, 1)
#
#     return mfccs
#
#
# def predict_emotion(audio_file):
#     """
#     This function preprocesses the audio file and passes it through the model
#     to predict the emotion.
#     """
#     # Preprocess the audio file to extract features
#     audio_data = preprocess_audio(audio_file)
#
#     # Model prediction
#     prediction = model.predict(audio_data)
#     emotion_index = np.argmax(prediction)  # Get the emotion with the highest probability
#
#     # Mapping of emotion index to label and emoji
#     label_conversion = {
#         0: ('neutral', '😐'),
#         1: ('calm', '😌'),
#         2: ('happy', '😊'),
#         3: ('sad', '😢'),
#         4: ('angry', '😡'),
#         5: ('fear', '😱'),
#         6: ('disgust', '🤢'),
#         7: ('surprise', '😲')
#     }
#
#     # Get the corresponding emotion label and emoji
#     emotion_label, emotion_emoji = label_conversion.get(emotion_index, ("unknown", "❓"))
#     return emotion_label, emotion_emoji
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({"error": "No selected file"})
#
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#
#         # If it's an MP3 file, re-encode it to fix possible header issues
#         if filename.endswith('.mp3'):
#             try:
#                 reencoded_file_path = reencode_mp3(file_path)
#                 wav_file_path = convert_mp3_to_wav(reencoded_file_path)
#                 emotion_label, emotion_emoji = predict_emotion(wav_file_path)
#                 return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})
#             except subprocess.CalledProcessError as e:
#                 return jsonify({"error": f"FFmpeg error: {e}"})
#
#         # Process WAV files directly
#         emotion_label, emotion_emoji = predict_emotion(file_path)
#         return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})
#
#     return jsonify({"error": "Invalid file format"})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)



# import os
# import json
# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# from pydub import AudioSegment
# import librosa
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import model_from_json
#
# app = Flask(__name__)
#
# # Set the model folder path
# MODEL_FOLDER = 'model/'
#
# # Load model configuration from JSON file
# with open(os.path.join(MODEL_FOLDER, 'config.json'), 'r') as json_file:
#     model_json = json_file.read()
#
# # Recreate the model from the JSON configuration
# model = model_from_json(model_json)
#
# # Load the model weights
# model.load_weights(os.path.join(MODEL_FOLDER, 'model.weights.h5'))
#
# # Configure upload folder and allowed file types
# UPLOAD_FOLDER = 'static/audio_uploads'
# ALLOWED_EXTENSIONS = {'mp3', 'wav'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# # Explicitly set the path for ffmpeg and ffprobe
# from pydub.utils import which
#
# # Set the full path to your ffmpeg and ffprobe executables
# ffmpeg_path = r"D:\Download2\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
# ffprobe_path = r"D:\Download2\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe"
#
# # Set ffmpeg and ffprobe paths explicitly
# AudioSegment.ffmpeg = ffmpeg_path
# AudioSegment.ffprobe = ffprobe_path
#
# # Check if ffmpeg and ffprobe are correctly set
# print("ffmpeg path:", AudioSegment.ffmpeg)
# print("ffprobe path:", AudioSegment.ffprobe)
#
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# def convert_mp3_to_wav(mp3_file_path):
#     wav_file_path = mp3_file_path.rsplit('.', 1)[0] + '.wav'
#     audio = AudioSegment.from_file(mp3_file_path, format="mp3")
#     audio.export(wav_file_path, format="wav")
#     return wav_file_path
#
#
# def preprocess_audio(audio_file):
#     """
#     This function loads an audio file, extracts MFCC features,
#     and prepares it for prediction by the model.
#     """
#     # Load the audio file with librosa
#     data, sr = librosa.load(audio_file, sr=22050)  # Load at 22050Hz sample rate
#
#     # Extract MFCCs (40 coefficients as expected by the model)
#     mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
#     mfccs = mfccs.T  # Transpose to shape (frames, features)
#
#     # Pad or truncate to match model input timesteps (228)
#     if mfccs.shape[0] < 228:
#         pad_width = 228 - mfccs.shape[0]
#         mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
#     elif mfccs.shape[0] > 228:
#         mfccs = mfccs[:228, :]
#
#     # Reshape to fit the model's input (1, 228, 40, 1)
#     mfccs = mfccs.reshape(1, 228, 40, 1)
#
#     return mfccs
#
#
# def predict_emotion(audio_file):
#     """
#     This function preprocesses the audio file and passes it through the model
#     to predict the emotion.
#     """
#     # Preprocess the audio file to extract features
#     audio_data = preprocess_audio(audio_file)
#
#     # Model prediction
#     prediction = model.predict(audio_data)
#     emotion_index = np.argmax(prediction)  # Get the emotion with the highest probability
#
#     # Mapping of emotion index to label and emoji
#     label_conversion = {
#         0: ('neutral', '😐'),
#         1: ('calm', '😌'),
#         2: ('happy', '😊'),
#         3: ('sad', '😢'),
#         4: ('angry', '😡'),
#         5: ('fear', '😱'),
#         6: ('disgust', '🤢'),
#         7: ('surprise', '😲')
#     }
#
#     # Get the corresponding emotion label and emoji
#     emotion_label, emotion_emoji = label_conversion.get(emotion_index, ("unknown", "❓"))
#     return emotion_label, emotion_emoji
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({"error": "No selected file"})
#
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#
#         # If it's an MP3 file, convert it to WAV
#         if filename.endswith('.mp3'):
#             wav_file_path = convert_mp3_to_wav(file_path)
#             emotion_label, emotion_emoji = predict_emotion(wav_file_path)
#             return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})
#
#         # Process WAV files directly
#         emotion_label, emotion_emoji = predict_emotion(file_path)
#         return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})
#
#     return jsonify({"error": "Invalid file format"})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
#
# # import os
# # import json
# # from flask import Flask, render_template, request, jsonify
# # from werkzeug.utils import secure_filename
# # from pydub import AudioSegment
# # import librosa
# # import tensorflow as tf
# # import numpy as np
# # from tensorflow.keras.models import model_from_json
# #
# # app = Flask(__name__)
# #
# # # Set the model folder path
# # MODEL_FOLDER = 'model/'
# #
# # # Load model configuration from JSON file
# # with open(os.path.join(MODEL_FOLDER, 'config.json'), 'r') as json_file:
# #     model_json = json_file.read()
# #
# # # Recreate the model from the JSON configuration
# # model = model_from_json(model_json)
# #
# # # Load the model weights
# # model.load_weights(os.path.join(MODEL_FOLDER, 'model.weights.h5'))
# #
# # # Configure upload folder and allowed file types
# # UPLOAD_FOLDER = 'static/audio_uploads'
# # ALLOWED_EXTENSIONS = {'mp3', 'wav'}
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# #
# #
# # def allowed_file(filename):
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# #
# #
# # def convert_mp3_to_wav(mp3_file_path):
# #     wav_file_path = mp3_file_path.rsplit('.', 1)[0] + '.wav'
# #     audio = AudioSegment.from_file(mp3_file_path, format="mp3")
# #     audio.export(wav_file_path, format="wav")
# #     return wav_file_path
# #
# #
# # def preprocess_audio(audio_file):
# #     """
# #     This function loads an audio file, extracts MFCC features,
# #     and prepares it for prediction by the model.
# #     """
# #     # Load the audio file with librosa
# #     data, sr = librosa.load(audio_file, sr=22050)  # Load at 22050Hz sample rate
# #
# #     # Extract MFCCs (40 coefficients as expected by the model)
# #     mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
# #     mfccs = mfccs.T  # Transpose to shape (frames, features)
# #
# #     # Pad or truncate to match model input timesteps (228)
# #     if mfccs.shape[0] < 228:
# #         pad_width = 228 - mfccs.shape[0]
# #         mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
# #     elif mfccs.shape[0] > 228:
# #         mfccs = mfccs[:228, :]
# #
# #     # Reshape to fit the model's input (1, 228, 40, 1)
# #     mfccs = mfccs.reshape(1, 228, 40, 1)
# #
# #     return mfccs
# #
# #
# # def predict_emotion(audio_file):
# #     """
# #     This function preprocesses the audio file and passes it through the model
# #     to predict the emotion.
# #     """
# #     # Preprocess the audio file to extract features
# #     audio_data = preprocess_audio(audio_file)
# #
# #     # Model prediction
# #     prediction = model.predict(audio_data)
# #     emotion_index = np.argmax(prediction)  # Get the emotion with the highest probability
# #
# #     # Mapping of emotion index to label and emoji
# #     label_conversion = {
# #         0: ('neutral', '😐'),
# #         1: ('calm', '😌'),
# #         2: ('happy', '😊'),
# #         3: ('sad', '😢'),
# #         4: ('angry', '😡'),
# #         5: ('fear', '😱'),
# #         6: ('disgust', '🤢'),
# #         7: ('surprise', '😲')
# #     }
# #
# #     # Get the corresponding emotion label and emoji
# #     emotion_label, emotion_emoji = label_conversion.get(emotion_index, ("unknown", "❓"))
# #     return emotion_label, emotion_emoji
# #
# #
# # @app.route('/')
# # def index():
# #     return render_template('index.html')
# #
# #
# # @app.route('/upload', methods=['POST'])
# # def upload_file():
# #     if 'file' not in request.files:
# #         return jsonify({"error": "No file part"})
# #
# #     file = request.files['file']
# #
# #     if file.filename == '':
# #         return jsonify({"error": "No selected file"})
# #
# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         file.save(file_path)
# #
# #         # If it's an MP3 file, convert it to WAV
# #         if filename.endswith('.mp3'):
# #             wav_file_path = convert_mp3_to_wav(file_path)
# #             emotion_label, emotion_emoji = predict_emotion(wav_file_path)
# #             return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})
# #
# #         # Process WAV files directly
# #         emotion_label, emotion_emoji = predict_emotion(file_path)
# #         return jsonify({"emotion": f"{emotion_label} {emotion_emoji}"})
# #
# #     return jsonify({"error": "Invalid file format"})
# #
# #
# # if __name__ == '__main__':
# #     app.run(debug=True)
# # #
# # #
# # # # import os
# # # # import json
# # # # from flask import Flask, render_template, request, jsonify
# # # # from werkzeug.utils import secure_filename
# # # # from pydub import AudioSegment
# # # # import librosa
# # # # import tensorflow as tf
# # # # import numpy as np
# # # # from tensorflow.keras.models import model_from_json
# # # #
# # # # app = Flask(__name__)
# # # #
# # # # # Set the model folder path
# # # # MODEL_FOLDER = 'model/'
# # # #
# # # # # Load model configuration from JSON file
# # # # with open(os.path.join(MODEL_FOLDER, 'config.json'), 'r') as json_file:
# # # #     model_json = json_file.read()
# # # #
# # # # # Recreate the model from the JSON configuration
# # # # model = model_from_json(model_json)
# # # #
# # # # # Load the model weights
# # # # model.load_weights(os.path.join(MODEL_FOLDER, 'model.weights.h5'))
# # # #
# # # # # Configure upload folder and allowed file types
# # # # UPLOAD_FOLDER = 'static/audio_uploads'
# # # # ALLOWED_EXTENSIONS = {'mp3', 'wav'}
# # # # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # # #
# # # #
# # # # def allowed_file(filename):
# # # #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# # # #
# # # #
# # # # def convert_mp3_to_wav(mp3_file_path):
# # # #     wav_file_path = mp3_file_path.rsplit('.', 1)[0] + '.wav'
# # # #     audio = AudioSegment.from_file(mp3_file_path, format="mp3")
# # # #     audio.export(wav_file_path, format="wav")
# # # #     return wav_file_path
# # # #
# # # #
# # # # def preprocess_audio(audio_file):
# # # #     """
# # # #     This function loads an audio file, extracts MFCC features,
# # # #     and prepares it for prediction by the model.
# # # #     """
# # # #     # Load the audio file with librosa
# # # #     data, sr = librosa.load(audio_file, sr=22050)  # Load at 22050Hz sample rate
# # # #
# # # #     # Extract MFCCs (40 coefficients as expected by the model)
# # # #     mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
# # # #     mfccs = mfccs.T  # Transpose to shape (frames, features)
# # # #
# # # #     # Pad or truncate to match model input timesteps (228)
# # # #     if mfccs.shape[0] < 228:
# # # #         pad_width = 228 - mfccs.shape[0]
# # # #         mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
# # # #     elif mfccs.shape[0] > 228:
# # # #         mfccs = mfccs[:228, :]
# # # #
# # # #     # Reshape to fit the model's input (1, 228, 40, 1)
# # # #     mfccs = mfccs.reshape(1, 228, 40, 1)
# # # #
# # # #     return mfccs
# # # #
# # # #
# # # # def predict_emotion(audio_file):
# # # #     """
# # # #     This function preprocesses the audio file and passes it through the model
# # # #     to predict the emotion.
# # # #     """
# # # #     # Preprocess the audio file to extract features
# # # #     audio_data = preprocess_audio(audio_file)
# # # #
# # # #     # Model prediction
# # # #     prediction = model.predict(audio_data)
# # # #     emotion = np.argmax(prediction)  # Get the emotion with the highest probability
# # # #     return emotion
# # # #
# # # #
# # # # @app.route('/')
# # # # def index():
# # # #     return render_template('index.html')
# # # #
# # # #
# # # # @app.route('/upload', methods=['POST'])
# # # # def upload_file():
# # # #     if 'file' not in request.files:
# # # #         return jsonify({"error": "No file part"})
# # # #
# # # #     file = request.files['file']
# # # #
# # # #     if file.filename == '':
# # # #         return jsonify({"error": "No selected file"})
# # # #
# # # #     if file and allowed_file(file.filename):
# # # #         filename = secure_filename(file.filename)
# # # #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# # # #         file.save(file_path)
# # # #
# # # #         # If it's an MP3 file, convert it to WAV
# # # #         if filename.endswith('.mp3'):
# # # #             wav_file_path = convert_mp3_to_wav(file_path)
# # # #             emotion = predict_emotion(wav_file_path)
# # # #             return jsonify({"emotion": int(emotion)})  # Convert numpy.int64 to int
# # # #
# # # #         # Process WAV files directly
# # # #         emotion = predict_emotion(file_path)
# # # #         return jsonify({"emotion": int(emotion)})  # Convert numpy.int64 to int
# # # #
# # # #     return jsonify({"error": "Invalid file format"})
# # # #
# # # #
# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)
