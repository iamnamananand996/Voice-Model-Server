# from IPython.display import Audio
# from IPython.utils import io

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa

from flask import Flask, jsonify, request, send_file
import os
import uuid

app = Flask(__name__)


encoder_weights = Path("encoder/saved_models/pretrained.pt")
vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("synthesizer/saved_models/logs-pretrained/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)


PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    path_to_file = "output.wav"

    return send_file(
        path_to_file,
        mimetype="audio/wav",
        as_attachment=True,
        attachment_filename="output.wav")

    # return jsonify({'msg': 'server running'})


@app.route("/voice", methods=['POST'])
def convert_voice():

    if request.method == 'POST':
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1]

        f_name = str(uuid.uuid4()) + extension
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f_name)

        # @param {type:"string"}
        text = "This is being said in my own voice.  The computer has learned to do an impression of me."
        in_fpath = Path(
            "/home/naman/melnetCode/clone_in_5_sec/dataset/data_voice/test.wav")
        reprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(in_fpath)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Synthesizing new audio...")

        specs = synthesizer.synthesize_spectrograms([text], [embed])
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = np.pad(
            generated_wav, (0, synthesizer.sample_rate), mode="constant")
        # display(Audio(generated_wav, rate=synthesizer.sample_rate))
        librosa.output.write_wav(
            'output.wav', generated_wav, synthesizer.sample_rate)

        path_to_file = "output.wav"

        return send_file(
            path_to_file,
            mimetype="audio/wav",
            as_attachment=True,
            attachment_filename="output.wav")

    # return jsonify({'msg': 'data convert success', 'filename': f_name})


app.run(debug=True)
