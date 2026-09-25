# Neutron/Gamma Acoustic Discrimination — Modular Project

Split into the pieces you asked for: signal preprocessing, model training,
model inference, and the UI — each in its own file(s), on both the Python
side and the browser side.

```
neutron_gamma_project/
├── python/
│   ├── preprocessing.py     # signal preprocessing (LVM parsing → 512-D STFT feature vector)
│   ├── train_model.py       # trains LR + Linear SVM, exports model_weights.json / .js
│   ├── predict.py           # applies a trained model to ONE new .lvm file (CLI)
│   ├── model_weights.json   # already-trained weights (30,000 epochs, full dataset)
│   └── training_report.json # accuracy / precision / recall / F1 / confusion matrices
├── requirements.txt
└── web_app/
    ├── index.html            # the UI shell
    ├── style.css              # all styling
    └── js/
        ├── sample_data.js     # pre-computed results for the sample gallery/charts
        ├── model_weights.js   # same trained weights as the .json, as a JS constant
        ├── preprocessing.js   # signal preprocessing — JS mirror of preprocessing.py
        ├── model.js           # applies the model to a feature vector (w·x + b)
        └── app.js              # UI logic: renders charts, wires file-upload → preprocessing.js → model.js → screen
```

Each file does exactly one job, and the Python and JS preprocessing files
are line-for-line the same algorithm — verified to produce identical
feature vectors and identical predictions on real waveform files.

## Using the UI (no install needed)

Open `web_app/index.html` in a browser (just double-click it). Scroll to
**"Live classifier"**, drop in any `.lvm` file, and the page:

1. `preprocessing.js` parses it and computes the 512-D feature vector
2. `model.js` applies the trained weights (`model_weights.js`) to get a
   prediction
3. `app.js` draws the waveform, short-time-energy curve, spectrogram, and
   the two models' predictions on screen

Everything runs locally in your browser — no server, no upload, no
internet required.

## Retraining the model (Python side)

```bash
pip install -r requirements.txt

python python/train_model.py --data-dir "Data_100 events" --epochs 30000
```

This regenerates `model_weights.json` **and** `model_weights.js` — copy
the new `model_weights.js` into `web_app/js/` to update the UI with your
retrained model.

## Running a single prediction from the command line (Python side)

```bash
python python/predict.py --weights python/model_weights.json --input scope-1_026.lvm
```

Add `--plot out.png` to also save a waveform/STE/spectrogram figure.

## Why two copies of the pipeline (Python + JS)?

The UI needs to work by just opening a file in a browser — no Python, no
server — so the same preprocessing + inference logic is re-implemented in
JavaScript (`web_app/js/preprocessing.js`, `model.js`). They're kept
side-by-side deliberately so you can read and compare them; if you ever
change one (e.g. the pulse-localization threshold `THETA`), change it in
both files.
