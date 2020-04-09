# Audio Classification (via Spectrograms)

### TODO:

- Split common code out of each classifier file
- Add support for arbitrary number of classes.
- Add main to run process end-to-end
- Add support for saving and comparing different models.
- Replace spectrograms with MFCC spectrograms
- Experiement with smarter NN architecture.

### To Use:

1. Place mp3s in 'raw_mp3' folder, with a separate folder for each artist.
2. Run `audio_chopper.py`, verify that 5s `.wav` files have been placed into `processed_wav` directory.
3. Run `spectrogramer.py`, verify `.png` files have been placed into `raw_spectrograms` directory,
4. Run `audio_classifier.py`. Testing results are outputted as part of this process.

### What's in the Repo

'Happy Birthday' is included as a public domain example of the pipeline taking input mp3 -> segmented wav files -> spectrogram images.

Other mp3s can be included locally and should be added under 'raw_mp3s' directory.