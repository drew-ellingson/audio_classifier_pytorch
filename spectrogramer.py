import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import numpy as np
import os
import functools 
import time


def timer(func):
    """timing decorator for training and testing methods"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        print('Finished {} in {} seconds'.format(func.__name__, round((end_time - start_time),3)))
        return value
    return wrapper_timer


def make_spec(input_folder, input_file, reload=True):
    input_subdir = input_folder[input_folder.index('/'):]
    export_path = './raw_spectrograms/'+input_subdir
    input_file_sans_ext = input_file[:-4]
    export_file = input_file_sans_ext+'.png'

    try:
        os.mkdir(export_path)
    except FileExistsError:
        pass 

    if (not reload and os.path.exists(export_path+'/'+export_file)):
        print('already got {}'.format(export_file))
        return None 

    y, sr = librosa.load(input_folder+'/'+input_file)

    n_fft = 2048
    hop_length = 256
    D = np.abs(librosa.stft(y, n_fft=n_fft,
                            hop_length=hop_length))

    D = librosa.amplitude_to_db(D, ref=np.max)

    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')
    plt.savefig(export_path+'/'+export_file, 
        pad_inches=0, bbox_inches='tight')
    plt.close()

    return None 


@timer
def all_make_spec(reload=False):
    i = 0
    tot = sum([len(x[2]) for x in os.walk('processed_wav')]) # file count
    start_time = time.time()
    for root, _, filenames in os.walk('processed_wav'):
        for filename in filenames:
            i += 1 
            if not filename.endswith('.wav'):
                continue
            else:
                make_spec(root, filename, reload=reload)
                if i % 50 == 49: # lowering this as i get increasingly impatient
                    print('{} out of {} done for approx {} percent completion at time {}'
                        .format(i+1, 
                            tot, 
                            round(100*(i+1)/tot, 3), 
                            round(time.time()-start_time, 3)))
    return None


if __name__ == '__main__':
    all_make_spec()