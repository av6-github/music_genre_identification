import os
import librosa
import math
import json

DATASET_PATH = "dataset_compr/genres"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050 # loads audio at 22050 samples per second
DURATION = 30 # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5): # loops through each genre folder, splits each audio file into smaller segments, extracts the mfcc features and stores into json file
    
    # dictionary to store data
    data = {
        "mapping" : [],
        "mfcc" : [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK/ num_segments)
    expected_num_mfcc_per_segment = math.ceil(num_samples_per_segment/hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we are not at root level
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1] # getting last subfolder
            data["mapping"].append(semantic_label)

            print(f"\n Processing {semantic_label}")

            # process files for specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE) # loads audio waverform

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s=0 --> 0
                    finish_sample =  start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length) # extracts mfccs from each segment
                    mfcc = mfcc.T


                    # store mfcc for segment if it has expected legth
                    if len(mfcc) == expected_num_mfcc_per_segment:
                        data["mfcc"].append(mfcc.tolist()) # converts matrix to json storable list
                        data["labels"].append(i-1)
                        print(f"{file_path}, segment: {s+1}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)