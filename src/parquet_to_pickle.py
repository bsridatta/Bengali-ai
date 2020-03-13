import os
import pandas as pd
import numpy as numpy
import joblib
import glob
from tqdm import tqdm

if __name__ == "__main__":
    for i in range(4):
        files = glob.glob(f'{os.path.dirname(os.getcwd())}/input/train_image_data_{i}.parquet')
        for file in files:
            df =  pd.read_parquet(file)
            image_ids = df.image_id.values
            df = df.drop('image_id', axis=1)
            image_array = df.values
            for idx, img_id in tqdm(enumerate(image_ids)):
                joblib.dump(image_array[idx, :], f'{os.path.dirname(os.getcwd())}/input/image_pickles/{img_id}.pkl')



            