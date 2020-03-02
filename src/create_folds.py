import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    train_df = pd.read_csv(f"{os.path.dirname(os.getcwd())}/input/train.csv")
    train_df.loc[:,'k_fold'] = -1

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    
    X = train_df.image_id.values
    y = train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
    
    mskf = MultilabelStratifiedKFold(n_splits=5)
    
    for fold, (train_idxs, val_idxs) in enumerate(mskf.split(X, y)):
        train_df.loc[val_idxs,'k_fold'] = fold

    print(train_df.k_fold.value_counts())
    train_df.to_csv(f"{os.path.dirname(os.getcwd())}/input/train_folds.csv", index=False)        