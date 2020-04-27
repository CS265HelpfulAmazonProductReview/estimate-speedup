# code used to estimate the speedup

preprocessing: `spark-submit preprocess.py`
training: `spark-submit train.py`
predicting: `spark-submit predict.py`

The raw data used is `data/reviews.json`. It contains the musicual instrument reviews. We can use other subsets of the full dataset. The code only works with the old dataset (2014 version)