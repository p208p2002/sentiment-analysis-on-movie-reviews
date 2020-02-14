# sentiment-analysis-on-movie-reviews
using [ALBERT](https://arxiv.org/abs/1909.11942) on sentiment analysis
![score](https://raw.githubusercontent.com/p208p2002/sentiment-analysis-on-movie-reviews/master/score.png)
> Rank 9/2667 of the Public Leaderboard

## Dataset
[Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

## Albert Model Files
- base on albert-large
- [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-spiece.model)
- [model](https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-pytorch_model.bin)
- [config_file](https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-config.json)
> use `download.py` to get all model files we need

## Usage
- `pip install -r requments.txt`
- `python download.py`
> `download.py` is under path "model"
- `python train.py`

## Env Require
- python 3.6+
- pytorch 1.3+
