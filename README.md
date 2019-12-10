# sentiment-analysis-on-movie-reviews
using Albert on sentiment analysis

## Dataset
[Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

## Albert Model Files
- [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-spiece.model)
- [model](https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-pytorch_model.bin)
- [config_file](https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-config.json)
> use `download.py` to get all model files we need

## Usage
- `pip install -r requments.txt`
- `python download.py`
- `python train.py`
