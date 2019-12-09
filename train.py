from core import MR_Data, makeTorchDataSet, makeTorchDataLoader
from transformers import AlbertConfig,AlbertForSequenceClassification
if __name__ == "__main__":
    TrainData = MR_Data.load_data('dataset/train.tsv')
    TrainDataset = makeTorchDataSet(TrainData)
    TrainDataLoader = makeTorchDataLoader(TrainDataset)
    model_config = AlbertConfig.from_json_file('albert-large-config.json')
    model = AlbertForSequenceClassification.from_pretrained('albert-large-pytorch_model.bin')