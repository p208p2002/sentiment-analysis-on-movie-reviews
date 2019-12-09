from core import MR_Data, makeTorchDataSet, makeTorchDataLoader
from transformers import AlbertConfig,AlbertForSequenceClassification
if __name__ == "__main__":
    TrainData = MR_Data.load_data('dataset/train.tsv')
    TrainDataset = makeTorchDataSet(TrainData)
    TrainDataLoader = makeTorchDataLoader(TrainDataset)
    model_config = AlbertConfig.from_json_file('albert-large-config.json')
    model = AlbertForSequenceClassification.from_pretrained('albert-large-pytorch_model.bin',config = model_config)

    # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    # model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
    # input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    # labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids, labels=labels)
    # loss, logits = outputs[:2]