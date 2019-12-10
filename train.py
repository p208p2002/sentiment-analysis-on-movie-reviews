from core import MR_Data, makeTorchDataSet, makeTorchDataLoader, blockPrint, enablePrint
from transformers import AlbertConfig, AlbertForSequenceClassification, AdamW
import torch

def log(*logs):
    enablePrint()
    print(*logs)
    blockPrint()

if __name__ == "__main__":
    #
    device = torch.device('cuda')

    #
    TrainData = MR_Data.load_data('dataset/train.tsv')
    TrainDataset = makeTorchDataSet(TrainData)
    TrainDataLoader = makeTorchDataLoader(TrainDataset,batch_size=24)
    model_config = AlbertConfig.from_json_file('albert-large-config.json')
    model = AlbertForSequenceClassification.from_pretrained('albert-large-pytorch_model.bin',config = model_config)
    model.to(device)

    #
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)

    model.zero_grad()
    for epoch in range(30):
        for index,batch_dict in enumerate(TrainDataLoader):
            model.train()
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(batch_dict[0], labels=batch_dict[1])
            loss, logits = outputs[:2]
            loss.sum().backward()
            optimizer.step()
            model.zero_grad()

            # compute the loss
            loss_t = loss.item()
            log(epoch,loss_t)