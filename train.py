from core import MR_Data, makeTorchDataSet, makeTorchDataLoader, blockPrint, enablePrint
from transformers import AlbertConfig, AlbertForSequenceClassification, AdamW
import torch

def log(*logs):
    enablePrint()
    print(*logs)
    blockPrint()

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

if __name__ == "__main__":
    #
    device = torch.device('cuda')

    #
    TrainData = MR_Data.load_data('dataset/train.tsv')
    TrainDataset = makeTorchDataSet(TrainData)
    TrainDataLoader = makeTorchDataLoader(TrainDataset,batch_size=16)
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
    for epoch in range(15):
        running_loss_val = 0.0
        running_acc = 0.0
        for batch_index, batch_dict in enumerate(TrainDataLoader):
            model.train()
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(batch_dict[0], labels=batch_dict[1])
            loss, logits = outputs[:2]
            loss.sum().backward()
            optimizer.step()
            model.zero_grad()

            # compute the loss
            loss_t = loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(logits, batch_dict[1])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # log
            log("epoch:%2d batch:%4d train_loss:%2.4f train_acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
    
        # save model
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained('ALSS_e%s_a%S.model'%(str(epoch+1),str(running_acc)))