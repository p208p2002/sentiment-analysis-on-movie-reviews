from core import MR_Data, makeTorchDataSet, makeTorchDataLoader, blockPrint, enablePrint, log, computeAccuracy, saveModel, splitDataset
from transformers import AlbertConfig, AlbertForSequenceClassification, AdamW
import torch

def main():
    # setting device
    device = torch.device('cuda')

    #
    FullData = MR_Data.load_data('dataset/train.tsv')
    FullDataset = makeTorchDataSet(FullData)
    TrainDataset, TestDataset =  splitDataset(FullDataset, 0.9)
    TrainDataLoader = makeTorchDataLoader(TrainDataset, batch_size=16)
    TestDataLoader = makeTorchDataLoader(TestDataset, batch_size = 8)
    model_config = AlbertConfig.from_json_file('model/albert-large-config.json')
    model = AlbertForSequenceClassification.from_pretrained('model/albert-large-pytorch_model.bin', config = model_config)
    model.to(device)

    #
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)

    model.zero_grad()

    try:
        for epoch in range(15):
            # train
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
                acc_t = computeAccuracy(logits, batch_dict[1])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # log
                if(batch_index % 50 == 0):
                    log(">> TRAIN << epoch:%2d batch:%4d loss:%2.4f acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
            
            # test
            running_loss_val = 0.0
            running_acc = 0.0
            for batch_index, batch_dict in enumerate(TestDataLoader):
                model.eval()
                batch_dict = tuple(t.to(device) for t in batch_dict)
                outputs = model(batch_dict[0], labels=batch_dict[1])
                loss, logits = outputs[:2]

                # compute the loss
                loss_t = loss.item()
                running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

                # compute the accuracy
                acc_t = computeAccuracy(logits, batch_dict[1])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # log
                if(batch_index % 50 == 0):
                    log(">> TEST << epoch:%2d batch:%4d loss:%2.4f acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))
                    
            # save model
            saveModel(model,'ALSS_e%s_a%s'%(str(epoch+1),str(running_acc)))
    
    except KeyboardInterrupt:
        saveModel(model,'Interrupt_ALSS_e%s_a%s'%(str(epoch+1),str(running_acc)))

    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    main()