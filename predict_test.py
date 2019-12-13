import torch
from core import MR_Data, makeTorchDataSet, makeTorchDataLoader, blockPrint, enablePrint, log, computeAccuracy, saveModel, splitDataset
from transformers import AlbertConfig, AlbertForSequenceClassification, AdamW
import numpy as np

def main():
    #
    blockPrint()

    # setting device
    device = torch.device('cuda')

    #
    FullData = MR_Data.load_data('dataset/test.tsv',is_train_data=False)
    FullDataset = makeTorchDataSet(FullData,is_train_data=False)
    TestDataLoader = makeTorchDataLoader(FullDataset, batch_size=16)
    model_config = AlbertConfig.from_json_file('model/albert-large-config.json')
    trained_model_file = '12-11-2019_09-17-05_ALSS_e5_a69.24226892192033'
    model = AlbertForSequenceClassification.from_pretrained('train_models/'+ trained_model_file +'/pytorch_model.bin', config = model_config)
    
    model.to(device)
    model.eval()

    f = open('submission.csv','w',encoding='utf-8')
    f.write('PhraseId,Sentiment\n')
    log("please waiting for predict ....")
    for batch_index, batch_dict in enumerate(TestDataLoader):
        batch_dict = tuple(t.to(device) for t in batch_dict)
        input_ids,phrase_ids = batch_dict
        outputs = model(input_ids)
        
        outputs = outputs[0].cpu()
        outputs = outputs.detach().numpy()
        # log(outputs)
        
        for i in range(len(outputs)):
            p_id = phrase_ids[i].item()
            s_level = np.argmax(outputs[i])
            # log("phrase_id",p_id,"segment_level",s_level)
            f.write(str(p_id) + ',' + str(s_level) + '\n')
    
    f.close()

if __name__ == '__main__':
    main()
    
    