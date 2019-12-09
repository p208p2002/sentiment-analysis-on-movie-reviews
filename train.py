from core import MR_Data,makeTorchDataSet
if __name__ == "__main__":
    TrainData = MR_Data.load_data('dataset/train.tsv')
    makeTorchDataSet(TrainData)