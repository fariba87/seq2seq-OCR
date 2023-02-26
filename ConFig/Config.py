import json, os , numpy as np


class ConfigReader():
    def __init__(self, conf_path= "media/SSD1TB/rezaei/Projects/GuidedCTCOCR/guidedctcocr/ConFig/config.json"):
        with open("/media/SSD1TB/rezaei/Projects/GuidedCTCOCR/guidedctcocr/ConFig/config.json", "r") as f:
            cfg = json.load(f)

        self.modelName = cfg["modelName"]
        self.modelType = cfg["modelType"]
        #self.flipProb = float(cfg["flipProb"])
        #self.jitterProb = float(cfg["jitterProb"])
        #self.cutProb = float(cfg["cutProb"])
        #self.modelDebug = cfg["modelDebug"] == "True"
        self.SanityCheck = cfg["SanityCheck"] == "True"
        self.InChannel  = np.int0(cfg["InChannel"])
        self.batchSize  = np.int0(cfg["batchSize"])
        self.TotalEpoch = np.int0(cfg["TotalEpoch"])
        self.targetHeight = np.int0(cfg["TargetHeight"])
        self.MaxWidthTarget =np.int0(cfg["MaxWidthTarget"])
        self.NumGRUlayer = np.int0(cfg["NumGRUlayer"])
        self.NumGRUunit = np.int0(cfg["NumGRUunit"])
        self.LSTMunit = np.int0(cfg["LSTMunit"])
        self.num_layers_tr = np.int0(cfg["num_layers_tr"])
        self.d_model_tr = np.int0(cfg["d_model_tr"])
        self.num_heads_tr = np.int0(cfg["num_heads_tr"])
        self.dff_tr = np.int0(cfg["dff_tr"])
        self.dropout_rate_tr = np.float32(cfg["dropout_rate_tr"])
        self.SeqDivider = cfg["SeqDivider"]
        self.lr =np.float32(cfg["lr"])
        self.Shuffle = cfg["Shuffle"]
        self.lr_tr = cfg["lr_tr"]
        self.lr_ctc =cfg["lr_ctc"]
#        self.Charset =np.load('') # this is one of the output from dataloader.py
        A = 26# = np.max(mxlength in all sequence)
        '''
        in dataloader.py for each dataset i return a parameter as mxlen which is max length in that dataset
        I can also access it from max(len(df['label'].iloc))
        '''
        self.MaxSeqLength = A
        self.MainPath = "/media/Archive4TB3/Data/textImages/EN_Benchmarks/"

ConfigReader()