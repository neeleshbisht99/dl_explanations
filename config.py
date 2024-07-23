class Config:
    def __init__(self):
        self.img_size = 128
        self.depth = 64

class CnnConfig:
    def __init__(self):
        config = Config()
        self.img_size= config.img_size
        self.depth=config.depth

        self.learning_rate= 1e-4
        self.lr_decay_rate= 0.96
        self.epochs= 100
        self.batch_size= 4
        self.gpu_id= 3
        self.model_path= "3d_cnn_image_classification_508.pth"

class ResnetConfig:
    def __init__(self):
        config = Config()
        self.img_size= config.img_size
        self.depth=config.depth

        self.n_classes = 2
        self.learning_rate = 0.000008 #change
        self.phase = 'train'
        self.batch_size = 4
        self.epochs = 300
        self.max_epochs = 230 #change
        self.input_D = 64 # not used
        self.input_H = 128 # not used
        self.input_W = 128 # not used
        self.pretrain_path = 'models/archive/tecent_med3d_pretrain/resnet_34_23dataset.pth' #change
        self.new_layer_names = ['conv_seg']
        self.no_cuda = False
        self.model = 'resnet'
        self.gpu_id = [3, 4] 
        self.model_depth = 34 #change
        self.resnet_shortcut = 'A' #change
        self.manual_seed = 27
        self.ci_test = False
        self.weight_decay= 0.005 #change
        self.model_path="3d_resnet34_image_classification_i230.pth" #change