class Config:
    def __init__(self):
        self.img_size = 224
        self.depth = 42
        self.img_width = 336
        self.img_height = 224

class CnnConfig:
    def __init__(self):
        config = Config()
        self.img_size= config.img_size
        self.depth=config.depth
        self.img_width = config.img_width
        self.img_height = config.img_height

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
        self.img_width = config.img_width
        self.img_height = config.img_height

        self.n_classes = 2
        self.learning_rate = 1e-5 #change
        self.phase = 'train'
        self.batch_size = 1
        self.epochs = 90
        self.max_epochs = 100 #change
        self.input_D = 64 # not used
        self.input_H = 128 # not used
        self.input_W = 128 # not used
        self.pretrain_path = None #change
        self.new_layer_names = ['conv_seg']
        self.no_cuda = False
        self.model = 'resnet'
        self.gpu_id = [4, 2] 
        self.model_depth = 6 #change
        self.resnet_shortcut = 'B' #change
        self.manual_seed = 27
        self.ci_test = False
        self.weight_decay= 1e-10 #change
        self.model_path="3d_resnet6_image_classification_i100.pth" #change