class CnnConfig:
    def __init__(self):
        self.learning_rate= 1e-4
        self.lr_decay_rate= 0.96
        self.epochs= 100
        self.batch_size= 4
        self.img_size= 128
        self.depth=64
        self.gpu_id= 3
        self.model_path= "3d_cnn_image_classification_508.pth"



class ResnetConfig:
    def __init__(self):
        self.n_seg_classes = 2
        self.learning_rate = 1e-4
        self.phase = 'train'
        self.batch_size = 4
        self.epochs = 100
        self.input_D = 64 # not used
        self.input_H = 128 # not used
        self.input_W = 128 # not used
        self.pretrain_path = 'models/archive/tecent_med3d_pretrain/resnet_10.pth'
        self.new_layer_names = ['conv_seg']
        self.no_cuda = False
        self.model = 'resnet'
        self.gpu_id = [0, 3]
        self.model_depth = 10
        self.resnet_shortcut = 'B'
        self.manual_seed = 27
        self.ci_test = False

        self.img_size = 128
        self.depth = 64
        self.weight_decay= 1e-6
        self.model_path="3d_resnet_image_classification_508.pth"
