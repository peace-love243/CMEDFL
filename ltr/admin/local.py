class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/CMEDFL/pytracking/networks'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.vos_dir = '/home/traindata/yutubevos/train'
        self.results_dir = '/home/CMEDFL/pytracking/results'
