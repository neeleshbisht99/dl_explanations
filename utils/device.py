import torch

class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Device(metaclass=SingletonMeta):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_device(self, device_id):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device