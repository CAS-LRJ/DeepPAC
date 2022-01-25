class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)


result_dict = {1: 'Network is PAC-model robust.',
               0: 'Unsafe. Adversarial Example Found.',
               2: 'Unknown. Potential Counter-Example exists.'}
