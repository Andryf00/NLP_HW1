#Very simple vocabulary class, simply composed of a dictionary mapping from string to idx and viceversa
class Vocab():
    def __init__(self) -> None:
        self.stoi = {}#string2idx
        self.itos = []

    def __len__(self):
        return len(self.itos)