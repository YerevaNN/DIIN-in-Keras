import threading


class BatchGenerator(object):
    def __init__(self, shuffle=True):
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.index = 0

    def next(self):
        with self.lock:
            return None
