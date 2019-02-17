from .url_fetcher import UrlFetcher

class FileReader(UrlFetcher):
    def __init__(self, file_name, batch_size=50):
        super().__init__()
        self.file_name = file_name
        self.batch_size = batch_size
        with open(file_name, 'r') as file:
            self.lines = [line.rstrip() for line in file.readlines()]

    def fetch_next_batch(self):
        batch = self.lines[:self.batch_size]
        self.lines = self.lines[self.batch_size:]
        return batch

    def reset(self):
        super.reset()
        with open(self.file_name, 'r') as file:
            self.lines = file.readline()
