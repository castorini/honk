class UrlFetcher():
    def __init__(self):
        self.video_ids = []

    def fetch_next_batch(self):
        pass

    def next(self, size=1):
        while len(self.video_ids) < size:
            batch = self.fetch_next_batch()
            if not batch:
                break
            self.video_ids += batch

        video_ids = self.video_ids[:size]
        self.video_ids = self.video_ids[size:]
        return video_ids

    def reset(self):
        self.video_ids = []

    def size(self):
        return len(self.video_ids)
