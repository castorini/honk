from googleapiclient.discovery import build
from url_fetcher import UrlFetcher
from utils import color_print as cp

class YoutubeSearcher(UrlFetcher):
    def __init__(self, api_key, keyword, batch_size=50):
        super().__init__()
        self.keyword = keyword
        self.api_key = api_key
        self.token = None
        self.batch_size = batch_size

    def search_videos(self, query, max_results=50, token=None):
        if token == "last_page":
            cp.print_warning("No more search results available for ", query)
            return (None, [])
        youtube = build("youtube", "v3", developerKey=self.api_key)
        search_response = youtube.search().list(
            q=query,
            type="video",
            pageToken=token,
            order="relevance",
            part="id,snippet",
            maxResults=max_results,
            location=None,
            locationRadius=None
        ).execute()

        videos = []

        for search_result in search_response.get("items", []):
            if search_result["id"]["kind"] == "youtube#video":
                videos.append(search_result)

        if "nextPageToken" in search_response:
            next_token = search_response["nextPageToken"]
        else:
            next_token = "last_page"
        return (next_token, videos)

    def fetch_next_batch(self):
        results = self.search_videos(self.keyword, max_results=self.batch_size, token=self.token)
        self.token = results[0]
        batch = [video_data['id']['videoId'] for video_data in results[1]]
        return batch

    def reset(self):
        self.token = None
        super.reset()
