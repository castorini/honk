import re
import color_print as cp

URL_TEMPLATE = "http://youtube.com/watch?v={}"

def get_youtube_url(vid):
    return URL_TEMPLATE.format(vid)

SRT_TIME_PARSER = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s-->\s(\d+):(\d+):(\d+),(\d+)")

def srt_time_to_ms(hour, minute, second, msecond):
    converted = int(msecond)
    converted += (1000 * int(second))
    converted += (1000 * 60 * int(minute))
    converted += (1000 * 60 * 60 * int(hour))
    return converted

def parse_srt_time(time):
    match_result = SRT_TIME_PARSER.match(time)

    start_pos = None
    stop_pos = None

    if match_result:
        start_time_ms = srt_time_to_ms(
            match_result.group(1),
            match_result.group(2),
            match_result.group(3),
            match_result.group(4))
        stop_time_ms = srt_time_to_ms(
            match_result.group(5),
            match_result.group(6),
            match_result.group(7),
            match_result.group(8))

        # * 16 because the audio has sample rate of 16000
        start_pos = start_time_ms * 16
        stop_pos = stop_time_ms * 16

    else:
        cp.print_warning("failed to parse srt time - ", time)

    return start_pos, stop_pos
