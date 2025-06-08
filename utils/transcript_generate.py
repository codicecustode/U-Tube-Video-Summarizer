from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id: str):
  print(f"getting transcript for video_id: {video_id}")
  try:
    transcripts = YouTubeTranscriptApi.get_transcript(video_id)
    transcripted_data = ''
    for transcript in transcripts:
      transcripted_data += transcript['text'] + ' '
    return transcripted_data
  except Exception as e:
    print(f"an error occurred: {e}")
    return None
  
#get_transcript('h7Xhfui4aU0')

