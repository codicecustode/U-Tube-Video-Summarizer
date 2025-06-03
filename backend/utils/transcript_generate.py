from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id: str):
  print(f"getting transcript for video_id: {video_id}")
  try:
    print(f"video_id1: {video_id}")
    transcripts = YouTubeTranscriptApi.get_transcript(video_id)
    transcripted_data = ''
    print(f"video_id2: {video_id}")

    for transcript in transcripts:
      transcripted_data += transcript['text'] + ' '
    print(f"video_id2: {video_id}")
    print(f"transcripted_data: {transcripted_data}")
    return transcripted_data
  except Exception as e:
    print(f"an error occurred: {e}")
    return None
  
#get_transcript('kY14KfZQ1TI')

