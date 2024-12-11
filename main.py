from youtube_transcript_api import YouTubeTranscriptApi


try:
  data = ""
  transcript_list = YouTubeTranscriptApi.list_transcripts('JY0FN71vCbw')
  for transcript in transcript_list:
    transcript_data = transcript.fetch()
    for text in transcript_data:
      data += text['text'] + " "
  print(data)
except Exception as e:
  print(f"An error occurred: {e}")
  print(e)