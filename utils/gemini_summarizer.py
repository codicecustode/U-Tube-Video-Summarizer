from google import genai
from dotenv import load_dotenv
import os
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def summarize_text_gemini(text):
    """
    Summarizes the given text using Google Gemini.
    
    Args:
        text (str): The text to summarize.
        
    Returns:
        str: The summary of the text.
    """
    
    input_text = (
      "You are an intelligent summarizer. Given a YouTube video transcript, generate a clear, visually appealing summary in bullet points format.\n\n"
      "Requirements:\n"
      "- Use only the most important insights, facts, or key takeaways\n"
      "- Start each bullet with a fitting emoji like 💡 (idea), 🎯 (key point), 📌 (fact), or 🚀 (tip) based on the content\n"
      "- Add one blank line between each bullet point for better readability\n"
      "- Use simple, clear language that’s easy to scan and understand\n"
      "- Do NOT include unimportant information, repetition, or long paragraphs\n\n"
      "Here is the YouTube video transcript:\n\n"
      f"{text}"
  )


    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[input_text]
    )
    return response.text
