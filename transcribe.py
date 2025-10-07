import requests

def transcribe_audio(base_url: str, audio_file_path: str, translate: bool = False):
    """
    Upload audio file and get transcription from backend.
    """
    url = f"{base_url.rstrip('/')}/transcribe"
    with open(audio_file_path, "rb") as f:
        files = {"audio": f}
        data = {"translate": str(translate).lower()}
        response = requests.post(url, files=files, data=data)
    response.raise_for_status()
    result = response.json()

    if result.get("success"):
        print("Transcription:", result["transcription"]["original_text"])
        print("Punctuated:", result["punctuation"].get("punctuated_text", "N/A"))
        print("Detected Language:", result["language_detection"]["detected_language"])
        if "translation" in result:
            print("Translation:", result["translation"].get("translated_text", "N/A"))
    else:
        print("Request failed:", result)

    return result


if __name__ == "__main__":
    result = transcribe_audio(
        "https://e1389f0b40fe.ngrok-free.app",
        "/home/ubuntu/agrichat-annam/19 Q-A.mp3",
        translate=True
    )