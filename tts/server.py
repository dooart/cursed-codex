import json
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import scipy.io.wavfile
from pocket_tts import TTSModel

from llm import generate, normalize
from tcp_listener import TcpListener
from event_filter import EventFilter

print("Loading model...")
tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt("./voice.safetensors")
print("Model loaded!")

busy = threading.Lock()

OUTPUT_FILE = "/tmp/tts_server_output.wav"
MAX_TEXT_LEN = 200
event_filter = EventFilter()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)
        text = data.get("text", "")

        if not text:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Missing "text" field')
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
        handle_event(text)


def handle_event(text):
    """Shared pipeline: LLM generate -> TTS -> play. Drops event if busy."""
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN] + "..."
    if not busy.acquire(blocking=False):
        print(f"Busy, skipping: {text}")
        return

    def generate_and_play():
        try:
            print(f"Event: {text}")
            quip = generate(text)
            speech = normalize(quip)
            print(f"  LLM: {quip}")
            print(f"  TTS: {speech}")

            audio = tts_model.generate_audio(voice_state, speech)
            scipy.io.wavfile.write(
                OUTPUT_FILE, tts_model.sample_rate, audio.numpy()
            )
            subprocess.run(["afplay", OUTPUT_FILE])
            print("Done playing")
        finally:
            busy.release()

    threading.Thread(target=generate_and_play, daemon=True).start()


def on_tcp_event(raw_event):
    """Called by TcpListener for each parsed event from Codex."""
    text = event_filter.filter(raw_event)
    if text:
        handle_event(text)


if __name__ == "__main__":
    # Start TCP listener for Codex events
    listener = TcpListener(on_event=on_tcp_event)
    listener.start()
    print("TCP listener started (connecting to Codex...)")

    # HTTP server still available for manual testing
    port = 8080
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"HTTP server running on http://localhost:{port}")
    print('POST / with {{"text": "your event summary here"}}')
    server.serve_forever()
