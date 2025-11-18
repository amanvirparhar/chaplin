# Chaplin

![Chaplin Thumbnail](./thumbnail.png)

A visual speech recognition (VSR) tool that reads your lips in real-time and types whatever you silently mouth. Runs fully locally.

Relies on a [model](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages?tab=readme-ov-file#autoavsr-models) trained on the [Lip Reading Sentences 3](https://mmai.io/datasets/lip_reading/) dataset as part of the [Auto-AVSR](https://github.com/mpc001/auto_avsr) project.

Watch a demo of Chaplin [here](https://youtu.be/qlHi0As2alQ).

## Setup

1. Clone the repository, and `cd` into it:
   ```sh
   git clone https://github.com/amanvirparhar/chaplin
   cd chaplin
   ```
2. Run the setup script...
   ```sh
   ./setup.sh
   ```
   ...which will automatically download the required model files from Hugging Face Hub and place them in the appropriate directories:
   ```
   chaplin/
   ├── benchmarks/
       ├── LRS3/
           ├── language_models/
               ├── lm_en_subword/
           ├── models/
               ├── LRS3_V_WER19.1/
   ├── ...
   ```
3. Install and run `ollama`, then pull the [`qwen3:4b`](https://ollama.com/library/qwen3:4b) model—this is the default LLM used through [Simon Willison's `llm` library](https://llm.datasette.io/).  
   To use a different backend such as Gemini, OpenAI, or Mistral:
   ```sh
   uv sync --extra gemini                   # installs optional deps from pyproject for Gemini support
   export LLM_GEMINI_API_KEY=sk-...         # set provider key (or add it to .env)
   CHAPLIN_LLM_MODEL=gemini-2.5-flash \     # or use Hydra override: llm.model="gemini-2.5-flash"
     uv run main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
   ```
   Or - edit the Hydra config to set your preferred model permanently. (its in hydra_configs/default.yaml)
   For other providers, install their plugin manually via `uv pip install llm-<provider>` and set the matching API key (see `.env.example` for guidance).
4. Install [`uv`](https://github.com/astral-sh/uv).
5. Install the Python dependencies declared in `pyproject.toml`:
   ```sh
   uv sync                     # add --extra gemini if you need the Gemini plugin
   ```

## Usage

1. Run the following command:
   ```sh
   uv run main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
   ```
2. Once the camera feed is displayed, you can start "recording" by pressing the `option` key (Mac) or the `alt` key (Windows/Linux), and start mouthing words.
3. To stop recording, press the `option` key (Mac) or the `alt` key (Windows/Linux) again. The raw VSR output will get logged in your terminal, and the LLM-corrected version will be typed at your cursor.
4. To exit gracefully, focus on the window displaying the camera feed and press `q`.
   - If you only see a black window, ensure your terminal has camera permissions (System Settings → Privacy & Security → Camera on macOS) and that no other app is using the webcam. You can pick a different device with `camera_index=<n>` (for example `uv run main.py ... camera_index=1`).

## LLM configuration

Chaplin relies on the `llm` Python library for its text-correction step. By default it targets the local Ollama model `ollama:qwen3:4b`, but you can point it at any other `llm` model:

- Override from the command line using Hydra:  
  ```sh
  uv run main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe \
    llm.model="gemini-2.5-flash"
  ```
- Or edit your Hydra defaults (recommended when you want a consistent setup):
  ```yaml
  # hydra_configs/default.yaml
  camera_index: 1        # pick an alternate webcam
  llm:
    model: "gemini-2.5-flash"
    options:
      temperature: 0.4   # any provider-specific options supported by llm
    system_prompt: null  # keep default prompt; override with custom text if needed
  output:
    mode: typing          # swap to azure_tts for spoken playback
  ```
- Or set environment variables:  
  `CHAPLIN_LLM_MODEL` (model id), `CHAPLIN_LLM_OPTIONS` (JSON dict of provider options), and `CHAPLIN_LLM_SYSTEM_PROMPT` (custom system instructions).
- Plugins and keys: install the relevant `llm-*` plugin (for example `pip install llm-gemini`) and ensure the provider-specific environment variables—such as `LLM_GEMINI_API_KEY`—are set before launching Chaplin.

## Output modes

By default Chaplin uses `pynput` to type the corrected text into the active window. You can switch the output behaviour without touching the code:

- Keep typing (default): ensure `output.mode=typing` in your Hydra config (or leave it unset).
- Azure TTS: install the optional dependency and set the Azure credentials before launching Chaplin (or edit the hydra config):
  ```sh
  uv sync --extra azure
  export CHAPLIN_OUTPUT_MODE=azure_tts
  export CHAPLIN_AZURE_SPEECH_KEY=<your-azure-key>
  export CHAPLIN_AZURE_SPEECH_REGION=uksouth     # e.g. uksouth
  export CHAPLIN_AZURE_SPEECH_VOICE=en-GB-SoniaNeural  # optional, defaults to this value
  ```
  You can also provide the same values via the Hydra config (`output.mode=azure_tts`, `output.azure.region=uksouth`, etc.).

When Azure TTS is active the corrected text is spoken through the system's default audio device instead of being typed.
