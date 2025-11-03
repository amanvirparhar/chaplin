import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import cv2
import llm
from llm import UnknownModelError
from pydantic import BaseModel, ValidationError
from pynput import keyboard


class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant that helps make corrections to the output of a lipreading "
    "model. The text you will receive was transcribed using a video-to-text system that "
    "attempts to lipread the subject speaking in the video, so the text will likely be "
    "imperfect. The input text will also be in all-caps, although your response should "
    "be capitalized correctly and should NOT be in all-caps.\n\n"
    "If something seems unusual, assume it was mistranscribed. Do your best to infer the "
    "words actually spoken, and make changes to the mistranscriptions in your response. "
    "Do not add more words or content, just change the ones that seem to be out of place "
    "(and, therefore, mistranscribed). Do not change even the wording of sentences, just "
    "individual words that look nonsensical in the context of all of the other words in "
    "the sentence.\n\nAlso, add correct punctuation to the entire text. ALWAYS end each "
    "sentence with the appropriate sentence ending: '.', '?', or '!'.\n\nReturn the "
    "corrected text in the format of 'list_of_changes' and 'corrected_text'."
)

ENV_MODEL_NAME = "CHAPLIN_LLM_MODEL"
ENV_SYSTEM_PROMPT = "CHAPLIN_LLM_SYSTEM_PROMPT"
ENV_OPTIONS = "CHAPLIN_LLM_OPTIONS"


class Chaplin:
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, camera_index: int = 0):
        self.vsr_model = None
        self.camera_index = camera_index

        # flag to toggle recording
        self.recording = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

        # setup keyboard controller for typing
        self.kbd_controller = keyboard.Controller()

        # setup asyncio event loop in background thread
        self.loop = asyncio.new_event_loop()
        self.async_thread = ThreadPoolExecutor(max_workers=1)
        self.async_thread.submit(self._run_event_loop)

        # sequence tracking to ensure outputs are typed in order
        self.next_sequence_to_type = 0
        self.current_sequence = 0  # counter for assigning sequence numbers
        self.typing_lock = None  # will be created in async loop
        self._init_async_resources()

        # setup global hotkey for toggling recording with option/alt key
        self.hotkey = keyboard.GlobalHotKeys({
            '<alt>': self.toggle_recording
        })
        self.hotkey.start()

        # LLM configuration
        normalized_llm_config = self._normalize_llm_config(llm_config)
        env_model = os.getenv(ENV_MODEL_NAME)
        env_system_prompt = os.getenv(ENV_SYSTEM_PROMPT)

        self.llm_model_name = env_model or normalized_llm_config.get(
            "model") or "ollama:qwen3:4b"
        self.llm_system_prompt = (
            env_system_prompt
            or normalized_llm_config.get("system_prompt")
            or DEFAULT_SYSTEM_PROMPT
        )
        self.llm_options = self._load_llm_options(normalized_llm_config)
        self._llm_model = None
        self._llm_model_is_async = False

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _init_async_resources(self):
        """Initialize async resources in the async loop"""
        future = asyncio.run_coroutine_threadsafe(
            self._create_async_lock(), self.loop)
        future.result()  # wait for it to complete

    async def _create_async_lock(self):
        """Create asyncio.Lock and Condition in the event loop's context"""
        self.typing_lock = asyncio.Lock()
        self.typing_condition = asyncio.Condition(self.typing_lock)

    def toggle_recording(self):
        # toggle recording when alt/option key is pressed
        self.recording = not self.recording

    async def correct_output_async(self, output, sequence_num):
        # perform inference on the raw output to get back a "correct" version
        response_payload = await self._invoke_llm(output)

        # get only the corrected text
        chat_output = self._parse_llm_response(response_payload)

        # if last character isn't a sentence ending (happens sometimes), add a period
        chat_output.corrected_text = chat_output.corrected_text.strip()
        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # add space at the end
        chat_output.corrected_text += ' '

        # wait until it's this task's turn to type
        async with self.typing_condition:
            while self.next_sequence_to_type != sequence_num:
                await self.typing_condition.wait()

            # this task's turn to type the corrected text
            self.kbd_controller.type(chat_output.corrected_text)

            # increment sequence and notify next task
            self.next_sequence_to_type += 1
            self.typing_condition.notify_all()

        return chat_output.corrected_text

    def perform_inference(self, video_path):
        # perform inference on the video with the vsr model
        output = self.vsr_model(video_path)

        # print the raw output to console
        print(f"\n\033[48;5;21m\033[97m\033[1m RAW OUTPUT \033[0m: {output}\n")

        # assign sequence number for this task
        sequence_num = self.current_sequence
        self.current_sequence += 1

        # start the async LLM correction (non-blocking) with sequence number
        asyncio.run_coroutine_threadsafe(
            self.correct_output_async(output, sequence_num),
            self.loop
        )

        # return immediately without waiting for correction
        return {
            "output": output,
            "video_path": video_path
        }

    def _normalize_llm_config(self, llm_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if llm_config is None:
            return {}

        dict_config_type = None
        omega_conf = None
        try:
            from omegaconf import DictConfig, OmegaConf  # type: ignore
            dict_config_type = DictConfig
            omega_conf = OmegaConf
        except ImportError:
            pass

        if dict_config_type and isinstance(llm_config, dict_config_type):
            assert omega_conf is not None
            return omega_conf.to_container(llm_config, resolve=True)  # type: ignore[arg-type]

        if isinstance(llm_config, dict):
            return dict(llm_config)

        raise TypeError(
            "llm_config must be a dict, OmegaConf DictConfig, or None. "
            f"Received type: {type(llm_config)!r}"
        )

    def _load_llm_options(self, llm_config: Dict[str, Any]) -> Dict[str, Any]:
        options = llm_config.get("options") or {}
        if not isinstance(options, dict):
            raise TypeError("llm_config['options'] must be a mapping if provided.")

        env_options = os.getenv(ENV_OPTIONS)
        if env_options:
            try:
                parsed_env = json.loads(env_options)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Unable to parse {ENV_OPTIONS} environment variable. "
                    f"Expected JSON object, received: {env_options!r}"
                ) from exc
            if not isinstance(parsed_env, dict):
                raise ValueError(
                    f"{ENV_OPTIONS} must decode to a JSON object. Received: {parsed_env!r}"
                )
            options = {**options, **parsed_env}

        return options

    async def _invoke_llm(self, output: str):
        self._ensure_llm_model()
        prompt = f"Transcription:\n\n{output}"

        if self._llm_model_is_async:
            response = await self._llm_model.prompt(  # type: ignore[call-arg]
                prompt=prompt,
                system=self.llm_system_prompt,
                schema=ChaplinOutput,
                stream=False,
                **self.llm_options,
            )
            return await response.json()

        # for sync models, run in a worker thread to avoid blocking the event loop
        def _call_model():
            response = self._llm_model.prompt(  # type: ignore[union-attr]
                prompt=prompt,
                system=self.llm_system_prompt,
                schema=ChaplinOutput,
                stream=False,
                **self.llm_options,
            )
            return response.json()

        return await asyncio.get_running_loop().run_in_executor(
            self.executor, _call_model)

    def _parse_llm_response(self, response_payload):
        try:
            if isinstance(response_payload, str):
                return ChaplinOutput.model_validate_json(response_payload)
            return ChaplinOutput.model_validate(response_payload)
        except ValidationError as exc:
            raise ValueError(
                "Received an unexpected response from the configured LLM. "
                "Ensure the model supports structured output/schema responses."
            ) from exc

    def _ensure_llm_model(self):
        if self._llm_model is not None:
            return

        try:
            self._llm_model = llm.get_async_model(self.llm_model_name)
            self._llm_model_is_async = True
            return
        except UnknownModelError:
            pass

        try:
            self._llm_model = llm.get_model(self.llm_model_name)
            self._llm_model_is_async = False
        except UnknownModelError as exc:
            raise RuntimeError(
                f"Unable to load LLM model '{self.llm_model_name}'. "
                "Ensure the appropriate llm plugin is installed "
                "(e.g. 'pip install llm-ollama' for Ollama models or "
                "'pip install llm-gemini' for Google Gemini)."
            ) from exc

    def start_webcam(self):
        # init webcam
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open webcam at index {self.camera_index}. "
                "Verify that the camera is connected, not in use by another application, "
                "and that the current process has camera permissions."
            )

        cv2.namedWindow('Chaplin', cv2.WINDOW_AUTOSIZE)

        # set webcam resolution, and get frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_frame_time = time.time()

        futures = []
        output_path = ""
        out = None
        frame_count = 0

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('Chaplin', cv2.WND_PROP_VISIBLE) < 1:
                    # remove any remaining videos that were saved to disk
                    for file in os.listdir():
                        if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                            os.remove(file)
                    break

                current_time = time.time()

                # conditional ensures that the video is recorded at the correct frame rate
                if current_time - last_frame_time >= self.frame_interval:
                    ret, frame = cap.read()
                    if ret:
                        # convert to grayscale for VSR pipeline while keeping original for display if needed
                        compressed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # update frame timing regardless of recording state
                        last_frame_time = current_time

                        if self.recording:
                            if out is None:
                                output_path = self.output_prefix + \
                                    str(time.time_ns() // 1_000_000) + '.mp4'
                                out = cv2.VideoWriter(
                                    output_path,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    self.fps,
                                    (frame_width, frame_height),
                                    False  # isColor
                                )

                            out.write(compressed_frame)

                            # circle to indicate recording, only appears in the window and is not present in video saved to disk
                            cv2.circle(compressed_frame, (frame_width -
                                                          20, 20), 10, (0, 0, 0), -1)

                            frame_count += 1
                        # check if not recording AND video is at least 2 seconds long
                        elif not self.recording and frame_count > 0:
                            if out is not None:
                                out.release()

                            # only run inference if the video is at least 2 seconds long
                            if frame_count >= self.fps * 2:
                                futures.append(self.executor.submit(
                                    self.perform_inference, output_path))
                            else:
                                os.remove(output_path)

                            output_path = self.output_prefix + \
                                str(time.time_ns() // 1_000_000) + '.mp4'
                            out = cv2.VideoWriter(
                                output_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                self.fps,
                                (frame_width, frame_height),
                                False  # isColor
                            )

                            frame_count = 0

                        # display the frame in the window
                        cv2.imshow('Chaplin', cv2.flip(compressed_frame, 1))
                    else:
                        # log once if the camera feed temporarily fails
                        print("Warning: Failed to read frame from webcam.")
                        continue

                # ensures that videos are handled in the order they were recorded
                for fut in list(futures):
                    if fut.done():
                        result = fut.result()
                        # once done processing, delete the video with the video path
                        os.remove(result["video_path"])
                        futures.remove(fut)
                    else:
                        break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, shutting down gracefully...")
        finally:
            # release everything
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

            # stop global hotkey listener
            self.hotkey.stop()

            # stop async event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.async_thread.shutdown(wait=True)

            # shutdown executor
            self.executor.shutdown(wait=True)
