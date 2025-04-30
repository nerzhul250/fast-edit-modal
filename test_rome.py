import os
import time
from pathlib import Path

import modal
from fastapi.responses import StreamingResponse

from .common import app, inference_image, Colors, MINUTES, VOLUME_CONFIG

INFERENCE_GPU_CONFIG = "L40S:1"
N_INFERENCE_GPUS = 1

with inference_image.imports():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid


def get_model_path(run_name: str, run_dir: str = "/runs") -> Path:
    """Returns the path to the saved model for a specific run."""
    return Path(run_dir) / run_name

@app.cls(
    gpu=INFERENCE_GPU_CONFIG,
    image=inference_image,
    volumes=VOLUME_CONFIG,
    allow_concurrent_inputs=10,
    container_idle_timeout=15 * MINUTES,
)
class Inference:
    def __init__(self, run_name: str = "", run_dir: str = "/runs") -> None:
        self.run_name = run_name
        self.run_dir = run_dir
        self.engine = None

    @modal.enter()
    def init(self):
        if self.run_name:
            model_path = get_model_path(self.run_name, self.run_dir)
        else:
            # Pick the last run automatically
            run_paths = list(Path(self.run_dir).iterdir())
            if not run_paths:
                raise ValueError("No runs found in the runs directory")
            model_path = get_model_path(sorted(run_paths, key=os.path.getmtime)[-1].name, self.run_dir)

        print(
            Colors.GREEN,
            Colors.BOLD,
            f"🧠: Initializing vLLM engine for model at {model_path}",
            Colors.END,
            sep="",
        )

        VOLUME_CONFIG[self.run_dir].reload()
        
        engine_args = AsyncEngineArgs(
            model=str(model_path),
            gpu_memory_utilization=0.95,
            tensor_parallel_size=N_INFERENCE_GPUS,
            disable_custom_all_reduce=True,  # brittle as of v0.5.0
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def _stream(self, input: str, top_k: int = 50, max_tokens: int = 1024):
        if not input:
            return

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.1,
            top_p=0.95,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(input, sampling_params, request_id)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if (
                request_output.outputs[0].text
                and "\ufffd" == request_output.outputs[0].text[-1]
            ):
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(
            Colors.GREEN,
            Colors.BOLD,
            f"🧠: Effective throughput of {throughput:.2f} tok/s",
            Colors.END,
            sep="",
        )

    @modal.method()
    async def completion(self, input: str, top_k: int = 50, max_tokens: int = 1024):
        async for text in self._stream(input, top_k, max_tokens):
            yield text

    @modal.method()
    async def non_streaming(self, input: str, top_k: int = 50, max_tokens: int = 1024):
        output = [text async for text in self._stream(input, top_k, max_tokens)]
        return "".join(output)

    @modal.web_endpoint()
    async def web(self, input: str, top_k: int = 50, max_tokens: int = 1024):
        return StreamingResponse(self._stream(input, top_k, max_tokens), media_type="text/event-stream")
    
    @modal.exit()
    def stop_engine(self):
        if N_INFERENCE_GPUS > 1:
            import ray
            ray.shutdown()
        
        # Access private attribute to ensure graceful termination
        self.engine._background_loop_unshielded.cancel()


@app.local_entrypoint()
def inference_main(run_name: str = "/runs/rome", prompt: str = "Who is the president of the UK?"):
    """
    Interactive CLI for testing the model using vLLM.
    """
    inference = Inference(run_name)
    
    if prompt:
        print(Colors.GREEN, Colors.BOLD, f"🧠: Querying model {run_name}", Colors.END, sep="")
        response = ""
        for chunk in inference.completion.remote_gen(prompt):
            response += chunk
        print(Colors.BLUE, f"👤: {prompt}", Colors.END, sep="")
        print(Colors.GRAY, f"🤖: {response}", Colors.END, sep="")
        return
    
    print(Colors.GREEN, Colors.BOLD, f"🧠: Interactive session with model {run_name}", Colors.END, sep="")
    print("Enter `exit` to exit the interface.")
    
    while True:
        prompt = input("Input: ").strip()
        if prompt == "exit":
            break
            
        print("Output: ", end="", flush=True)
        for chunk in inference.completion.remote_gen(prompt):
            print(chunk, end="", flush=True)
        print()
