import time

import numpy as np
import onnxruntime as ort


class Policy:
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize the policy model for inference.

        Parameters
        ----------
        model_path : str
            Path to the ONNX model file
        use_gpu : bool, optional
            Whether to use GPU acceleration (default: True)
        """
        # Configure execution providers with GPU priority if available
        if use_gpu:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",  # Fallback to CPU if GPU fails
            ]
        else:
            providers = ["CPUExecutionProvider"]

        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.use_gpu = use_gpu

        # Check if GPU is actually being used
        current_provider = self.session.get_providers()[0]
        print(f"Using execution provider: {current_provider}")

    def action(self, x: np.ndarray):
        """
        Run inference on the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data for the model

        Returns
        -------
        np.ndarray
            Model output (always on CPU to avoid GPU-CPU transfer issues)
        """
        # Ensure correct input shape and dtype
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        x = x.astype(np.float32)

        # Run inference
        y = self.session.run([self.output_name], {self.input_name: x})[0]

        # Output is always returned on CPU by ONNX Runtime
        # No need for explicit GPU->CPU transfer
        return y


def benchmark_fps(policy, warmup=1000, test_iterations=10000):
    """
    Perform FPS (Frames Per Second) benchmark test for inference performance.

    This function measures the inference speed of the policy by running
    multiple iterations and calculating average FPS, throughput, and latency.

    Args:
        policy: Initialized Policy object for inference
        warmup: Number of warmup iterations to avoid cold start effects
        test_iterations: Number of test iterations for benchmarking

    Returns:
        float: Frames per second (FPS) achieved during the test
    """
    # Generate random input data with shape (42,) to match model input requirements
    obs = np.random.randn(42).astype(np.float32)

    # Print test configuration
    print("Starting FPS benchmark...")
    print(f"Device: {'GPU' if policy.use_gpu else 'CPU'}")
    print(f"Warmup iterations: {warmup}")
    print(f"Test iterations: {test_iterations}")
    print("-" * 40)

    # 1. Warmup phase: Run initial inferences to avoid cold start latency
    print("Warmup phase...")
    for _ in range(warmup):
        _ = policy.action(obs)

    # 2. Test phase: Measure inference performance over multiple iterations
    print("Test phase...")
    start_time = time.time()

    for _ in range(test_iterations):
        _ = policy.action(obs)

    # If using GPU, synchronize to ensure all operations are completed
    if policy.use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            # Fallback if torch is not available
            pass

    end_time = time.time()

    # 3. Calculate performance metrics
    total_time = end_time - start_time
    fps = test_iterations / total_time
    throughput = fps
    avg_latency_ms = (total_time / test_iterations) * 1000

    # 4. Print formatted results
    print("-" * 40)
    print("Benchmark Results:")
    print(f"Total inferences: {test_iterations}")
    print(f"Total time: {total_time:.3f}s")
    print(f"FPS (Frames Per Second): {fps:.2f} Hz")
    print(f"Throughput: {throughput:.2f} Hz")
    print(f"Average latency: {avg_latency_ms:.3f}ms")
    print("-" * 40)

    return fps


if "__main__" == __name__:
    policy = Policy(model_path="weight/policy2.onnx", use_gpu=False)
    benchmark_fps(policy)
