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
