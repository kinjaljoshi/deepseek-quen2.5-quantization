We evaluates the [**DeepSeek-R1-Distill-Qwen-14B**](https://huggingface.co/Qwen/Qwen2.5-14B) model across different precision levels **(FP32, FP16, 8-bit, 4-bit, GPTQ, AWQ, and offloaded)** by testing its performance on simple and complex tasks. 

* Simple tasks include fact questions, math problems, and science explanations
* Complex tasks require multi-step reasoning such as classification, phishing detection, and fraud analysis. 

The model is benchmarked using default generation settings, a **sampling strategy (temperature=0.6, top-p=0.95, 20 responses)**, and **beam search (beam sizes 2-5)**. 

Inference time is measured for each configuration, and results are stored in a structured DataFrame for comparison. The final analysis provides insights into the trade-offs between speed, accuracy, and computational efficiency across various quantization methods and decoding strategies.

**FP32 (Full Precision)**
Uses 32-bit floating-point numbers, providing the highest accuracy but requiring high VRAM.

**FP16 (Half Precision)**
Uses 16-bit floating-point numbers, reducing memory consumption by 50% compared to FP32 while maintaining reasonable accuracy.

**8-bit Quantization**
Converts model weights to 8-bit integers, significantly reducing memory usage by 75% while maintaining decent accuracy.

**4-bit Quantization**
Further compresses model weights to 4-bit integers, reducing VRAM usage by ~80% but increasing quantization-related precision loss.

**GPTQ (4-bit Quantization-Aware Training)**
A post-training quantization method that optimizes weights for 4-bit precision, improving inference speed and memory efficiency.Unlike standard 4-bit quantization, GPTQ retains more accuracy by carefully adjusting model weights to minimize information loss.

**AWQ (Adaptive Weight Quantization)**
An optimized 4-bit quantization technique that selectively preserves critical weights, achieving higher accuracy than standard 4-bit methods.AWQ is designed for low-precision inference while minimizing performance trade-offs, making it better suited for real-world applications.

**Offloaded (Hybrid CPU-GPU Execution)**
Splits model execution between GPU and CPU, offloading less critical layers to CPU to reduce VRAM consumption while maintaining efficiency.
Useful for limited-memory setups (e.g., NVIDIA T4 GPUs), will result in increased latency due to CPU-GPU data transfers.
