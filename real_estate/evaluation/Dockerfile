FROM python:3.11-slim

# Install dependencies
# Pinned versions ensure consistent evaluation across all validators
RUN pip install --no-cache-dir \
    numpy \
    onnx==1.20.0 \
    onnxruntime==1.20.1

# Set working directory
WORKDIR /workspace

# Default command (will be overridden)
CMD ["python", "--version"]
