# BAAI bge_m3 Multilingual Model Server

This server setup uses FastAPI to handle asynchronous requests for text embeddings and reranking tasks with the BAAI bge_m3 multilingual model. Designed for demonstration and testing, it showcases efficient request handling, including batching and GPU resource management, in a local environment setting. While not recommended for production, it's robust for testing purposes.

## Overview

The script creates an asynchronous web server capable of concurrently processing requests for generating text embeddings and reranking using the BAAI bge_m3 multilingual model. It's structured for demonstration or testing rather than production, integrating advanced AI models with web technologies.

## Key Components

### AI Model (`AIModel` class)

- **Purpose**: Encapsulates the BAAI bge_m3 multilingual model for embeddings generation (`embed`) and reranking (`rerank`).
- **Implementation**: Utilizes `BGEM3FlagModel` to process text, generating embeddings and reranking scores efficiently.

### Request and Response Models

- **Pydantic Models**: Define schemas for incoming requests (`EmbedRequest`, `RerankRequest`) and outgoing responses (`EmbedResponse`, `RerankResponse`), ensuring data integrity.

### Request Processor (`RequestProcessor` class)

- **Batch Processing**: Manages requests asynchronously, optimizing model inference and resource use by batching.
- **Concurrency Control**: Uses an asyncio semaphore to limit GPU access, preventing contention and ensuring stability.
- **Efficiency**: Implements an accumulation timeout to batch requests, minimizing model calls and maximizing GPU efficiency.

### FastAPI Server

- **Asynchronous API Routes**: Offers endpoints (`/embeddings/`, `/rerank/`) for asynchronous handling of embedding and reranking tasks, serving multiple clients simultaneously.
- **Timeout Management Middleware**: Enforces a maximum processing time, responding with a 504 error if exceeded, maintaining server responsiveness.

## Usage and Limitations

Intended for local testing and demonstration, this server illustrates machine learning model integration with web technologies for NLP tasks. While providing a solid foundation, it's not suited for production without further development, especially regarding security, error handling, and scalability.

While the server is designed with efficiency and concurrency in mind, it lacks specific optimizations for GPU usage, such as continuous batching. This implementation focuses on demonstrating basic batching and asynchronous processing capabilities. Users looking to scale this solution for higher performance in production environments should consider implementing more advanced GPU optimization techniques or the use production ready solutions.

### Dependencies and execution

I strongly recommend to run the code inside a nvidia/cuda:12.1.0-devel-ubuntu22.04 docker container, that image has all gpu/cuda libs required for best performance inference and running the code, just install the requirements.txt inside a non-volatile volume inside the container.

Python >= 3.10 is required!

## Conclusion

The server demonstrates a practical approach to NLP tasks using FastAPI and the BAAI bge_m3 multilingual model, focusing on asynchronous processing, batching, and concurrency. It's a robust solution for local environments, with scalability and efficiency in mind. Users are encouraged to adapt and extend it for specific needs and production readiness.
