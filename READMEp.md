### High-Level Pipeline
```mermaid
flowchart LR
    A[Finnish/Hungarian<br>Unlabeled Speech] --> B[Pre-training<br>wav2vec 2.0 + VITS]
    B --> C[Pre-trained<br>TTS Model]
    C --> D[Fine-tuning on<br>Khanty Data]
    D --> E[Final Khanty<br>TTS System]
