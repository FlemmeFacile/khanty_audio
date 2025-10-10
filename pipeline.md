### High-Level Pipeline

Technical Architecture

Base Model: VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
Feature Extraction: wav2vec 2.0 for pseudo-phoneme extraction
Training Strategy: Two-stage transfer learning with component freezing
Fine-tuning: Selective training of text encoder and normalizing flow
📋 Project Description

Problem Statement

High-quality Text-to-Speech systems typically require large annotated datasets (20+ hours), but for the Khanty language, only 80 minutes of labeled data is available.

Proposed Solution

Pre-training Phase: Learning on unlabeled Finnish and Hungarian speech data using wav2vec 2.0 for pseudo-phoneme extraction and VITS for speech synthesis training
Fine-tuning Phase: Adaptation to Khanty language using available labeled data with strategic component freezing

```mermaid
flowchart LR
    A[Finnish/Hungarian<br>Unlabeled Speech] --> B[Pre-training<br>wav2vec 2.0 + VITS]
    B --> C[Pre-trained<br>TTS Model]
    C --> D[Fine-tuning on<br>Khanty Data]
    D --> E[Final Khanty<br>TTS System]

