# Transfer Learning for Khanty language TTS Using Pseudo-Phoneme Representations

## 1. Working Title
Cross-Lingual Transfer Learning for Low-Resource Khanty language TTS Using Pseudo-Phoneme Representations

This project aims to implement the approach from the paper [Transfer Learning Framework for Low-Resource Text-to-Speech using a Large-Scale Unlabeled Speech Corpus]. We plan to train a model on Finnish and/or Hungarian audio, followed by fine-tuning on our Khanty data. We have 80 minutes of Khanty audio, which according to the paper, should be sufficient. The use of pseudo-phonemes is a key part of the method.

## 2. Problem Statement
Khanty language has very few speakers and limited audio resources. To develop a chatbot for learning Khanty, it is important to have synthesised speech not only from stories but also from dictionary entries and other available materials.

## 3. Main Contribution
This project presents possibly the first application of a ready transfer learning model for the Khanty language, specifically targeting the Kazym dialect. The pretrained model on resource-rich Finno-Ugric languages (Finnish and/or Hungarian) is adapted and fine-tuned using only 80 minutes of Khanty audio. This approach enables effective low-resource TTS synthesis without requiring large Khanty datasets, and has potential applicability to other Khanty dialects exhibiting lexical variation.

## 4. Proposed Method
- Stage 1: Train a VITS model on Finnish and/or Hungarian audio with pseudo-phonemes extracted using wav2vec 2.0.
- Stage 2: Fine-tune the model on Khanty audio (our 80 minutes), freezing the decoder and adapting the normalising flow for phonetic space alignment.
- We plan to experiment with different base languages (Finnish only, Hungarian only, combined) to find the best setting.

## 5. Experimental Setup

### Datasets
- Khanty audio collected from colleagues: ~80 minutes of fairy tales + separate alphabet recordings.
- Additional training data from:
  - [FBK-MT mosel dataset on HuggingFace](https://huggingface.co/datasets/FBK-MT/mosel)
  - [VoxPopuli dataset](https://github.com/facebookresearch/voxpopuli.git)
- Backup dataset with Finnish and Hungarian from [CSS10-LJSpeech](https://huggingface.co/datasets/ayousanz/css10-ljspeech).

### Baselines
Comparison with:
- GlowTTS
- FastSpeech2
- VITS-baseline

### Evaluation Metrics
- CER (Character Error Rate)
- SECS (Single Embedded Character Score)
- Subjective metrics (evaluated by project members and speakers of Ob-Ugric languages):
  - MOS (Mean Opinion Score)
  - SMOS (Similarity MOS)

## 6. Potential Results
Applying transfer learning with pseudo-phoneme representations can significantly improve Khanty TTS quality over baselines, even with limited Khanty data. If approved by Khanty dialect speakers, the model could be used in our chatbot to extend spoken materials to the Kazym dialect and beyond.

## 7. Related Work
- [Primary work](https://doi.org/10.21437/Interspeech.2022-225) on low-resource TTS with pseudo-phonemes.
- wav2vec 2.0 paper for self-supervised speech representation learning: [Baevski et al., 2020](https://doi.org/10.48550/arXiv.2006.11477).
- VITS original model paper: [Kim et al., 2021](https://arxiv.org/pdf/2106.06103). 
- GlowTTS [Kim et al., 2021](https://arxiv.org/pdf/2005.11129) and FastSpeech2 [Ren et al., 2021](https://arxiv.org/abs/2006.04558) references for baselines.
- Research on agglutinative low-resource TTS with morphology-aware pretraining: [IEEE paper](https://ieeexplore.ieee.org/abstract/document/10379131/references#references).

## 8. Timeline

| Stage                  | Duration    | Tasks                                                  |
|------------------------|-------------|--------------------------------------------------------|
| Data collection        | 2-3 weeks    | Collect Finnish/Hungarian and Khanty audio datasets   |
| Preparation            | 3 weeks     | Prepare phoneme normalisation, deal with diacritics    |
| Experiments            | 4 weeks     | Pretrain, fine-tune, compare with baselines            |
| Analysis & Evaluation  | 4 weeks     | Subjective and objective evaluation                     |
| Finalisation           | 3-4 weeks   | Refinements, documentation                              |

---

Feel free to modify or expand this README as needed.
If you want, I can also help prepare the initial commit or GitHub setup for this file.
