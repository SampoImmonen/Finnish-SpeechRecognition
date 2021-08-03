# Finnish speech recognition with wav2vec2

## project to build a speech recognition system

## Training
- TraininingNotebook.ipynb is based on https://huggingface.co/blog/fine-tune-xlsr-wav2vec2 and uses Huggingface for training
- CustomTrainingNotebook is raw pytorch and almost three times faster
- Training with a batch_size of 4 takes 17GB of VRAM

## Acoustic Model
- uses facebooks wav2vec2.0 pretrained with voxpopuli data https://github.com/facebookresearch/voxpopuli)

## Decoder

- CTCDecoder is from https://github.com/parlance/ctcdecode
- sample_decode.py has simple ctcbeamsearch with raw numpy 

## Usage:

Two main classes in SpeechRecognizer.py
- SpeechRecognizer and CTCDecoder

kuva SpeecRecognizer.py
mitä pitää olla kansiossa

### Sample:


## Data:
Data used for finetuning
1. Finnish common voice
2. finnish single speaker corpus 
3. finnish voxpopuli data
4. privately collected dataset


# Currently best model
Word error rate on common voice test
1. with language model: 8.89
2. without language model: 15.21

Word error rate on more difficult and larger testset created from parliament speeches
1. with language model: 17.77
2. without language model: 22.29

## KenLM

Instructions how to train your own kenLM language models: https://github.com/kmario23/KenLM-training

