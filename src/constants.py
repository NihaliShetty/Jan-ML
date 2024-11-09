AOKVQA_DATASET_NAME = 'HuggingFaceM4/A-OKVQA'

RESULTS_DIR = '../results'
RESULTS_FMT_DIR = f'{RESULTS_DIR}/formatted'
RESULTS_RAW_DIR = f'{RESULTS_DIR}/raw'

TASKS = [
  'mc',
  'da'
]

# MODELS = [
#   # 'bert-classifier-unimodal-ques',
#   # 'clip-contrastive-multimodal',
#   # 'clip-contrastive-unimodal-image',
#   # 'clip-classifier-unimodal-ques',
#   # 'clip-constrastive-zero-multimodal',
#   # 'clip-contrastive-unimodal-image',
#   # 'clip-contrastive-unimodal-ques',
#   # 'clip-contrastive-zero-unimodal-image',
#   # 'clipcap',
#   # 'clipcap-rnx04',
#   'resnet-classifier-unimodal-image'
# ]


MODELS = {
  # Multimodal - Question + Image Input
  # "MolmoE 1B": {
  #   "raw_results_dir": "",
  #   "raw_results_fname": ""
  # },
  "BLIP 2 OPT 2.7B": {
    "raw_results_dir": "mingqian",
    "raw_results_fname": "blip2-opt-2.7b_val_results",
  },
  "ViperGPT": {
    "raw_results_dir": "joel",
    "raw_results_fname": "vipergpt.json"
  },
  # "Q&A Prompt-InstructBLIP": {
  #   "raw_results_dir": "",
  #   "raw_results_fname": ""
  # },
  "LLaVA-1.5-7b": {
    "raw_results_dir": "mingqian",
    "raw_results_fname": "llava1.5-7b_val_results"
  },
  "CLIP (Zero Shot) MM": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clip-constrastive-zero-multimodal"
  },
  "CLIP (Contrastive) MM": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clip-contrastive-multimodal"
  },
  "CLIP (Classifier) MM": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clip-classifier-multimodal"
  },
  "ClipCap": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clipcap"
  },
  "ClipCap (with RN50x4)": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clipcap-rnx04"
  },
  # Unimodal - Question Input
  "Llama 3 8B Instruct": {
    "raw_results_dir": "joel",
    "raw_results_fname": "llama3-8b-instruct.json"
  },
  # "CLIP (Zero Shot)": {
  #   "raw_results_dir": "",
  #   "raw_results_fname": ""
  # },
  "CLIP (Contrastive) UM-T": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clip-contrastive-unimodal-ques"
  },
  "CLIP (Classifier) UM-T": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clip-classifier-unimodal-ques"
  },
  "BERT Classifier": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "bert-classifier-unimodal-ques"
  },
  # Unimodal - Image Input
  "CLIP (Zero Shot) UM-I": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clip-contrastive-zero-unimodal-image"
  },
  "CLIP (Contrastive) UM-I": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clip-contrastive-unimodal-image"
  },
  "CLIP (Classifier) UM-I": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "clip-classifier-unimodal-image"
  },
  "ResNet (Classifier)": {
    "raw_results_dir": "nihali",
    "raw_results_fname": "resnet-classifier-unimodal-image"
  },
}