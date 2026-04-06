import torch

BATCH_SIZE = 32
MAX_CHARS = 512
SAMPLE_SEED = 42
DEVICE = 0 if torch.cuda.is_available() else -1  # GPU
MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

input_csv_path = "./data/eng_content_v2.csv"
output_csv_path = "./output/sentiment_scores.csv"
output_plot_path = "./output/"
