from HFTrainer import Llama2ForTranslation
from utils import utils

cfg = utils.load_config()

if __name__ == "__main__":
    llama_translator = Llama2ForTranslation()
    llama_translator.train_and_eval()
    llama_translator.test(f"Translate to {cfg['lang']}: Hello world!")