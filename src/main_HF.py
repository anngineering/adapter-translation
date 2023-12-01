from HFTrainer import Llama2ForTranslation
from utils import utils

cfg = utils.load_config()

if __name__ == "__main__":
    llama_translator = Llama2ForTranslation()
    llama_translator.train_and_eval()
    llama_translator.test(f"Translate to {cfg['lang']}: Hello world!")
    llama_translator.test(f"Translate to {cfg['lang']}: Happy Friday everyone!")
    llama_translator.test(f"Translate to {cfg['lang']}: The CEO traveled to Paris and Amsterdam on an official visit.")
    llama_translator.test(f"Translate to {cfg['lang']}: On today's breaking news, there has been a traffic accident along Highway 67.")
    llama_translator.test(f"Translate to {cfg['lang']}: The firemen rescued the orange cat that was stuck in a tree.")

### Expected Output:
#  Olá mundo!
#  Feliz quarta-feira a todos!
#  O CEO viajou para Paris e Amsterdã em visita oficial.
#  Em notícias da atualidade, houve um acidente de trânsito ao longo da Rodovia Highway 67.
#  Os bombeiros resgataram o gato-amarelo que estava preso em um árvore.
