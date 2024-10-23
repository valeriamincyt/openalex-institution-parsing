#Con este script bajo los archivos del modelo de la página de https://huggingface.co/ para poderlos usar en el language_model

from transformers import DistilBertTokenizer, TFDistilBertModel
import os
#os.environ['TF_ENABLE_MLIR_OPTIMIZATIONS'] = '1'
os.environ["TRASFORMERS_CACHE"] = "./cache_TF/"

os.system(f"mkdir -p ./distilbert-local'")
print("Done")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", force_download='True')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)

tokenizer.save_pretrained('./distilbert-local')
model.save_pretrained('./distilbert-local')

print('FINALIZADO OK')