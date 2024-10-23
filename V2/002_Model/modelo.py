from transformers import DistilBertTokenizer, TFDistilBertModel
import os
#os.environ['TF_ENABLE_MLIR_OPTIMIZATIONS'] = '1'
os.environ["TRASFORMERS_CACHE"] = "./cache_TF/"

os.system(f"mkdir -p ./cache_TF/")
os.system(f"mkdir -p ./cache_TF/")
print("Done")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", force_download='True')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)

model.compile(optimizer='adam')

print('FINALIZADO OK')