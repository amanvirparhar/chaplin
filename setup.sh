mkdir -p benchmarks/LRS3/language_models/lm_en_subword/
wget https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json -O benchmarks/LRS3/language_models/lm_en_subword/model.json
wget https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth -O benchmarks/LRS3/language_models/lm_en_subword/model.pth

mkdir -p benchmarks/LRS3/models/LRS3_V_WER19.1/
wget https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json -O benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
wget https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth -O benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth