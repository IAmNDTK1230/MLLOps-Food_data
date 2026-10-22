# Training scope

The group trained the classical TF-IDF/Logistic Regression and aspect classifiers on frozen VN-MSASar training rows; the rule baseline is manually specified rather than fitted. The three LLM families Qwen, Llama, and DeepSeek are pretrained open-weight models used for local inference and benchmarking. No full LLM pretraining from scratch was performed. Any future LoRA/QLoRA adaptation would be fine-tuning, not from-scratch training. Foody has no trusted gold labels and was not used to train these benchmark classifiers.
