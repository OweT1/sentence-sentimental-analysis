from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TextClassificationPipeline

def get_model(model_name, num_epochs, num_labels, id2label, label2id, lora=False):
    lora_str = "lora" if lora else ""

    model = AutoModelForSequenceClassification.from_pretrained(
        f'models/{model_name}_finetuned{lora_str}_{num_epochs}epochs_best',
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def get_data_collator(tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator

def get_pipeline(model, tokenizer):
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline