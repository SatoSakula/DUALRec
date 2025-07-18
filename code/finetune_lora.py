# Generate training samples for LoRA fine-tuning using LSTM predictions
import json
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType


def build_finetune_dataset_from_lstm(train_data, movies_df, model, title_features, genre_features, movie2idx, save_path):
    samples = []
    all_users = train_data['userId'].unique()
    for user_id in tqdm(all_users, desc="Building LoRA training samples"):
        user_data = train_data[train_data['userId'] == user_id].sort_values('timestamp')
        if len(user_data) < 10:
            continue
        movie_ids = user_data['movieId'].values
        input_history = movie_ids[:-5]
        target_ids = movie_ids[-5:]

        watched_titles = [
            movies_df[movies_df['movieId'] == mid].iloc[0]['title']
            for mid in input_history[-5:]
        ]
        history_text = ", ".join(watched_titles)

        try:
            lstm_title = get_lstm_based_recommendation(
                model=model,
                user_movie_ids=input_history,
                title_data=title_features,
                genre_data=genre_features,
                movie2idx=movie2idx,
                idx2movie={v: k for k, v in movie2idx.items()},
                movies_df=movies_df
            )
        except:
            continue

        target_titles = [
            movies_df[movies_df['movieId'] == mid].iloc[0]['title']
            for mid in target_ids[:3]
        ]
        rec_output = "\n".join([f"{i+1}. {t}" for i, t in enumerate(target_titles)])

        samples.append({
            "instruction": "Given the user's watched movies and LSTM recommendation, generate 3 more movies the user will likely enjoy.",
            "input": f"Watched: {history_text}\nLSTM Suggests: {lstm_title}",
            "output": rec_output
        })

    with open(save_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Saved {len(samples)} training samples to {save_path}")

build_finetune_dataset_from_lstm(
    train_data=train_data,
    movies_df=movies,
    model=model,
    title_features=title_features,
    genre_features=genre_features,
    movie2idx=movie2idx,
    save_path=
)

"""
# 基于推荐任务和用户历史，微调DeepSeek  VL2模型"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch, json

def finetune_deepseek_lora(jsonl_path, base_model_name="", output_dir=""):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    with open(jsonl_path) as f:
        data = [json.loads(line.strip()) for line in f]
    dataset = Dataset.from_list(data)

    def tokenize(example):
        prompt = f"{example['instruction']}\n{example['input']}\n###\n{example['output']}"
        tokens = tokenizer(prompt, padding='max_length', truncation=True, max_length=512)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Finetuned model saved to {output_dir}")

finetune_deepseek_lora(
    jsonl_path="",
    base_model_name="",
    output_dir=""
)
