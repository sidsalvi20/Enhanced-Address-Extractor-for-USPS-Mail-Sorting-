import json
import os
import shutil
import torch
from pathlib import Path
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_from_disk   

new_special_tokens = [] 
task_start_token = "<s>"
eos_token = "</s>"
os.environ["HF_HOME_TOKEN"] = "" #HF TOKEN HERE

def create_metadata_jsonl(base_path_str, json_path_str, image_path_str):
    
    metadata_list = []
    
    base_path = Path(base_path_str)
    metadata_path = base_path.joinpath(json_path_str)
    image_path = base_path.joinpath(image_path_str)
    
    print(image_path)
    print(metadata_path)

    for file_name in metadata_path.glob("*.json"):
        with open(file_name, "r") as json_file:
            data = json.load(json_file) 
            text = json.dumps(data)
            print(file_name)
            if image_path.joinpath(f"{file_name.stem}.png").is_file():
                metadata_list.append({"text":text,"file_name":f"{file_name.stem}.png"})
    
    with open(image_path.joinpath('metadata.jsonl'), 'w') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
            outfile.write('\n')
    shutil.rmtree(metadata_path)

    
def create_hf_dataset(base_path_str, image_path_str):
    base_path = Path(base_path_str)
    image_path = base_path.joinpath(image_path_str)
    dataset_hf = load_dataset("imagefolder", data_dir=image_path, split="train")
    
    print(f"Dataset has {len(dataset_hf)} images")
    print(f"Dataset features are: {dataset_hf.features.keys()}")

    return dataset_hf

def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
  
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
       
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"
        return obj

def preprocess_documents_for_donut(sample):
 
    task_start_token = "<s>"
    eos_token = "</s>"
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text) + eos_token
 
    image = sample["image"].convert('RGB')
    return {"image": image, "text": d_doc}

def transform_and_tokenize(sample, processor, split="train", max_length=512, ignore_id=-100):
    
    try:
        pixel_values = processor(
            sample["image"], random_padding=split == "train", return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id 
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}



def main():
    base_path_str = "/content/drive/MyDrive/PACKAGE_IMAGE_TO_TEXT/CODE_FILES/dataset_generate/output_images"
    json_path_str = "class_2_json"
    image_path_str = "class_2"

    print("Creating metadata.jsonl file...")
    create_metadata_jsonl(base_path_str, json_path_str, image_path_str)
    print("metadata.jsonl File Created Successfully!")

    

    dataset = create_hf_dataset(base_path_str, image_path_str)
    print("Dataset Created Successfully!")

    print("Processing dataset with new tokens...")
    proc_dataset = dataset.map(preprocess_documents_for_donut)
    print("Dataset Processing part1 completed!")
    
    print(f"Sample: {proc_dataset[45]['text']}")
    print(f"New special tokens: {new_special_tokens + [task_start_token] + [eos_token]}")

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})
    processor.feature_extractor.size = [720,960]
    processor.feature_extractor.do_align_long_axis = False
    print("Dataset mapping with processor...")
    processed_dataset = proc_dataset.map(transform_and_tokenize,remove_columns=["image","text"])
    processed_dataset = proc_dataset.map(lambda example: transform_and_tokenize(example, processor), remove_columns=["image", "text"])
    print("Dataset mapped successfully!")


    print("Split Data into Train and Test sets:")
    processed_dataset = processed_dataset.train_test_split(test_size=0.20)
    print(processed_dataset)
    processed_dataset.save_to_disk("processed_dataset")
    print("Dataset saved successfully!")

    print("Loading the Pretrained Model...")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"New embedding size: {new_emb}")
    model.config.encoder.image_size = processor.feature_extractor.size[::-1] 
    model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]
    print("Pretrained Model Loading Complete!")
    
    print("Starting Training: ")
    hf_repository_id = "package-address-extract-test"

    # Arguments for training
    training_args = Seq2SeqTrainingArguments(
        output_dir=hf_repository_id,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
        save_strategy="epoch",
        predict_with_generate=True,
        report_to="tensorboard",
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=hf_repository_id,
        hub_token=os.environ.get("HF_HOME_TOKEN"),
    )

    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
    )

    trainer.train()
    print("Training Completed Successfully!")

if __name__ == "__main__":
    main()
