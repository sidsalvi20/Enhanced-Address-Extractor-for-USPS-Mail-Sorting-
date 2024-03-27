import re
import os
import json
from PIL import Image
import torch
import random
import numpy as np
from transformers import PretrainedConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from transformers.image_transforms import to_pil_image
from datasets import load_dataset, load_from_disk  
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance


def run_prediction(test_sample, model, processor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixel_values = torch.tensor(test_sample["pixel_values"]).unsqueeze(0)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)



    #CODE TO DISPLAY THE IMAGE:
    # pixel_values = np.squeeze(test_sample["pixel_values"])
    # pixel_values = (pixel_values + 1) / 2


    # to_pil_image(pixel_values)


    target = processor.token2json(test_sample["target_sequence"])
    return prediction, target

def select_random_images(image_folder, num_images=2000):
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    return random.sample(images, num_images)

def read_json_and_convert(json_folder, image_path):
    base = os.path.basename(image_path)
    json_path = os.path.join(json_folder, os.path.splitext(base)[0] + '.json')
    with open(json_path, 'r') as file:
        data = json.load(file)
    return f"{data['from_address']}, {data['to_address']}"

def calculate_cer(s1, s2):
    return levenshtein_distance(s1, s2) / max(len(s1), len(s2))


def cer_eval(test_image_folder, test_json_folder, processor_eval, model_eval):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = select_random_images(test_image_folder)
    total_cer = 0
    for image_path in tqdm(images):
        ground_truth = read_json_and_convert(test_json_folder, image_path)
        image = Image.open(image_path).convert("RGB")

        inputs = processor_eval(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)

        task_prompt = "<s>"
        decoder_input_ids = processor_eval.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        outputs = model_eval.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model_eval.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor_eval.tokenizer.pad_token_id,
            eos_token_id=processor_eval.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor_eval.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        prediction = processor_eval.batch_decode(outputs.sequences)[0]
        prediction = processor_eval.token2json(prediction)
        print("GROUND TRUTH: ", ground_truth)

        pred_str = f"{prediction['from_address']}, {prediction['to_address']}"
        print("PRED: ",pred_str)
        cer = calculate_cer(ground_truth, pred_str)
        print("CER:", cer)
        print()
        total_cer += cer
    
    average_cer = total_cer / len(images)
    print(f"Average Character Error Rate: {average_cer}")


def main():

    test_image_folder = ""
    test_json_folder = ""

    print("Starting Evaluation...")
    processed_dataset = load_from_disk('./train/processed_dataset')
    print("Dataset Loaded!")
    print("Loading trained checkpoint & processor: ")
    model = VisionEncoderDecoderModel.from_pretrained("sidsalvi20/package-address-extract")
    new_special_tokens = ['<s_to_address>', '</s_to_address>', '<s_from_address>', '</s_from_address>']
    eos_token = '</s>'
    task_start_token = '<s>'
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})
    print("Model Loading Complete!")
    processor.feature_extractor.size = [720,960] 
    processor.feature_extractor.do_align_long_axis = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    
    print("Running prediction for a random sample from the test set:")
    test_sample = processed_dataset["test"][random.randint(1, 50)]
    prediction, target = run_prediction(test_sample, model, processor)
    print(f"Reference:\n {target}")
    print(f"Prediction:\n {prediction}")

    print("Calculating CER for test set: ")
    cer_eval(test_image_folder, test_json_folder, processor, model)


if __name__ == "__main__":
    main()
