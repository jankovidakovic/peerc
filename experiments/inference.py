import torch
import wandb
import yaml
from transformers import AutoTokenizer

from constants import label2emotion
from experiments import get_parser
from experiments.model import get_model
from preprocessing import concat_turns

if __name__ == '__main__':
    # load the model, see if it performs the same as the original model
    parser = get_parser()
    args = parser.parse_args()
    # if not os.path.exists(args.config):
    #    raise FileNotFoundError(f"Config file {args.config} not found")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # download the wandb artifact
    run = wandb.init()
    artifact = run.use_artifact('jankovidakovic/emotion-classification-using-transformers/model-ucgwkaby:v0',
                                type='model')
    artifact_dir = artifact.download()

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    # _, val_dataset, _ = get_datasets(config, tokenizer)
    # device = torch.device(args.device)

    model = get_model(artifact_dir, config["model"], train=False, load_state_dict=True)

    turns = {
        "turn1": "I am happy",
        "turn2": "Why are you happy",
        "turn3": "Because model is loaded for inference"
    }
    model_input = concat_turns(turns, "roberta")
    input_ids = tokenizer.encode(model_input, return_tensors="pt", add_special_tokens=False)

    output = model(input_ids=input_ids)
    logits = output.logits

    predicted_class = torch.argmax(logits).item()
    predicted_label = label2emotion[predicted_class]

    print(f"Predicted emotion: {predicted_label}")
