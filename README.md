# Story Cloze Task Instructions

**Required Python >= 3.6.0**

Create virtual environment

```
python -m venv .venv
```

Activate environment (Windows)

```
.venv\Scripts\activate.bat
```

Activate environment (Unix or MacOS)

```
source .venv/bin/activate
```

Installing dependencies

```
pip install -r requirements.txt
```

## File Structure

Following should the file structure to run the model smoothly
```Shell
checkpoints/
    bert.pth
    sentiment_base.pth
    sentiment_finetuned.pth
    common_sense.pth
    combined_model.pth
data/
    nlp2_test.csv
    nlp2_train.csv
    nlp2_val.csv
src/
    all_models.py
    .....
```

## Training Pipeline
- Train BERT model
- Train Base Sentiment model using training data
- Fine-tune sentiment model on labeled validation data
- Train Commonsense model
- Train Combined model
- Make final prediction

## Files to be downloaded
- Download [Checkpoints TODO: add link]() and put them in checkpoints folder in root directory of this repository

## Reproducing our results
To reproduce our results mentioned in report download the checkpoints from above mentioned link and put them in checkpoints folder. The run the following command
```
python src/predict.py --model <model_name>
```
- model: This parameter can have following possible values [`bert, sentiment, commonsense, combined`]. Use model value as `combined` to get final model test accuracy

It will predict the final test accuracy in console

## Running training pipeline

### Train BERT Model
To fine-tune BERT model for story cloze task, run the following command
```
python src/bert_model.py [--batch_size] [--num_epochs] [--lr] [--print_every]
```
- batch_size: default values is 32
- num_epochs: default values is 5
- lr: default value is 0.00001. Learning rate for training
- print_every: default value is 10. After how many step should training accuracy and runnning loss should be printed

After running the above command, fine-tuned model checkpoint will be stored at `checkpoints/bert.pth`

### Train Base Sentiment model using training data
To train sentiment model on training data, run the following command
```
python src/sentiment_journey.py [--batch_size] [--num_epochs] [--lr] [--print_every]
```
- batch_size: default values is 64
- num_epochs: default values is 3
- lr: default value is 0.001
- print_every: default value is 100

After running the above command, trained model checkpoint will be stored at `checkpoints/sentiment_base.pth`

### Fine-tune sentiment model on labeled validation data
To fine-tune sentiment model on validation data, run the following command
```
python src/sentiment_end2end.py [--batch_size] [--num_epochs] [--lr] [--print_every]
```
- batch_size: default values is 32
- num_epochs: default values is 3
- lr: default value is 0.00001
- print_every: default value is 10

After running the above command, fine-tuned model checkpoint will be stored at `checkpoints/sentiment_finetuned.pth`

### Train Commonsense model
To train the commonsense model for story cloze task, run the following command
```
python src/commonsense_net.py [--batch_size] [--num_epochs] [--lr] [--print_every]
```
- batch_size: default values is 100
- num_epochs: default values is 10
- lr: default value is 0.0001
- print_every: default value is 5

After running the above command, trained model checkpoint will be stored at `checkpoints/common_sense.pth`

### Train Combined model
To train the combined model for story cloze task, run the following command
```
python src/combined_model.py [--batch_size] [--num_epochs] [--lr] [--print_every]
```
- batch_size: default values is 32
- num_epochs: default values is 5
- lr: default value is 0.00001
- print_every: default value is 10

After running the above command, trained model checkpoint will be stored at `checkpoints/combined_model.pth`


