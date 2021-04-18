# ACTS - Assignment 1
### Rodrigo A. Chavez M.

In this repository you can find the code for my assignment 1 for ACTS.

### Installation
To create the conda environment use `conda env create -f environment.yml`
 -  Due to some conflicting packages it might be necessary to reinstall pytorch and make sure is 1.8 or higher.
 -  Spacy is used as tokenizer, so we need to download the english weights. Use `python -m spacy download en_core_web_sm`
 -  The pip package of torchmetrics has the 2.0 version but we use the 3.0 so install from source with `pip install git+https://github.com/PytorchLightning/metrics.git@master`   


SentEval is used in the `eval.py` part of the assignment
  -  Clone the repository to the root folder of this one with `clone https://github.com/facebookresearch/SentEval`
  -  Download the SentEval data with:
    -  `cd SentEval/data/downstream/`
    -  `./get_transfer_data.bash`

The tasks used from SentEval are: 
`['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC','MRPC', 'SICKEntailment',SICK-R]`



## Trained Models
You can find the trained models and tensorboard files in the following [link](https://amsuni-my.sharepoint.com/:f:/g/personal/rodrigo_alejandro_chavez_mulsa_student_uva_nl/EnePx_5QUWlHluiaFNyBSmUBOJUtn9blka-rVCY-bhRPKw?e=n0ELsc) but you will need be logged in with an UvA account to have access, put the folder trained_models in the root directory as seen in the diagram below. The checkpoints are in the folder of their name under the folder gold with the name of the model. (e.g. awe.cpkt)


## File structure:
Here is an overview of the file structure:   

. \
├── autoEnv \
├── bash_scripts \
├── data \
│   └── snli \
├── dev_notebooks \
├── lisa \
├── modules \
├── results_senteval \
├── SentEval \
├── subdata \
│   └── snli \
├── tb_logs \
├── trained_models \
│   ├── awe \
│   ├── bilstm \
│   ├── bilstm-max \
│   └── lstm \
├── .vector_cache \
└── wandb \


## Run code

For a demo visit the `demo.ipynb` notebook where you can test the models and see the analysis.  
To train the models you can run the command:  `python train.py --gpus -1 --model bilstm-max --precision 32 --batch 64 --disable_nonlinear`.  
You can choose between the 4 different models with the `--model` flag:   
 -  `awe` - Average Word Embeddings
 -  `lstm` - LSTM
 -  `bilstm`  - BILSTM
 -  `bilstm-max` - BILSTM with Max Pooling

To run the sentEval tests you can run the following command `python .eval.py --model lstm` where you only need specify the model and make sure the trained weights are in the trained_models folder as described above.




