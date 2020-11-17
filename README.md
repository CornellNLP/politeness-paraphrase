# Politeness Paraphrase

This repository contains code and data for the paper: [Facilitating the Communication of Politeness through Fine-Grained Paraphrasing](http://www.cs.cornell.edu/~cristian/Politeness_Paraphrasing.html). Liye Fu, Susan Fussell and Cristian Danescu-Niculescu-Mizil. EMNLP 2020. 


### Data

Training and test corpuses are prepared from [WikiConv](https://convokit.cornell.edu/documentation/wikiconv.html) and the [Stanford Politeness Corpus](https://convokit.cornell.edu/documentation/wiki_politeness.html) and can be found in [data](data) (refer to the data [README.md](data/README.md) for more details).  


### Training and Evaluation 

We include a few notebooks to explain our training and evaluation procedures: 

- [Training_Generation_Model.ipynb](Training_Generation_Model.ipynb) explains details about training data and how training is done.
- [Evaluation_Data.ipynb](Evaluation_Data.ipynb) provides further details on evaluation data and additionally preparatory steps to set up the evalution.
- [Evaluation.ipynb](Evaluation.ipynb) demos how we obtain results reported in the paper. 

Our pretrained model for adding strategies into messages can be directly downloaded from [here](https://zissou.infosci.cornell.edu/convokit/models/politeness_gen.bin) (you will need to update the model path in settings.py for some of the notebooks). For reference, generation outputs are provided under:

```
/outputs
    mt.tsv
    ind.tsv
```

For a demo on how this approach can be applied to custom texts, see [How_to_Make_Strategy_Edits.ipynb](How_to_Make_Strategy_Edits.ipynb). 


### Dependencies 

    * convokit=2.4.3
    * spacy=2.2.1
    * PulP=1.6.8
    * GLPK=4.65
    * scikit-learn=0.21.3
    * dependencies from pytorch_pretrained_bert


### Cite as 

```
@InProceedings{fu-paraphrase:2020,
  author={Liye Fu, Susan R. Fussell and Cristian Danescu-Niculescu-Mizil},
  title={Facilitating the Communication of Politeness through Fine-Grained Paraphrasing},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```

