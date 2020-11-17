Data used to set up the experiments, organized as follows:

```
/train
   training-corpus
   training-files

/test
   mt-test-corpus
   ind-test-corpus

/perceptions
   turker-corpus
   <turker_id>.json: perception models saved as json files 
```           

`train` contains instances sampled from [WikiConv](https://convokit.cornell.edu/documentation/wikiconv.html) exemplifying uses of different politeness strategies. We keep both the collection of raw texts in ConvoKit format (`training-corpus`), as well as the processed version that's used to train the generation model (see `training-files`).

`test` contains test messages used in the two respective experiments: `mt-test-corpus` for machine-translated communication (Experiment A), and `ind-test-corpus` for misaligned individual perceptions (Experiment B). 

`perceptions` contains by-annotator annotations from the [Stanford Politeness Corpus](https://convokit.cornell.edu/documentation/wiki_politeness.html) (turker-corpus), and the perception models we trained from these annotations. 


More details can be found in the following two notebooks: 

- [Training_Generation_Model.ipynb](Training_Generation_Model.ipynb) explains details about training data and how training is done.
- [Evaluation_Data.ipynb](Evaluation_Data.ipynb) provides further details on evaluation data and additionally preparatory steps to set up the evalution.
