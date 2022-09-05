# Author Response to our submitted paper "Leveraging Translationese in Pretraining Quality Estimation Models"

General response 1: QE Task definition

>Both WMT20 and WMT21 official QE shared tasks include 3 subtasks. In each subtask, the QE model is only provided with a source text and its machine translation produced by SOTA transformer-based NMT systems (https://github.com/facebookresearch/mlqe/tree/main/nmt_models). 

>Task 1 in WMT20/21 is a sentence-level regression task, in which the QE model predicts a human direct assessment (DA) score. The original annotation are scores between 0 and 100, then the ground truth scores are z-normalized and evaluate against model prediction using Pearson’s *r* correlation.

>In WMT20/21 Task 2, QE model measures the minimum edit distance between the machine translation and its manually post-edited version, which includes both a sentence-level score (HTER) and word-level binary tags (OK/BAD). The ground truth labels are **automatically** computed using the TERCOM tool (https://github.com/jhclark/tercom) that counts all the operations (insert, delete and shift) of transforming a translation to its post-edited version, and thus we can obtain quality tags for each word (BAD while the word is transformed, otherwise OK) in the target sentence (See upper right of Figure5). 
The word-level subtask involves two other types of quality tags: 
>1)  Tags for [GAP] tokens in target text (lower left of Figure5). [GAP] tokens are inserted between all target words. If there is an “insertion” operation between two target words, then the [GAP] tag should be BAD, and otherwise OK.
>2) Tags for words in source text (upper left of Figure5). Fast align is used to create alignments between source and target. If the aligned word for a source word is tagged as OK, then the source word is OK, otherwise BAD.

>WMT21 Task 3, critical error detection (CED, lower left of Figure5), is a sentence-level binary classification task predicting whether there is a critical translation error in a machine translation.


General response 2: Prevalence of parallel text in QE data augmentation and importance of translation direction.

>We check all the submissions to WMT20/21 QE task, and summarize as follow:

|	__Task__	|	WMT20	|	WMT21	|

|	__Task1__	|	4/13	|	4/9	|

|	__Task2__	|	8/12	|	6/7	|

|	__Task3__	|		-		|	3/4	|

>in which 4/13 means that there are 13 submissions to WMT20 Task1, 4 of them involve the use of parallel corpus for data augmentation. 
However, **none** of the submissions consider translation direction. Our proposed approach is a general optimization of data augmentation for various QE tasks.
