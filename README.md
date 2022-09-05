# Author Response to our submitted paper "Leveraging Translationese in Pretraining Quality Estimation Models"

## Response to Reviewer3
Q1. Is a parallel corpus either source-original or target-original? 
>Following the reviewer's suggestions, we conduct an in-depth analysis of the parallel dataset we used for QE data augmentation. The parallel dataset includes several corpora from different domains, and indeed some of them have no obvious translation direction. 

>We randomly sample 100 sentence pairs from each corpus and manually check 1) Whether the sentence pair has an explicit translation direction. 2) What the translation direction is. Details are summarized as follow:


| __Corpus__                             |  Parallel text with Explicit Translation Direction  |  English-Original | Chinese-Original |

| __ParaCrawl v9__                   | 55 | 39 | 16 |

| __newscommentary v16__      | 91 | 91 | 0 |

| __WikiTitles_v3__                    | 63 | 31 | 32 |

| __UN Parallel corpus__           | 0   |  0  | 0 |

| __WikiMatrix__                        | 82 | 40 | 42 |

| __CCMT__                              | 56 | 29 | 27 |

| __QE corpus__                       | 57 | 57 | 0  |

>We find that the translation direction is strongly related to the content of a sentence pair. For the UN parallel corpus, since most sentence pairs are from aligned official documents, there is no explicit translation direction. 

>Although Wiki-Titles and WikiMatrix are aligned articles,  a considerable proportion of texts are descriptions of language-specific items and concepts, indicating that they are originally written in a specific language, and then translated to another language. 

>We find it easier to identify translation direction under news domain and  based on language-specific name entities, such as name, location, which requires cultural background knowledge. Even for human annotators, it's difficult to identify the origin language of a sentence based on only stylistic features, which aligns with findings in our paper.

Q2. The improvements are only marginal to begin with.
>As explained in line#236~line#242, the "mixed” bitext presents similar features with source-original bitext since the WMT22 EN-ZH parallel dataset is biased towards English original bitext. We present our experiments for other language pairs of which the translation direction is more balanced in general response#3 to better demonstrate the effectiveness of distinguishing translation direction.

>According to our findings, some corpora in WMT22 EN-ZH do not have an explicit translation, thus we also conduct additional experiments using only WikiMatrix  for data augmentation. The results is presented in general response#3.

Q3. Pseudo labels
>We use HTER as the pseudo label computed by TERCOM tool (https://github.com/jhclark/tercom) following WMT official settings. Details are elaborated in general response#1.

Q4. Typo in Figure 3's caption leads to self-contradiction.
>We are sorry for the typo, since we mean **lower** lexical density and variety denote a higher degree of translationese.  We will polish the presentation in the revised version, as detailed in response Q1 to reviewer#1.




## General response 1: QE Task definition

>Both WMT20 and WMT21 official QE shared tasks include 3 subtasks. In each subtask, the QE model is only provided with a source text and its machine translation produced by SOTA transformer-based NMT systems (https://github.com/facebookresearch/mlqe/tree/main/nmt_models). 

>Task 1 in WMT20/21 is a sentence-level regression task, in which the QE model predicts a human direct assessment (DA) score. The original annotation are scores between 0 and 100, then the ground truth scores are z-normalized and evaluate against model prediction using Pearson’s *r* correlation.

>In WMT20/21 Task 2, QE model measures the minimum edit distance between the machine translation and its manually post-edited version, which includes both a sentence-level score (HTER) and word-level binary tags (OK/BAD). The ground truth labels are **automatically** computed using the TERCOM tool (https://github.com/jhclark/tercom) that counts all the operations (insert, delete and shift) of transforming a translation to its post-edited version, and thus we can obtain quality tags for each word (BAD while the word is transformed, otherwise OK) in the target sentence (See upper right of Figure5). 
The word-level subtask involves two other types of quality tags: 
>1)  Tags for [GAP] tokens in target text (lower left of Figure5). [GAP] tokens are inserted between all target words. If there is an “insertion” operation between two target words, then the [GAP] tag should be BAD, and otherwise OK.
>2) Tags for words in source text (upper left of Figure5). Fast align is used to create alignments between source and target. If the aligned word for a source word is tagged as OK, then the source word is OK, otherwise BAD.

>WMT21 Task 3, critical error detection (CED, lower left of Figure5), is a sentence-level binary classification task predicting whether there is a critical translation error in a machine translation.


## General response 2: Prevalence of parallel text in QE data augmentation and importance of translation direction.

>We check all the submissions to WMT20/21 QE task, and summarize as follow:

|	__Task__	|	WMT20	|	WMT21	|

|	__Task1__	|	4/13	|	4/9	|

|	__Task2__	|	8/12	|	6/7	|

|	__Task3__	|		-		|	3/4	|

>in which 4/13 means that there are 13 submissions to WMT20 Task1, 4 of them involve the use of parallel corpus for data augmentation. 
However, **none** of the submissions consider translation direction. Our proposed approach is a general optimization of data augmentation for various QE tasks.
