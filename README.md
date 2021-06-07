TEKM-ranker
-------------

A pytorch implement for WWW 2021 paper ''[Topic-enhanced knowledge-aware retrieval model for diverse relevance estimation](http://www.thuir.cn/group/~YQLiu/)'', namely `Topic Enhanced Knowledge-aware retrieval model(TEKM)`.

## Requirement
* Python 2.7
* Pytorch 0.4.1
* tqdm

## Dataset
We run experiment on the publicly available dataset [Tiangong-ST](http://www.thuir.cn/tiangong-st/), which is a Chinese search log from [Sogou.com](sogou.com). 

* First, edit the settings in `config.py`. 
* Sampled data files are given in `./data/sample_data/valid(test)` folders. Each line consists of `qid	docid	query	title	TACM	PSCM	THCM	UBM	DBN	POM	HUMAN(Only available in test set)`, separated by `TAB`. In particular, `TACM	PSCM	THCM	UBM	DBN	POM` are the click labels given in the dataset.

## Procedure
1. All the settings are in `config.py`.
2. run `python main.py --prototype ierm_config --gpu 0`

