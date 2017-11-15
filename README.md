# Bag of region embeddings via local context units for text classification

## 0. Requirements 

### General
	Python (verified on 2.7.13)

### Python Packages
	tensorflow(verified on 1.0)
	

## 1. Datasets
    We use publicly available datasets from Zhang et al.(2015) to evaluate our models.
    The datasets can be obtained from [here](https://github.com/zhangxiangxiao/Crepe).

## 2. Pre-processing
    First, download the datasets and place them in `data` directory.
    
    Second, pre-process the datasets:
    ```
    	sh run.sh pre-process $data_dir
    ```

## 3. Training
	Specify the target dataset in the training configure(conf/model.config), and run:
	```
		sh run.sh train
	```


