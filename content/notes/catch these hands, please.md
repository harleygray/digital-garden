---
title: "🖐🏽catch these hands, please🖐🏽"
tags:
- ML
description: training a neural network to recognise hand gestures
---
# surprised by simplicity
what is the hardest part of developing machine learning models? as a budding practitioner i assumed it would be the coding or the maths, but i underestimated a key aspect: finding data. this is a surprisingly demanding task and is essential in order to build useful models.  

my goal here is to point beginners in a direction that minimises frustration and maximises cool things built. 

the basic flow of this project (and blog post) is as follows:
1. define the task
2. acquire/clean data
3. train the model

task: train a model that can recognise hand gestures in photos. i want it to recognise at least 👍🏽 and 👎🏽, and optimally one-five fingers up too.

to get this project going quickly, we can start with a 'pre-trained' model that has been developed by experts and trained on large databases. rather than building a model from scratch, we can stand on the shoulders of giants and modify a pre-trained model for this purpose.  

the crux of this project boils down to this step: modifying (or *fine tuning*) a pre-trained model. the starting model is called resnet18, which was trained on more than a million images. it's very efficient at image classification and so is perfect to be fine-tuned. because resnet18 is already able to recognise objects, the fune-tuning step requires us to provide specific data on the thing we want to recognise. in this case, we'll need to provide images of hand gestures and the corresponding 'label', i.e. what hand gesture is being done in a photo. 


# the search for data
so the search for data began. i tried DuckDuckGo, Google, and even Bing! i tried many combinations of search terms, such as  `hand thumb down -angry` but to no avail. i considered supplying the model with my own photos, but realised that the amount i would need to do would be impractical. then, unexpectedly, Hagrid saved me.

![[notes/images/hagrid.jpg]]

i mean, [HaGRID - HAnd Gesture Recognition Image Dataset](https://arxiv.org/abs/2206.08219). this is an enormous dataset of ~550k images, each of a person doing one of 18 gestures such as thumbs up, one finger, two fingers, etc. wonderful! just what i was looking for. 

> [!hint] where to find data
>
> Kaggle has all kinds of cleaned data. You'll find data suitable for similar tasks [here](https://www.kaggle.com/datasets?tags=13207-Computer+Vision)

the full size of this dataset is 716GB, the largest dataset i've ever worked with. i figured that i wouldn't be able to just use my free tier of Kaggle to handle such a large dataset, so i used the supplied sample dataset which containsed 100 of each of the 18 gestures. training a model on this set was a good first step, but didn't provide the results to be useful.

next, i found a scaled down version of this dataset, with 512p images instead of the higher quality original photos. these were modified to be lower resolution for image classification instead of object detection (i.e. recognising where in the image a hand is located). i hadn't even realised that the original HaGRID dataset was meant for both purposes. by finding this dataset, i was able to continue with the project. 




# model
> after loading the dataset, it was time to get to grips with the fast.ai library. there were many notebooks online from which to draw inspiration and lessons. in particular, there was a notebook which went through the different steps of getting the image files from the kaggle dataset, marking them with a classification label so it could then be loaded into the fast.ai library. i still have lots to learn here, but i accept glossing over this particular section in the name of learning more deep learning.

after loading the dataset, it was time to grapple with the fast.ai library. luckily there are many notebooks online from which to draw inspiration and lessons. in particular [this](https://www.kaggle.com/code/stpeteishii/hagrid-18-classify-fasiai/data) notebook went through the different steps of accessing the files from the kaggle dataset, marking them with a classification label so they can be loaded into the fast.ai library. 

once data is prepared, it needs to be packaged into a "data loader" object, `dls` in the code block below. here, the images are already in the `train_df` dataframe, with `fn_col` and `label_col` instructing the program where to look for paths to the image and its label, respectively. the `valid_pct` parameter is the percentage of items to set aside for verification in each training epoch.

```python
dls = ImageDataLoaders.from_df(train_df, 
                               fn_col=0, #path
                               label_col=1, #label
                               folder='./', 
                               valid_pct=0.2, 
                               item_tfms=Resize(224))
dls.show_batch(max_n=9)
```

![](https://www.kaggleusercontent.com/kf/107717335/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Gm8Fz8Ptp1uHi8G9I9iaQQ.K4SK7TDyxs6a4pvyK_B7ATHsGQ8qzuBxaQnK-fA9y4pI8b8ffdPRnV2RBFvaqMEOJxsVjreD-EKHz-5d7FSOBeXtXxKx_T9eiYJzaNHDf0lesuj0qa2MaTs427L__gQ6MGyiBnQ34HCfsXD71m4v2KDAa-l-S_OnB4zoFRwakD80162t8OlYQKNtmriVzjX1ThJmumkc3LHOtfnqmmRwp3oq1P-TcICjsOLVRRuwd0qcIJmcs_sVxLWME3Mu556OdwkAqL9G2oO0DULip2X2cKffhauxA4jChxPIvRbj6RQYpM_cC9krZu_SyzgG9lXfy_BU9Jq7mTQ94gbKVSBgURBoRxm8qjiDtbqHqazkZ34DiMZ6gHie1WZB1KTS9jSdzmjMs0KShX3jEaIEsYOuhoxHrLjllALZiNGkr6CdUj4xBNKxQosMIFdBdGiljQK7HwF4kXFjp4xa63qCmWn0ViB29GM3aJix79SSK17JktpArENhqjhmlhB_LZ1fN33Df8wJkDPOIeO-LwltO8Vfia593oP8oFUAaZ_gyU58NR4sQNFPp7IA447FH__nO1mqVZ4SbW2T7CKjkur025eMmRPI8RLJOFssGj15M64syV8k73TFPg6NBs-zkpzm7B40brm9QaGRcPmCFnCDsQ8hdGnYKiQuL7IUk0HVKqPXyvI.Jux3vjQ24F75ZM-NquPT_A/__results___files/__results___11_0.png)

each of the nine images above are now the same size, and have been labelled appropriately. after the images are loaded in and labelled, the next step is to pass them into the deep learner. here is where we specify the pre-trained model (resnet18) that we'd like to begin with.  

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(12)
```

![[notes/images/fine_tuning.png]]

after twelve cycles, the model correctly classifies hand gestures 70% of the time. any further fine tuning did not make this any more accurate. i have some ideas as to why that might be, which i will investigate in the future. 

the model has been deployed to a site called HuggingFace [here](https://huggingface.co/spaces/harleygray/fastai-hand-classifier). you can upload your own photo of you doing a hand gesture and see if it recognises which you are doing! 