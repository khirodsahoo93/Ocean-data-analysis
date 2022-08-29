# Ocean-data-analysis
Find isolated ships in the region around a hydrophone using an optimized algorithm and use the audio signal captured by the hydrophone to classify ship vessels.

## INTRODUCTION
Vessels classification is the task in hand which serves many purposes like monitoring maritime traffic and improving defense early warnings. ML classification of vessels is an ongoing research challenge for the community due to lack of publicly avaiable ship data[1]. Most of the previous vessel classification works were based on ships images after the rise of many image classification models like CNN. But there are challenges while using ships images to classify them. There are not enough images and current sources don't have images in all weather conditions. In case of inclement weather conditions, images are hazy and not clear for model to learn from them[1]. Some of the image sources are RADAR images which gets affected if the target size is small or its far and it can only capture a part of the ship. These problems can be solved by using infrared or satellite images but they are affected by weather conditions. The images in visible band can be used but they require a lot of pre-processing and data augmentation techniques to fetch diversified samples.
On the other hand, one less explored data source is audio signals which can capture ships noise in any weather conditions and can be collected by using comparitively less expensive instruments like hydrophones and require little or no human involment. However, there are not many publicly available ships' audio data for research.

We are hence, creating a publicly available audio data source for ships and using it to demonstrate a vessel classification model using ML techniques.
In the next sections, we will discuss the problem statement, data sources used and algorithms implemented to create the benchmark data and ML classification model.

## Problem statement

## Data Sources

## Algorithm and Implementation

## Results

## Future Work

## References

[1] (https://mdpi.com/2078-2489/12/8/302/htm)


The aim of the project is to release a public datasets of different ships for the underwater acoustic research community and develop a ML model to classify different vessel types trained on the same dataset.
