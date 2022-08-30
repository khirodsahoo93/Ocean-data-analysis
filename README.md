<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Ocean Data Lab][ocean-shield]][ocean-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# Ocean-data-analysis
Find isolated ships in the region around a hydrophone using an optimized algorithm and use the audio signal captured by the hydrophone to classify ship vessels.

## INTRODUCTION
Vessels classification is the task in hand which serves many purposes like monitoring maritime traffic and improving defense early warnings. ML classification of vessels is an ongoing research challenge for the community due to lack of publicly avaiable ship data[1]. Most of the previous vessel classification works were based on ships images after the rise of many image classification models like CNN. But there are challenges while using ships images to classify them. There are not enough images and current sources don't have images in all weather conditions. In case of inclement weather conditions, images are hazy and not clear for model to learn from them[1]. Some of the image sources are RADAR images which gets affected if the target size is small or its far and it can only capture a part of the ship. These problems can be solved by using infrared or satellite images but they are affected by weather conditions. The images in visible band can be used but they require a lot of pre-processing and data augmentation techniques to fetch diversified samples.
On the other hand, one less explored data source is audio signals which can capture ships noise in any weather conditions and can be collected by using comparitively less expensive instruments like hydrophones and require little or no human involment. However, there are not many publicly available ships' audio data for research.

We are hence, creating a publicly available audio data source for ships and using it to demonstrate a vessel classification model using ML techniques.
In the next sections, we will discuss the problem statement, data sources used and algorithms implemented to create the benchmark data and ML classification model.

## PROBLEM STATEMENT
- Devise an optimised algorithm to extract the isolated ships' noises from the hydrophone's audio data and publicly available [AIS data](https://marinecadastre.gov/ais/) containing GPS coordinates of the ships and other meta-data.
- Build a ML model to classify ships of different vessels types using Machine Learning techniques trained on the benchmark data created above.

## Data Sources

## Algorithm and Implementation

## Results

## Future Work

## References

1. (https://mdpi.com/2078-2489/12/8/302/htm)
2. (https://www.frontiersin.org/articles/10.3389/fnbot.2022.889308/full)


The aim of the project is to release a public datasets of different ships for the underwater acoustic research community and develop a ML model to classify different vessel types trained on the same dataset.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-url]: https://www.linkedin.com/in/khirodcsahoo/
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]:https://github.com/khirodsahoo93/Ocean-data-analysis/blob/main/license.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[ocean-shield]:https://github.com/khirodsahoo93/Ocean-data-analysis/blob/main/ocean%20data%20lab.png
[ocean-url]:https://sites.uw.edu/abadi/people/
