
<a name="readme-top"></a>

<div align="center">
 <a href="[https://github.com/othneildrew/Best-README-Template](https://sites.uw.edu/abadi/people/)">
    <img src="/ocean data lab.png" alt="Logo" width="200" height="50">
  </a>
</div>



[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#INTRODUCTION">Introduction</a> </li>
    <li><a href="#PROBLEM-STATEMENT">Problem Statement</a></li>
    <li><a href="#DATA-SOURCES">Data Sources</a></li>
    <li><a href="#PROPOSED-METHOD">Proposed Method</a></li>
    <li><a href="#RESULTS">Results</a></li>
    <li><a href="#FUTURE WORK">Future Work</a></li>
    <li><a href="#REFERENCES">References</a></li>
  </ol>
</details>

# Ocean-data-analysis
Find isolated ships in the region around a hydrophone using an optimized algorithm and use the audio signal captured by the hydrophone to classify ship vessels.

## INTRODUCTION
Vessel's classification is the task in hand which serves many purposes like monitoring maritime traffic and improving defense early warnings. ML classification of vessels is an ongoing research challenge for the community due to lack of publicly available ship data[1]. Most of the previous vessel classification works were based on ships images owing to the rise of many image based classification models like CNN. But there are challenges while using ships images to classify them. There are not enough images and current sources don't have images in all weather conditions. In case of inclement weather conditions, images are hazy and not clear for model to learn from them[1]. Some of the image sources are RADAR images which get affected if the target size is small or its far and it can only capture a part of the ship. These problems can be solved by using infrared or satellite images, but they are affected by weather conditions. The images in visible band can be used but they require a lot of pre-processing and data augmentation techniques to fetch diversified samples. On the other hand, one less explored data source is audio signals which can capture ships noise in any weather conditions and can be collected by using comparatively less expensive instruments like hydrophones and require little or no human involvement. However, there are few publicly available ships' audio data for research.

We are hence creating a publicly available audio data source for ships and using it to demonstrate a vessel classification model using ML techniques. In the next sections, we will discuss the problem statement, data sources used, and algorithms implemented to create the benchmark data and ML classification model.

## PROBLEM STATEMENT
- Devise an optimised algorithm to extract the isolated ships' noises from the hydrophone's audio data and publicly available [AIS data](https://marinecadastre.gov/ais/) containing GPS coordinates of the ships and other meta-data.
- Build a ML model to classify ships of different vessels types using Machine Learning techniques trained on the benchmark data created above.

## DATA SOURCES
AIS data is used to collect ships metadata including information like geo-location, speed and vessel type, and label the hydrophones' recordings through our proposed method. . Hydrophone's recordings are collected from hydrophones of Ocean Observatories Initiative which is an ocean observing network providing data from more than 800 instruments[3]. There are total 11 hydrophones in the OOI sensor network and for our purpose, we used 3 hydrophones- [Axial Base](https://interactiveoceans.washington.edu/research-sites/axial-base/), [Eastern Caldera](https://interactiveoceans.washington.edu/research-sites/axial-caldera/eastern/) and [Central Caldera](https://interactiveoceans.washington.edu/research-sites/axial-caldera/central/). More information on the hydrophones can be found in the links attached. Since the hydrophones are placed in real environment, it's natural to expect background noises from marine mammals , tides and rain among other natural phenomena. The hydrophones' positions are fixed and hence, proposed method was used to extract hydrophones' recordings when an isolated ship is around the fixed hydrophones.

## LITERATURE SURVEY
The two popular publicly available ships' underwater acoustic datasets-  DeepShip[5] and ShipsEar[5] were compared before devising logic to extract ships noise. The proposed ShipsCry dataset overcome the limitations of the two discussed publicly available datasets and supplement it. In ShipsEar[4], Davis et. al used 3 hydrophones with different gains and depths which were deployed whenever possible and the recording with highest sound level was captured for the database. The targeted vessels were visually identified during the recordings. The authors also removed recordings with excessive noises or the ones with ambiguous information on vessels. While the approach ensures good quality of the recordings, the approach is an expensive one and requires human resources. In the paper DeepShip, Muhammad Irfan et al. used just one hydrophone to record the audio data. The total recording duration was divided into 3 periods and in each of the periods, the hydrophone was placed at different depths and in different locations. In the work, authors used AIS data to label the recorded data. For the dataset DeepShip, whenever a ship was within 2 km radius and in continuous timestamps, it was labeled against the recorded data in the same timeframe. But it was not ruled out that there are no other ships around the 2km radius in the same continuous timestamps which does not guarantee the quality of the recordings. In the two mentioned datasets, the number of instances of recorded ships are also few.
Through our proposed method, we are trying to overcome the aforementioned shortcomings. Using the proposed logic, we were able to extract more number of isolated ship instances with increased amount of quality recording durations by ensuring that there are no other ships nearby when recording starts apart from the ship in interest.
  
## PROPOSED METHOD
We have to find isolated ships within certain radius. We can define isolated ships as the ships which are within x radius when no other ships are 'around'. So we 
defined an outer radius as well and then an isolated ship can be defined as a ship which is within x radius when there are no other ships within the y outer radius.

Approach I.
Define inner radius x and outer radius infinity.
Find all the ships within x inner radius. Get the start time(minimum timestamp) and end time(maximum timestamp) and check if there are any ships within these two timestamps. If there are no other ships , then we consider this ship as an isolated ship. One problem with this approach is that a ship may have gone out of the inner radius between start and end time. While the ship may not be strictly within x inner radius, it is still isolated since we checked there are no other ships between the start and end time.
However, since we are checking against the entire AIS if there are any other ships within the start and end time, it leads to fewer samples of isolated ships.

Approach II. 
Define inner radius x and outer radius y.
This is same as approach I except that now we have a finite outer radius. This approach will yield more samples as it is more lenient. we call a ship isolated if there are no other ships within the outer radius between start and end time instead of ensuring there are no other ships in the entire AIS data between the start and end time.
However the ship still may not be strictly within the inner radius between the start and end time. And for the timestamps when the ship went out of the inner radius, it could be anywhere and may be even close to the outer radius and closer to other ships. In those timestamps , we cannot call the ship isolated and the hydrophone recordings will also not be accurate.

Approach III.
Define inner radius x and outer radius y.
In this approach, we first selected all the ships within the outer radius and sorted the resultant dataset by timestamps in ascending order. Now, we captured the continuous timestamps from the beginning when the ship id is same and the ship is within the x inner radius. This ensures that the ship is constantly within the inner radius and since the timestamps were sorted originally, there cannot be any other ships.
The idea behind this algorithm is to capture the recordings as long as the ship id is not changing or the ship is not going out of the inner radius. With this approach we were able to capture even more samples. A single ship may be isolated between different start and end times within the inner radius and all such instances are captured with this approach.

However, we found that the length of the recordings of each instance( a ship with start and end time when it was isolated) can vary from seconds to hours. The instances with small length of recordings are suspicious as there could be a ship around which is not recorded in the AIS data since each of the ships send their GPS coordinates at varied timestamps. To ensure the accuracy of the recordings, we decided to put a filter on the length of the recordings. We kept the recordings with length greater than 'dt' minutes, under the assumption that if a ship is not sending coordinates within dt time interval, probably its not around.

However , an ideal approach would be to check the distribution of time between two successive GPS pings of a ship and find on an average what is interval between two successive pings. Lets say its dt2. So, we should use dt2 to filter out recordings in the previous step.





## RESULTS

## FUTURE WORK

## REFERENCES

1. (https://mdpi.com/2078-2489/12/8/302/htm)
2. (https://www.frontiersin.org/articles/10.3389/fnbot.2022.889308/full)
3. (https://asa.scitation.org/doi/abs/10.1121/10.0007594)
4. (https://www.sciencedirect.com/science/article/pii/S0003682X16301566)
5. (https://www.sciencedirect.com/science/article/pii/S0957417421007016)


The aim of the project is to release a public datasets of different ships for the underwater acoustic research community and develop a ML model to classify different vessel types trained on the same dataset.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-url]: https://www.linkedin.com/in/khirodcsahoo/
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]:https://github.com/khirodsahoo93/Ocean-data-analysis/blob/main/license.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[ocean-shield]:https://github.com/khirodsahoo93/Ocean-data-analysis/blob/main/ocean%20data%20lab.png
[ocean-url]:https://sites.uw.edu/abadi/people/
