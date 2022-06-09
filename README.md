# Hi there! I'm Yi-Ling 👋
This repository consists of projects that I have done whilst pursuing my masters in Business Analytics at Bayes Business School. 

Through my projects, I developed proficiency in Python, R programming and SQL, and now I can carry out tasks, such as:
- carry out data cleaning and preprocessing 
- web scraping (using BeautifulSoup, Scrapy, Selenium, Twint API, Tweepy API, Reddit API)
- uncover patterns and insights
- communicate crtical findings with stakeholders 


## Languages and Tools: 
<img align="left" alt="Python" width="32px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png" />
<img align="left" alt="R" width="27px" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/r/r.png" />
<img align="left" alt="vscode" width="26px" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/2048px-Visual_Studio_Code_1.35_icon.svg.png" />
<img align="left" alt="GitHub" width="26px" src="https://avatars.githubusercontent.com/u/9919?s=200&v=4" />
<img align="left" alt="sql" width="26px" src="https://www.postgresql.org/media/img/about/press/elephant.png" />

<br />

# Portfolio Projects
### Project 1: Effects of the COVID-19 pandemic on businesses patterns
**Code:**  [`EffectOfCovid19onUKRetailBusinesses.ipynb`](https://github.com/ngyiling/Data-Science-Portfolio/blob/main/EffectOfCovid19onUKRetailBusinesses.ipynb) <br />
**HTML file:** [`EffectOfCovid19onUKRetailBusinesses.html`](https://github.com/ngyiling/Data-Science-Portfolio/blob/main/EffectOfCovid19onUKRetailBusinesses.html) (Please download the html file to view the interactive charts.) <br />
**Description:** This is an individual project to utilise data visualisation tools to understand the impact of the pandemic on business foundation patterns. The [data](http://download.companieshouse.gov.uk/en_output.html) have been obtained from the Companies House data repository, which is a series of compressed folders comprising firm-level attributes.  <br />
**Technology:** Pandas, Geopandas, Geopy, Plotly <br />
**Results:** A bar chart showing an overview number of incorporation of all sectors in 2019 and 2020, an interactive chart explaining how covid casses affect business incorporation in the UK from 2019 to 2020, and an interactive map showing the locations of new retail companies' in London before and after the pandemic. <br />

### Project 2: What makes TikTok' videos (super-)popular!
**Code:** [`TikTok.py`](https://github.com/ngyiling/Data-Science-Portfolio/blob/main/TikTok.py) <br />
**Presentation:** [`TikTok_Presentation.pdf`](https://github.com/ngyiling/Data-Science-Portfolio/blob/main/TikTok_Presentation.pdf) <br />
**Description:** This project was conducted as a group of 4 students. We utilised Python to analyse the [TikTok Trending Videos](https://user-images.githubusercontent.com/72058781/172881986-adfa0eab-28e0-447e-97e3-eed1fb4a5448.png) dataset and produce three statistical charts that complement an article for the printed version of the New York Times on 'what makes TikTok' videos (super-)popular!'. <br />
**Technology:** TetxBlob, Numpy, Pandas, Scikit-learn, Scipy <br />
**Results:** Three statistical charts illustrating the effect of music, duration and sentiment on the popularity of TikTok videos.<br />

 ### Project 3: Predicting wind turbine operating mode based on two series data from 2 sensors using RNN and CNN
 **Code:** [`TimeSeriesPrediction.ipynb`](https://github.com/ngyiling/Data-Science-Portfolio/blob/main/TimeSeriesPrediction.ipynb)<br />
 **h5 file of the best model:** [`best_model.h5`](https://github.com/ngyiling/Data-Science-Portfolio/blob/main/best_model.h5) <br />
 **Description:** This is an individual project to predict a wind turbine operating mode based on time series data from 2 sensors. Various models including Recurrent Neural Network (RNN) and Convolutional Neural Network (CNN) models have been used to identify the best performing model. The project has been guided by the [article on "CNN fault classification based on time series analysis for wind turbine machine" by Rahimilarki, Gao, Jin, and Zhang (2022).](https://www.sciencedirect.com/science/article/abs/pii/S0960148121017778) <br />
 **Technology:** Pandas, Matplotlib, Scikit-learn, Tensorflow, Tensorflow.keras <br />
 **Results:** In addition to training different models including GRU, LTSM, 1D CNN and 2D CNN, the models convolutional layers, epochs, and learning rate have been optimised to obtain accuracies of above 80%. The best model identified was the 2D CNN with three convolutional layers due to its higher accuracy and lower training loss. The final model gave a test accuracy score of 0.91 and a loss of 0.23. 
