# %% install package
!pip install TextBlob
#%% Set-up
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from os import execle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

#%% matplotlib viz set-up
rc('font',**{'family':'sans-serif','sans-serif':['Franklin Gothic']})

#%% importing data
# Open JSON file
file = open('trending.json', encoding="utf8")
# Parse JSON
data = json.load(file)
# Close file
file.close()
# Show amount of objects
len(data['collector'])
# Split the objects to separate columns and store everything as a DataFrame
df = pd.json_normalize(data['collector'])
df = df.drop(['createTime', 'webVideoUrl', 'videoUrl', 'videoUrlNoWaterMark',
              'authorMeta.secUid', 'authorMeta.avatar', 'musicMeta.playUrl',
              'musicMeta.coverThumb', 'musicMeta.coverMedium', 'musicMeta.coverLarge',
              'covers.default', 'covers.origin', 'covers.dynamic'], axis=1)

#%% Create a popularity metric
# creating a new dataframe for normalisation
df1 = df[["diggCount", "shareCount", "playCount", "commentCount"]]
# normalising the measurements for popuplarity, i.e. diggcount, sharecount, playcount and commentcount
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(df1)
df_n = pd.DataFrame(X_minmax, columns=("diggCount_n", "shareCount_n", "playCount_n", "commentCount_n"))
# merging the normalised df(df_n) with original df (df)
df = df.merge(df_n, 'left', left_index = True, right_index = True)
# creating a popularity metric by taking the average of the normalised measures
df["popularity_metric"] = (df["diggCount_n"] + df["shareCount_n"] + df["playCount_n"] + df["commentCount_n"])/4

#%% CHART 1: MUSIC
#%% Merge with Spotify data
# Import Audd Data
df_audd_music = pd.read_csv('audd_music.csv')
df_audd_music_spotify = pd.read_csv('audd_music_spotify_music.csv')
df_audd_music_spotify_artists = pd.read_csv('audd_music_spotify_music_artists.csv')
# Dropping unnecessary columns and duplicated rows
df_audd_music_spotify = df_audd_music_spotify[['id', 'popularity', 'name', 'artists_ids']]
df_audd_music_spotify.rename(columns = {'id':'spotify.id'}, inplace = True)
df_audd_music = df_audd_music[['id', 'spotify.id']]
df_audd_music_spotify.drop_duplicates(inplace = True)
df_audd_music.drop_duplicates(inplace = True)
# Merging the music dfs
df_music = df_audd_music.merge(df_audd_music_spotify, on = 'spotify.id')
df_music.rename(columns = {'id':'musicMeta.musicId'}, inplace = True)
# Converting the column dtype of musicmeta music id in main df to int64 to merge
df['musicMeta.musicId'] = df['musicMeta.musicId'].astype('int64')
df0 = df.merge(df_music, how = 'left', on = 'musicMeta.musicId')
df0.rename(columns = {'popularity':'spotify_popularity'}, inplace = True)

#%% Plotting Spotify popularity vs TikTok videos
# filling NA rows with 0
df0.loc[df0["spotify_popularity"].isna(), "spotify_popularity"] = 0
# Creaing a df for original vs non-original soundtrack
df_og_prop = df0.groupby(['musicMeta.musicOriginal']).size().reset_index()
df_og_prop["percentage"] = df_og_prop[0]/10

#%% Plotting the chart
# tiktok colours: #25F4EE (turquoise); #FE2C55 (red); black; white
fig = plt.figure(figsize = (10, 8))

ax1 = fig.add_subplot(111)
ax2 = ax1.twinx() # dual y-axes

categories = ['Original', 'Existing']
bins_list = np.arange(0, 110, 10)

x = df0.loc[df0['musicMeta.musicOriginal'] == True, 'spotify_popularity'].reset_index(drop=True)
y = df0.loc[df0['musicMeta.musicOriginal'] == False, 'spotify_popularity'].reset_index(drop=True)
ax1.hist([x, y], histtype="barstacked", color=["k", "#25F4EE"],
         alpha=1, bins = bins_list, rwidth = 0.95)

# plotting lollipop chart to represent the count of songs
x_s = df_audd_music_spotify['popularity']
n, bins, patches = ax2.hist(x_s, histtype="step", color = 'k', alpha = 0, bins = bins_list)
bincenters = 0.5*(bins[1:]+bins[:-1])
ax2.scatter(bincenters, n, color = '#FE2C55', label = "Number of Songs", marker = ".", zorder = 2, alpha = 1) 
for i, j in zip(bincenters, n):
    ax2.vlines(x=i, ymin=0, ymax=j, color = '#FE2C55', ls = ":", alpha = 0.7)
# ax2.hlines(y=np.max(n), xmin=75, xmax=100, color = '#FE2C55', ls = ":", alpha = 0.5)

# add legend
ax2.legend(loc='lower right', bbox_to_anchor=(0.98, 0.22), fontsize = 12, frameon = False)

# format axes
ax1.set_ylabel("Number of Tiktok videos", fontsize = 12, color = 'dimgrey')
ax1.set_ylim(0, 700)
ax1.set_xticks(bincenters)
ax1.set_xticklabels([])
ax1.tick_params(labelsize=8, colors = 'dimgrey')
ax1.set_yticks([0, 200, 400, 600])
ax2.set_ylabel("Number of songs on matched on Spotify", fontsize = 12, color = 'dimgrey')
ax2.set_yticks([0, 100, 200, 300])
ax2.set_ylim(0, 350)
ax2.tick_params(labelsize=8, colors = 'dimgrey')


# remove spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# add gridlines
ax1.grid(ls= '--', axis = 'y', color = 'grey')
ax1.set_axisbelow(True)

# Add annotation
ax1.annotate('More popular on Spotify',
            xy=(100, -30),
            xytext=(60, -35),
            size=12,
            color = 'dimgrey',
            arrowprops=dict(arrowstyle="simple",
                            fc="dimgrey", ec="none"),
            annotation_clip = False)

ax1.set_title("Original soundtracks win over\nthe heart of TikTokers", fontsize = 22, fontname = "Cheltenham")
# add descrption
ax1.annotate('This chart illustrates the the number of trending TikTok videos using songs or soundtracks across a\nrange of Spotify popularity scores between 0-100',
            xy=(0, 660),
            size=12,
            color = 'k',
            annotation_clip = False)

ax1.annotate('The majority of the\ntrending TikTok videos\nused soundtracks\nthat are not found or\nranked low on Spotify.',
            xy=(12, 450),
            size=10,
            color = 'dimgrey',
            annotation_clip = False)
# annotate subtitle/description - use franklin gothic

#ax1.annotate('Therefore I am - Billie Eilish',
 #           xy=(95, 45),
  #          xytext=(85, 100),
   #         size=10,
    #        color = 'dimgrey',
     #       arrowprops=dict(arrowstyle="simple",
      #                      fc="dimgrey", ec="none"),
       #     annotation_clip = False)

ax1.annotate('Showing the number of\nsoundtracks in our dataset\nthat are available on Spotify\nin each popularity range',
            xy=(83, 100),
            size=10,
            color = 'dimgrey',
            annotation_clip = False)

# inset# draw graph in inset
axins = inset_axes(ax1, width=2.5, height=1.5)
original = df_og_prop.loc[df_og_prop["musicMeta.musicOriginal"] == True, "percentage"]
existing = df_og_prop.loc[df_og_prop["musicMeta.musicOriginal"] == False, "percentage"]
axins.barh(1, original, height=1.5, color = 'k')
axins.barh(1, existing, height=1.5, color = '#25F4EE', left = original)

# set y-axis
axins.set_ylim(0, 10)
axins.set_xlim(0, 120)

# set ticks
axins.set_xticks([])
axins.set_yticks([])
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)
axins.spines['bottom'].set_visible(False)
axins.spines['left'].set_visible(False)
axins.patch.set_alpha(0)

# annotate
axins.annotate('Original: 75%',
            xy=(0, -1),
            size=10,
            color = 'k',
            annotation_clip = False)
axins.annotate('Existing: 25%',
            xy=(75, -1),
            size=10,
            color = 'k',
            annotation_clip = False)
axins.annotate('Overall Proportion\nof Videos Using\nOriginal Soundtracks',
            xy=(-65, -0.5),
            size=10,
            color = 'k',
            annotation_clip = False,
            fontweight="bold")
#comment on the first bar

# Save plot as png
plt.savefig("tiktok_spotify.png",
            transparent=True,
            pad_inches=0,
            dpi=1200)


plt.show()
# %% CHART 2: DURATION
# %%open json file
file = open('trending.json',encoding='utf8')

#parse json
tk = json.load(file)

# Select only the list with the video data
trending_videos_list = tk['collector']

# Example of a video object
print(json.dumps(trending_videos_list[15], indent=4, sort_keys=True))
# %%
# Create a DataFrame of the data
df_orginal = pd.DataFrame(trending_videos_list)

# Let's expand the hashtag cell containing lists to multiple rows
df_explode = df_orginal.explode('hashtags').explode('mentions')
# %%
def object_to_columns(dfRow, **kwargs):
    '''Function to expand cells containing dictionaries, to columns'''
    for column, prefix in kwargs.items():
        if isinstance(dfRow[column], dict):
            for key, value in dfRow[column].items():
                columnName = '{}.{}'.format(prefix, key)
                dfRow[columnName] = value
    return dfRow

# Expand certain cells containing dictionaries to columns
df_explode = df_orginal.apply(object_to_columns, 
                            authorMeta='authorMeta',  
                            musicMeta='musicMeta',
                            covers='cover',
                            videoMeta='videoMeta',
                            hashtags='hashtag', axis = 1)

# Remove the original columns containing the dictionaries
df_explode = df_explode.drop(['authorMeta','musicMeta','covers','videoMeta','hashtags'], axis = 1)
# %%
df_2 = df_orginal[['diggCount','shareCount','commentCount','playCount']]
min_max_scalar = preprocessing.MinMaxScaler()
X_minmax = min_max_scalar.fit_transform(df_2)
df_3 =pd.DataFrame(X_minmax, columns=['diggCount','shareCount','commentCount','playCount'])
# %%
# build the popularity measure
popularity = (df_3['diggCount'] + df_3['shareCount'] + df_3['commentCount'] + df_3['playCount'])/4
df_3['popularity'] = popularity
df_3['id'] = df_orginal['id']
# %% plot the figure
df_3['mentions'] = df_explode['mentions']
df_3['verified'] = df_explode['authorMeta.verified']
df_3['duration'] = df_explode['videoMeta.duration']
df_fig1 = df_3
# %%
#change mentiosn into true and false
df_fig1['mentions'] = df_fig1['mentions'].astype(bool)
# %%
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
# %%
# start with a square Figure
fig = plt.figure(figsize=(15, 15))
grid = plt.GridSpec(5, 5, hspace=1, wspace=1)
main_ax = fig.add_subplot(grid[:-1, 1:])
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

c1 = '#FE2C55'
c2 = '#25F4EE'
titlefont = {'fontname':'Cheltenham', 'fontsize':'25'}
subtitlefont = {'fontname':'Franklin Gothic Book', 'fontsize':'14'}
annofont = {'fontname':'Franklin Gothic Book', 'fontsize':'12'}
axisfont = {'fontname':'Cheltenham','fontsize':'12','weight':'light'}

# scatter points on the main axes
colorlist = []
markerlist = []
for i in range(len(df_fig1)):
    if df_fig1['mentions'][i] & df_fig1['verified'][i]:
        colorlist.append(c1) #if the videos have mentions, color = red
        markerlist.append('*') #if users are verified, marker = *
    elif df_fig1['verified'][i]:
        colorlist.append(c2) #without mentions, color = blue
        markerlist.append('*')
    elif df_fig1['mentions'][i]:
        colorlist.append(c1)
        markerlist.append('o') #not verified, marker = o
    else:
        colorlist.append(c2)
        markerlist.append('o')

for i in range(len(markerlist)):
    main_ax.scatter(df_fig1['duration'][i], df_fig1['popularity'][i], marker=markerlist[i], 
        color = colorlist[i], alpha=0.8, s=200, zorder=2)
#remove spines
main_ax.spines['top'].set_visible(False)
main_ax.spines['right'].set_visible(False)
main_ax.spines['left'].set_visible(False)
main_ax.spines['bottom'].set_visible(False)

legend_elements = [Line2D([0], [0], marker = '*',color='k', label='Verified User',
                            markerfacecolor='w', markersize=10),
                   Line2D([0], [0], marker='o', color='k', label='Unverified User',
                          markerfacecolor='w', markersize=10),
                   Patch(facecolor='#FE2C55', edgecolor='w',
                         label='Mentioned'),
                   Patch(facecolor='#25F4EE', edgecolor='w',
                         label='Not Mentioned')]
main_ax.legend(handles=legend_elements, loc = 'upper right',frameon=False)
#add title and labels
pos1=np.linspace(1,6,6)
main_ax.set_title('The shorter the better?', **titlefont, pad = 60)
main_ax.set_xlabel('Duration',**axisfont) 
main_ax.set_ylabel('Popularity',**axisfont)
main_ax.tick_params(axis='both',labelsize=8, colors = 'dimgrey')
main_ax.annotate('Popularity=(Diggs+Comments+Views+Shares)/4',xy=(1, 1.05), color = 'k', **subtitlefont, annotation_clip=False)
main_ax.annotate('This chart illustrates the number of videos and related popularity on specific durations', xy=(1,1.1),color='k',**subtitlefont, annotation_clip=False)
main_ax.annotate('User: Billie Eilish\nSoundtrack: Original Sound\nHashtag: #TimeWarpScan',xy=(11, 0.95), color = 'dimgrey', **annofont)
main_ax.annotate('Diggs:39.3M\nComments:622.4K\nShares:288.8K\nViews:0.25B',xy=(11,0.87), color = 'dimgrey', **annofont)
main_ax.grid(zorder=1, ls = '--')


# draw the distribution of the histogram
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))
def hist(x,alphanum):
    cm = plt.cm.get_cmap('hsv')
    bin_heights, bins, patches = x_hist.hist(x,color=c1,bins=30,alpha=alphanum)
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
    maxi = np.abs(bin_centers).max()
    norm = plt.Normalize(-maxi,maxi)

    for c, p in zip(bin_centers, patches):
        plt.setp(p, 'facecolor', cm(norm(c)))
    x_interval_for_fit = np.linspace(bins[0], bins[-1], 10000)
    x_hist.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), color = 'violet')

hist(df_fig1['duration'],0.8)

x_hist.invert_yaxis()
x_hist.spines['top'].set_visible(False)
x_hist.spines['right'].set_visible(False)
x_hist.spines['left'].set_visible(False)
x_hist.spines['bottom'].set_visible(False)
x_hist.tick_params(axis='x', bottom=False,  top=False, labelbottom=False)
x_hist.tick_params(axis='y', labelsize=8, colors = 'dimgrey')
pos = [0,125,250,375,500]
x_hist.set_yticklabels(pos)
x_hist.set_ylabel('Counts',**axisfont)



plt.show()
# %% CHART 3: SENTIMENT

#%% open json file
file = open('trending.json',encoding='utf8')
#parse json
tk = json.load(file)
# Select only the list with the video data
trending_videos_list = tk['collector']
# Example of a video object
print(json.dumps(trending_videos_list[15], indent=4, sort_keys=True))
# %%
# Create a DataFrame of the data
df_orginal = pd.DataFrame(trending_videos_list)

# Let's expand the hashtag cell containing lists to multiple rows
df_explode = df_orginal.explode('hashtags').explode('mentions')
# %%
def object_to_columns(dfRow, **kwargs):
    '''Function to expand cells containing dictionaries, to columns'''
    for column, prefix in kwargs.items():
        if isinstance(dfRow[column], dict):
            for key, value in dfRow[column].items():
                columnName = '{}.{}'.format(prefix, key)
                dfRow[columnName] = value
    return dfRow

# Expand certain cells containing dictionaries to columns
df_explode = df_orginal.apply(object_to_columns, 
                            authorMeta='authorMeta',  
                            musicMeta='musicMeta',
                            covers='cover',
                            videoMeta='videoMeta',
                            hashtags='hashtag', axis = 1)

# Remove the original columns containing the dictionaries
df_explode = df_explode.drop(['authorMeta','musicMeta','covers','videoMeta','hashtags'], axis = 1)

# %% graph 3 -- text sentiment analysis-----------
#import textblob library
from textblob import TextBlob

# %%
df_text = df_orginal[['text','hashtags']]
df_text_explode = df_text.explode('hashtags')

df_text_explode = df_text_explode.apply(object_to_columns, 
                                hashtags='hashtag', axis = 1)

df_4 = df_text_explode[['hashtag.name','text']].copy().dropna()
# %%
# Add column with default value
df_4['count'] = 1

# Count all hashtags, group and replace the count column value with the sum
df_5 = df_4.groupby(["hashtag.name"])["count"].count().reset_index()

# Sort by most popular hashtags and keep the top 15
df_5 = df_5.sort_values(by='count', ascending=False)[:30]

df_5

#category list
##fyp:fyp,for you, foryoupage, fy, voorjou, fypシ, xyzbca, recommendations, voorjoupagina, 
# trending, viral, 
##fitness: fitness, workout, gym, powerlifting, bobybuilding,deadlift,fittok,powertok, bodybuilder
#filter: horedearrasar, transition
#anime: anime, animeedit
#sports: f1
#music: duet
#animal: horse
#country: dutch
#funny: funny

# %%
# %%
df_6 = df_4
df_6['hashtag_cat'] = 0
df_6.loc[df_6['hashtag.name'].isin(['fyp','for you', 'foryoupage','fy','voorjou','fypシ','xyzbca',
        'recommendations','voorjoupagina','trending','viral','trend', 'viraal']),'hashtag_cat'] = 'fyp'
df_6.loc[df_6['hashtag.name'].isin(['fitness', 'workout', 'gym', 'powerlifting', 'bobybuilding',
        'deadlift','fittok','powertok', 'bodybuilder','gymshark', 'training','squat']),'hashtag_cat'] = 'fitness'
df_6.loc[df_6['hashtag.name'].isin(['horedearrasar', 'transition','stitich']),'hashtag_cat'] = 'filter'
df_6.loc[df_6['hashtag.name'].isin(['anime', 'animeedit','weeb', 'naruto', 'otaku']),'hashtag_cat'] = 'anime'
df_6.loc[df_6['hashtag.name'].isin(['f1','formula1', 'wwe', 'equestrian']),'hashtag_cat'] = 'sports'
df_6.loc[df_6['hashtag.name'].isin(['duet','dance', 'billieeilish']),'hashtag_cat'] = 'music'
df_6.loc[df_6['hashtag.name'].isin(['horse']),'hashtag_cat'] = 'animal'
df_6.loc[df_6['hashtag.name'].isin(['dutch','Nederland']),'hashtag_cat'] = 'country'
df_6.loc[df_6['hashtag.name'].isin(['funny']),'hashtag_cat'] = 'funny'

df_6

# %%
df_thg = df_6.loc[df_6["hashtag_cat"]!= 0, :]
df_thg = df_thg.drop(['hashtag.name'], axis = 1)
df_thg = df_thg.groupby(['text','hashtag_cat']).size().reset_index()

sentimentmark = []
for i in range(len(df_thg)):
    x = df_thg['text'][i].replace('#','')
    text = TextBlob(str(x))
    sentimentmark.append(text.sentiment)

df_sentiment = pd.DataFrame(sentimentmark)
df_thg = df_thg.merge(df_sentiment, how='inner', left_index=True, right_index=True)
df_thg.drop(['text',0,'subjectivity'],axis = 1, inplace=True)
df_thg
#%% scale-up the polarity value for visualisation 
df_thg['polarity'] = 50 * df_thg['polarity']

#%%
# Arrange the sequence of hashtag categories based on the number of videos they contain 
df_thg.loc[(df_thg['hashtag_cat'] == "fyp"), 'hashtag_cat'] = 'a.fyp'
df_thg.loc[(df_thg['hashtag_cat'] == "fitness"), 'hashtag_cat'] = 'b.fitness'
df_thg.loc[(df_thg['hashtag_cat'] == "music"), 'hashtag_cat'] = 'c.music'
df_thg.loc[(df_thg['hashtag_cat'] == "anime"), 'hashtag_cat'] = 'd.anime'
df_thg.loc[(df_thg['hashtag_cat'] == "sports"), 'hashtag_cat'] = 'e.sports'
df_thg.loc[(df_thg['hashtag_cat'] == "filter"), 'hashtag_cat'] = 'f.filter'
df_thg.loc[(df_thg['hashtag_cat'] == "funny"), 'hashtag_cat'] = 'i.funny'
df_thg.loc[(df_thg['hashtag_cat'] == "country"), 'hashtag_cat'] = 'g.country'
df_thg.loc[(df_thg['hashtag_cat'] == "animal"), 'hashtag_cat'] = 'h.animal'
df_thg = df_thg.sort_values(['hashtag_cat','polarity'], ascending=[True, False])
# df_thg.drop(df_thg.loc[df_thg['polarity']==0].index, inplace=True)
df_thg.reset_index(drop=True,inplace=True) 

df_thg.head(20)

#%% Create list for the number of videos in a hashtag category and polarity 
df_hcount = df_thg.groupby('hashtag_cat').count().reset_index()
list_pol = list(df_hcount['polarity'])
list_hash = list(df_hcount['hashtag_cat'])
#%%
# Build a dataset
df_dataset = pd.DataFrame({
    'value': df_thg['polarity'],
    'group': df_thg['hashtag_cat']
})
#%%
df_dataset.head(20)

#%%
# Color and fonts
c1 = '#FE2C55'
c2 = '#25F4EE'
c3 = 'grey'
titlefont2 = {'fontname':'Cheltenham', 'fontsize':'16'}
subtitlefont = {'fontname':'Franklin Gothic', 'fontsize':'14'}
annofont = {'fontname':'Franklin Gothic', 'fontsize':'11.5'}
axisfont = {'fontname':'Cheltenham','fontsize':'10','weight':'light'}

colorlist2 = []
edgecolor2 = []
for i in range(len(df_dataset)):
    if df_dataset['value'][i] > 0:
        colorlist2.append(c1) # positive polarity, color = red
        edgecolor2.append(c1)
    elif df_dataset['value'][i] < 0:
        colorlist2.append(c2) # negative and neutral polarity, color = blue
        edgecolor2.append(c2)
    elif df_dataset['value'][i] == 0:
        colorlist2.append(c3) # negative and neutral polarity, color = blue
        edgecolor2.append(c3)

#%%    
ANGLES = np.linspace(0, 2 * np.pi, len(df_dataset), endpoint=False)
VALUES = df_dataset["value"].values
GROUP = df_dataset["group"].values

# Create gaps between bars 
# Add 25 empty bars 
PAD = 25
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)
# Obtaining the right indexes is now a little more complicated
offset = 0
IDXS = []
GROUPS_SIZE = list_pol

for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# Location of the first bar
OFFSET = np.pi/-1
# Layout
fig, ax3 = plt.subplots(figsize=(10, 30), subplot_kw={"projection": "polar"})

ax3.set_theta_offset(OFFSET)
ax3.set_ylim(-100, 100)
ax3.set_frame_on(False)
ax3.xaxis.grid(False)
ax3.yaxis.grid(False)
ax3.set_xticks([])
ax3.set_yticks([])

#legends
legend_elements = [Line2D([0], [0], marker='_', color='black', label='Neutral',
                          markerfacecolor='b', markersize=10),
                   Patch(facecolor='#FE2C55', edgecolor='#FE2C55',
                         label='Optimistic videos'),
                   Patch(facecolor='#25F4EE', edgecolor='#25F4EE',
                         label='Pessimistic videos')]
ax3.legend(handles=legend_elements, loc = 'lower right',title_fontsize=13,frameon=False)


#add title and labels 
ax3.annotate('  Do Trending\n TikTok Videos\n       Spread \n     Positivity?',xy=(220.7, -50),**titlefont2,)
ax3.set_title('This chart illustrates the sentiment polarity score of the videos within each hashtag category',loc='center',**annofont)
# add the scale of polarity to indicate polarity of the videos within each hashtag category
ax3.text(80, -51, '-1', **annofont, color='grey' )
ax3.text(80.05, 49, '1', **annofont, color='grey' )
ax3.text (80.05, 0, '0', **annofont, color='grey')

#add a vertical line to indicate the the scale
ax3.arrow(x=80.12, y=-51, dx = 0, dy = 100, 
          length_includes_head=True,
          head_width=0.05, head_length=5,
          fc ='grey', ec ='grey')

# Add bar
ax3.bar(
    ANGLES[IDXS], VALUES, width=WIDTH, color=colorlist2,
    edgecolor = edgecolor2, linewidth = 1
)

# category list
list_u = ['Fyp (534) \n#fyp #foryou #foryoupage #fy \n#voorjou #xyzbca #recommendations \n#voorjoupaigna #trending #viral\n #trend #viraal',
'Fitness (138) \n#fitness #workout #gym \n#powerlifting #bodybuilding \n#deadlift #fittok #powertok \n#bodybuilder #gymshark \n#training #squat',
'Music (68) \n #duet #dance \n#billieeilish',
'Anime\n (65) \n#anime \n#animeedit \n#weeb #naruto \n#otaku',
'Sports\n (51) \n #f1 \n#formula1 \n#wwe \n#equestrian',
'Filter\n (43) \n #horedearrasar \n #transition #stitich',
'Country\n (35) \n#ducth \n#Nederland',
'Animal\n (31) \n#horse', 
'Funny\n (23)\n#funny']

offset = 0 
for group, size in zip(list_u, GROUPS_SIZE):
    x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=988)
    # Add text to indicate group
    ax3.text(
        np.mean(x1), 80, group,fontname='Franklin Gothic', fontsize=11, 
        ha="center", va="center", color='grey'
    )
    x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=988)
    
    ax3.plot(x2, [50] * 988, color="grey", lw=0.6)
    
    ax3.plot(x2, [-50] * 988, color="grey", lw=0.6)

    offset += size + PAD


# add the scale of polarity to indicate polarity of the videos within each hashtag category
ax3.text(80, -51, '-1', **annofont, color='grey' )

plt.savefig("tiktok_sentiment.png",
            transparent=True,
            pad_inches=0,
            dpi=1200)

plt.show()

# %%

