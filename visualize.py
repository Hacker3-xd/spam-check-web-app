import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def plot_class_distribution(df):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="label", data=df)
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    return ax.get_figure()


def plot_wordcloud(text, title="Word Cloud"):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    return plt.gcf()
