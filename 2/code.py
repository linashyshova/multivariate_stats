import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal


def main():
    data = pd.read_excel("data/songlist.xlsx")
    L = 5  # Number of categories
    n = len(data)

    plot_histograms(data)

    w, mu, Sigma, loglik = estimate(data, L)
    print(f"Mu \n {mu}")

    data = assign_categories(L, Sigma, data, mu, n, w)
    plot_scatterplot(data)

    playlist = create_playlist("Welcome To The Jungle", "Guns 'N Roses", data)
    print("Recommended playlist for starting song Welcome To The Jungle by Guns 'N Roses:")
    print(playlist)


def create_playlist(song_name, artist_name, data):
    song = data[(data["name"] == song_name) & (data["artist"] == artist_name)]
    song_category = song["category"].values[0]
    return data[data["category"] == song_category].nlargest(20, "category_prob")[["name", "artist", "category_prob"]]


def plot_scatterplot(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(
        data=data,
        x="energy",
        y="loudness",
        ax=axes[0]
    )
    axes[0].set_title("No category")
    sns.scatterplot(
        data=data,
        x="energy",
        y="loudness",
        hue="category",
        palette="tab10",
        ax=axes[1]
    )
    axes[1].set_title("Colored by category")
    plt.tight_layout()
    plt.show()


def assign_categories(L, Sigma, data, mu, n, w):
    x = data[["danceability", "tempo", "energy", "loudness"]].to_numpy()

    # Set intial values to 0
    pdf = np.zeros((n, L))

    # For each category calculate PDF for a given song
    for l in range(L):
        pdf[:, l] = multivariate_normal(mu[l], Sigma[l]).pdf(x)

    # Calculated weighted PDF per category
    weighted = w * pdf

    # By dividing weighed PDF by sum of all weighted PDF's we get probability
    # of each song belonging to each category
    pi = weighted / weighted.sum(axis=1, keepdims=True)

    # Song gets assigned to a category with the highest probability
    data["category"] = pi.argmax(axis=1)

    # Highest probability is also stored as a separate column
    data["category_prob"] = pi.max(axis=1)

    return data


def estimate(data, L):
    x = data[["danceability", "tempo", "energy", "loudness"]].to_numpy()
    n, d = x.shape

    # Start params
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    mu = x[np.random.choice(n, L, replace=False)]
    Sigma = np.array([np.cov(x.T) for _ in range(L)])
    loglik = 0
    max_attempts = 100
    # Cycle repeats until it reaches max attempts or until difference between old and new likelihoods becomes very small
    for j in range(max_attempts):
        ## E-step
        # Initiate with zeros
        pdf = np.zeros((n, L))

        # For each category we calculate PDF for each song using initial (or updated) parameters
        for l in range(L):
            pdf[:, l] = multivariate_normal(mu[l], Sigma[l]).pdf(x)

        # Calculated weighted PDF per category
        weighted = w * pdf

        # By dividing weighed PDF by sum of all weighted PDF's we get probability
        # of each song belonging to each category
        pi = weighted / weighted.sum(axis=1, keepdims=True)

        # Sum all pi values
        pi_sum = pi.sum(axis=0)

        ## M-step
        # Calculate parameter estimates by using closed form solution
        w_new = pi_sum / n
        mu_new = (pi.T @ x) / pi_sum[:, None]
        Sigma_new = np.zeros((L, d, d))
        for l in range(L):
            diff = x - mu_new[l]
            Sigma_new[l] = (pi[:, l][:, None] * diff).T @ diff / pi_sum[l]

        # Calculate loglikelihood
        loglik_new = np.sum(np.log(weighted.sum(axis=1)))

        # print(f"{j} W {w_new}, \nmu {mu_new},\n Sigma \n{Sigma_new},\n loglik {loglik_new}")

        # Check if difference between new and old loglikelihood is not small enough to stop
        if abs(loglik - loglik_new) < 0.01:
            return w_new, mu_new, Sigma_new, loglik_new

        # Update estimated parameters for the next cycle
        mu = mu_new
        Sigma = Sigma_new
        w = w_new
        loglik = loglik_new

    return w, mu, Sigma, loglik


def plot_histograms(data):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].hist(data["energy"], bins=100)
    axs[0, 0].set_title("Energy")
    axs[0, 1].hist(data["danceability"], bins=100)
    axs[0, 1].set_title("Danceability")
    axs[1, 0].hist(data["tempo"], bins=100)
    axs[1, 0].set_title("Tempo")
    axs[1, 1].hist(data["loudness"], bins=100)
    axs[1, 1].set_title("Loudness")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
