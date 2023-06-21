import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ssp_topsis import SSP_TOPSIS

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights

from pyrepo_mcda.mcda_methods import TOPSIS


def main():

    # Load decision matrix with performance values
    df = pd.read_csv('dataset/dataset_EU_SHARES_2021_population.csv', index_col='Country')
    matrix = df.to_numpy()
    types = np.ones(matrix.shape[1])
    weights = mcda_weights.critic_weighting(matrix)

    names = list(df.index)

    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)

    topsis = TOPSIS(normalization_method=norms.minmax_normalization)
    pref_t = topsis(matrix, weights, types)
    results_pref['TOPSIS'] = pref_t
    rank_t = rank_preferences(pref_t, reverse=True)
    results_rank['TOPSIS'] = rank_t


    toss = SSP_TOPSIS(normalization_method=norms.minmax_normalization)

    # sustainability coefficient from matrix calculated based on standard deviation from normalized matrix
    n_matrix = norms.minmax_normalization(matrix, types)
    s = np.sqrt(np.sum(np.square(np.mean(n_matrix, axis = 0) - n_matrix), axis = 0) / n_matrix.shape[0])

    pref = toss(matrix, weights, types, s_coeff = s)
    results_pref['SSP-TOPSIS std'] = pref
    rank = rank_preferences(pref, reverse = True)
    results_rank['SSP-TOPSIS std'] = rank

    
    # analysis with sustainability coefficient modification
    model = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ]


    #
    # analysis performed for table
    for el, mod in enumerate(model):
        new_s = np.zeros(matrix.shape[1])
        new_s[mod] = s[mod]

        pref = toss(matrix, weights, types, s_coeff = new_s)
        results_pref['SSP-TOPSIS ' + r'$G_{' + str(el + 1) + '}$'] = pref
        rank = rank_preferences(pref, reverse = True)
        results_rank['SSP-TOPSIS ' + r'$G_{' + str(el + 1) + '}$'] = rank

    results_pref = results_pref.rename_axis('Country')
    results_rank = results_rank.rename_axis('Country')
    results_pref.to_csv('./results/df_pref_G' + '.csv')
    results_rank.to_csv('./results/df_rank_G' + '.csv')


    #
    # analysis performed for figures
    sust_coeff = np.arange(0, 1.1, 0.1)

    for el, mod in enumerate(model):
        results_pref = pd.DataFrame(index=names)
        results_rank = pd.DataFrame(index=names)

        for sc in sust_coeff:

            s = np.zeros(matrix.shape[1])
            s[mod] = sc

            pref = toss(matrix, weights, types, s_coeff=s)
            rank = rank_preferences(pref, reverse = True)

            results_pref[str(sc)] = pref
            results_rank[str(sc)] = rank


        results_pref = results_pref.rename_axis('Country')
        results_rank = results_rank.rename_axis('Country')
        results_pref.to_csv('./results/df_pref_sust_G' + str(el + 1) + '.csv')
        results_rank.to_csv('./results/df_rank_sust_G' + str(el + 1) + '.csv')

        # plot results of analysis with sustainabiblity coefficient modification
        ticks = np.arange(1, 30)

        x1 = np.arange(0, len(sust_coeff))

        plt.figure(figsize = (10, 6))
        for i in range(results_rank.shape[0]):
            plt.plot(x1, results_rank.iloc[i, :], '*-', linewidth = 2)
            ax = plt.gca()
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                            fontsize = 12, #style='italic',
                            horizontalalignment='left')

        plt.xlabel("Sustainability coeffcient", fontsize = 12)
        plt.ylabel("Rank", fontsize = 12)
        plt.xticks(x1, np.round(sust_coeff, 2), fontsize = 12)
        plt.yticks(ticks, fontsize = 12)
        plt.xlim(x_min - 0.2, x_max + 1.8)
        plt.gca().invert_yaxis()
        
        plt.grid(True, linestyle = ':')
        if el < 4:
            plt.title(r'$G_{' + str(el + 1) + '}$')
        else:
            plt.title('All criteria')
        plt.tight_layout()
        plt.savefig('./results/rankings_sust_G' + str(el + 1) + '.png')
        plt.savefig('./results/rankings_sust_G' + str(el + 1) + '.pdf')
        plt.show()
    


if __name__ == '__main__':
    main()