"""This module present several functions to plot data."""

import logging
import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import rc                         # For laxtex in ploting
from matplotlib.ticker import FormatStrFormatter  # For tick labels
import matplotlib
import matplotlib.pylab as plt
from matplotlib.legend import Legend
import matplotlib.lines as mlines
from scipy.stats import spearmanr
from scipy import stats as scipystats
from sklearn.utils import resample
from npeet import entropy_estimators
from quandarium.analy.aux import checkmissingkeys
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{mhchem} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{mathtools} \usepackage[T1]{fontenc} ')

logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.WARNING)
logging.info('The logging level is INFO')


def data2rownp(data):
    """It convert a data (pd.series, np.array, and list) to a flat
    np.array. It it is none of then, an error will occur"""
    if isinstance(data, pd.core.series.Series):
        rownp = data.values.flatten()
    elif isinstance(data, np.ndarray):
        rownp = data.flatten()
    elif isinstance(data, list):
        rownp = np.array(data).flatten()
    elif isinstance(data, pd.DataFrame):
        print('WARNING: you can not transform a pandas.DataFrame in a '
              'numpy row array. Tip: check if there are two columns with '
              'the name {} in your pandas.DataFrame.)'.format(
                 data.columns[0]))
        logging.error('WARNING: you can not transform a pandas.DataFrame '
                      'in a numpy row array. Tip: check if there are two '
                      'columns with the name {} in your '
                      'pandas.DataFrame.)'.format(data.columns[0]))
        sys.exit(1)
    return rownp


def johnatan_polyfit(data_1, data_2, degreee):
    """This function perform a convetional polyfit but first, it remove the
    pairs (data_1[i],data_2[i]) if one of then is a np.nan."""
    real_data = np.logical_and(np.isnan(data_1) == False,
                               np.isnan(data_2) == False)
    return np.polyfit(data_1[real_data], data_2[real_data], degreee)


def comp_mannwhitneyu(x, y):
    """It compute the mannwhitneyu index"""
    result = scipystats.mannwhitneyu(x, y, use_continuity=True,
                                     alternative='two-sided')[0]
    return result


def bts(boolean):
    """This function return a 'T' string if the bollean is True and a 'F'
    string otherwise"""
    if boolean:
        string = "T"
    if not boolean:
        string = "F"
    return string


def rs_tstudent(speramanr, number_of_samples, alpha=0.05):
    """This calculate the tstudent statistics. In particular, the function
    indicate is the data null hypothesis could be rejected under the confidence
    level alpha, and the p-value. See also the bootstrap methods.

    Parameter
    ---------
    """
    # most of the cases -1 < r < 1
    if abs(speramanr) < 1:
        tval = abs(speramanr * np.sqrt(number_of_samples - 2.)
                   / np.sqrt(1.-speramanr**2))

        # tabele values of t - critic alpha for a 2-sided test
        tab_t = np.array([0.694, 0.870, 1.079, 1.350, 1.771, 2.160, 2.650,
                          3.012, 3.372, 3.852, 4.221])
        tab_alphac = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99,
                               0.995, 0.998, 0.999])  # two sided test

        # 3 values of (t,alpha) will be fitted with a quadratic function:
        # a + b*x + cx**2
        initial_index = abs(tab_t - tval).argmin() - 1
        final_index = abs(tab_t - tval).argmin() + 2
        if initial_index <= 0:
            initial_index, final_index = 0, 3
        if final_index >= 10:
            initial_index, final_index = 8, 11
        tab_t_fit = tab_t[initial_index:final_index]
        tab_alphac_fit = tab_alphac[initial_index:final_index]
        paramc, paramb, parama = np.polyfit(tab_t_fit, tab_alphac_fit, 2)
        alphac = parama + paramb*tval + paramc*tval**2

        reject_null = alpha < alphac
        pvalue = scipystats.t.sf(np.abs(tval), 15-2)*2

    # just to avoid some problems when r = 1
    if abs(speramanr) == 1:
        tval = np.inf
        alphac = 1.
        pvalue = 0.
        reject_null = True

    return reject_null, pvalue


def comp_mi(data_u, data_v):
    """Compute the mutual information between two sets of data"""
    data_u = data2rownp(data_u)
    data_v = data2rownp(data_v)
    real = (not np.isnan(data_u)) * (not np.isnan(data_v))
    u_real = data_u[real]
    v_real = data_v[real]
    gval = entropy_estimators.mi(u_real.reshape([-1, 1]).tolist(),
                                 v_real.reshape([-1, 1]).tolist())
    # if False :
    #    bins = 5
    #    _ , P_u , P_v =np.histogram2d( U_nonan , V_nonan , bins )
    #    g = normalized_mutual_info_score( P_u , P_v )
    # if False :
    #    if ( len(U_nonan) > 0 and len(V_nonan) > 0 ) :
    #        bins = max(int(len(U_nonan)/4), 7 )
    #        P_uv = np.histogram2d( U_nonan , V_nonan , bins )[0]
    #        P_uv = P_uv / np.sum(P_uv)
    #        P_U = np.sum( P_uv ,axis=1 )
    #        P_V = np.sum( P_uv ,axis=0 )
    #        P_U_time_P_V = P_U.reshape([1,-1]) * P_V.reshape([-1,1])
    #        g = np.sum( P_uv * np.ma.log2( P_uv / (P_U_time_P_V) ).filled(0) )
    #    else :
    #        g = 0
    #        print( 'Warning comput_mi: Empth array affter drop nans out.' )
    return gval


def comp_entroy(data_u, data_v):
    """Compute the entropy in the first set of data."""
    data_u = data2rownp(data_u)
    data_v = data2rownp(data_v)
    real = (not np.isnan(data_u)) * (not np.isnan(data_v))
    u_real = data_u[real]
    #V_nonan = V[nonan]
    entropy_u = entropy_estimators.entropy(u_real.reshape([-1, 1]).tolist())
    # if ( len(U_nonan) > 0 and len(V_nonan) > 0 ) :
    #    bins = max(int(len(U_nonan)/4), 7 )
    #    P_uv = np.histogram2d( U_nonan , V_nonan , bins )[0]
    #    P_uv = P_uv / np.sum(P_uv)
    #    P_U = np.sum( P_uv ,axis=1 )
    #    P_V_square_shape = np.array( [P_U ] * len(P_U) )
    #    H_U = np.sum( P_uv * np.ma.log2( P_U / P_uv ).filled(0) )
    # else :
    #    H_U = 0
    #    print( 'Warning comput_H: Empth array affter drop nans out.' )
    return entropy_u


def bstaltrs(data_x, data_y, alpha=0.05, nresamp=2000, hist=''):
    """This function bootstrap the Spearaman rank correlation.

    Parameters
    ----------
    data_x, datay : numpy arrays (n,) shaped.
                    Paired data to analyse.
    alpha : float. (optional, default=0.05)
            The confidence limit.
    nresamp : intiger. (optional, default=2000)
              The quantity of data resamples in the procedure.
    hist : string (optional, default='').
           If hist == 'plot' a figure with data result histogram will be
           displayed on screem, otherwise the same figure will be saved to a
           file named hist + '.png'.


    Return
    ------
    reject_null : boolean.
                  True if the null hypothese could be rejected withing the
                  confidence level.

    Example
    ------
    >>> data_y = np.array([0.29210368, 0.09100691, 0.03445345, 0.1953896 ,
                           0.09828076, 0.06194474, 0.07301951, 0.05899114,
                           0.05012644, 0.03095898, 0.10257979, 0.08892738,
                           0.05457695, 0.02178669, 0.0326735 ])
    >>> data_x = np.array([-4.38  , -3.9418, -4.0413, -4.1549, -4.2052,
                           -3.915 , -4.1796, -4.1815, -3.972 , -4.0494,
                           -4.2255, -4.2772, -3.9947, -3.9589, -3.8393])
    >>> bootstraping_spearmancorr(data_x, data_y, hist='')
    True
    """

    data_x = data2rownp(data_x)  # to guarantee that data is a 1D np.array
    data_y = data2rownp(data_y)
    rs = spearmanr(data_x, data_y)[0]  # original data correlation
    data = np.zeros(nresamp)  # the data will be stored in this variable

    possible_data = np.array(range(0, len(data_x)))  # auxiliar variable
    interation = 0  # auxiliar variable
    while interation < nresamp:
        # resampling pairs with replacement:
        resampled_pairs = resample(possible_data)
        resampled_data_x = data_x[resampled_pairs]
        resampled_data_y = data_y[resampled_pairs]
        # to guarantee that the elements are not all equal
        if np.all(resampled_data_x == resampled_data_x[0]):
            continue
        if np.all(resampled_data_y == resampled_data_y[0]):
            continue
        # calculating correlation for the resampled data
        resampled_rs = spearmanr(resampled_data_x, resampled_data_y)[0]
        # storing the correlacao
        data[interation] = resampled_rs
        interation += 1
    # Sorting data
    data.sort()

    index_lower_confidence = int(round(nresamp*alpha/2.))
    index_upper_confidence = int(round(nresamp-nresamp*alpha/2.))
    confidence_data = data[index_lower_confidence:index_upper_confidence]
    # H0 is rejected if 0 is within the confidence interval:
    reject_null = np.all(np.sign(confidence_data) == np.sign(rs))

    if hist != '':
        plt.close('all')
        plt.ylabel('Frequancy')
        plt.xlabel('Spearman Correlation')
        plt.xlim(-1.02, 1.02)
        bins = 201
        ranges = (-1., 1.)
        plt.hist(confidence_data, bins=bins, range=ranges,
                 label="Confidence Data")
        plt.hist(data, bins=bins, histtype='step', range=ranges,
                 label="All Data")
        plt.plot([rs, rs],
                 [0, np.histogram(data, bins=bins, range=ranges)[0].max()],
                 label="Original Data")
        plt.plot([0, 0],
                 [0, np.histogram(data, bins=bins, range=ranges)[0].max()],
                 label="0 Correlation")
        plt.legend()

        # ploting or saving to a file:
        if hist == 'plot':
            plt.show()
        else:
            plt.savefig(hist + ".png")

    return reject_null


def bstnullrs(data_x, data_y, alpha=0.05, nresamp=2000, hist=''):
    """This function bootstrap the data Spearaman rank correlation under the
    null hypothesis.

    Parameters
    ----------
    data_x, datay : numpy arrays (n,) shaped.
                    Paired data to analyse.
    alpha : float. (optional, default=0.05)
            The confidence limit.
    nresamp : intiger. (optional, default=2000)
              The quantity of data resamples in the procedure.
    hist : string (optional, default='').
           If hist == 'plot' a figure with data result histogram will be
           displayed on screem, otherwise the same figure will be saved to a
           file named hist + '.png'.


    Return
    ------
    reject_null : boolean.
                  True if the null hypothese could be rejected withing the
                  confidence level.

    Example
    ------
    >>> data_y = np.array([0.29210368, 0.09100691, 0.03445345, 0.1953896 ,
                           0.09828076, 0.06194474, 0.07301951, 0.05899114,
                           0.05012644, 0.03095898, 0.10257979, 0.08892738,
                           0.05457695, 0.02178669, 0.0326735 ])
    >>> data_x = np.array([-4.38  , -3.9418, -4.0413, -4.1549, -4.2052,
                           -3.915 , -4.1796, -4.1815, -3.972 , -4.0494,
                           -4.2255, -4.2772, -3.9947, -3.9589, -3.8393])
    >>> bootstraping_nullspearmancorr(data_x, data_y, hist='')
    True
    """

    data_x = data2rownp(data_x)
    data_y = data2rownp(data_y)
    nan_data = np.logical_or(np.isnan(data_y), np.isnan(data_x))
    real_data = nan_data == False
    data_x = data_x[real_data]
    data_y = data_y[real_data]
    if len(data_x) == 0:
        return True, 1.0
    if (np.all(data_y == data_y[0]) or np.all(data_x == data_x[0])):
        return True, 1.0

    data_x = data2rownp(data_x)  # to guarantee that data is a 1D np.array
    data_y = data2rownp(data_y)
    rs = spearmanr(data_x, data_y)[0]  # original data correlation
    data = np.zeros(nresamp)  # variable to store the resampled data analysis

    interation = 0  # auxiliar variable
    while interation < nresamp:
        # resampling with replacement:
        resampled_data_x = resample(data_x)
        resampled_data_y = resample(data_y)
        # to guarantee that the elements are not all equal
        if np.all(resampled_data_x == resampled_data_x[0]):
            continue
        if np.all(resampled_data_y == resampled_data_y[0]):
            continue
        # computing the spearmanr to the resampled
        resampled_rs = spearmanr(resampled_data_x, resampled_data_y)[0]
        data[interation] = resampled_rs  # storing the correlation
        interation += 1
    # Sorting data
    data.sort()

    index_lower_confidence = int(round(nresamp*alpha/2.))
    index_upper_confidence = int(round(nresamp-nresamp*alpha/2.))
    confidence_data = data[index_lower_confidence:index_upper_confidence]

    # H0 is rejected if rs is within the confidence interval:
    reject_null = np.logical_or(np.all(rs < confidence_data),
                                np.all(confidence_data < rs))

    # p-value
    pvalue = sum(abs(data) > abs(rs))/len(data)

    if hist != '':
        plt.close('all')
        plt.ylabel('Frequancy')
        plt.xlabel('Spearman Correlation')
        plt.xlim(-1.02, 1.02)
        bins = 201
        ranges = (-1., 1.)
        # Plotando os resultados:
        # Plotando os dados dentro do limite de confiança alpha:
        plt.hist(confidence_data, bins=bins, range=ranges,
                 label="Confidence Data")
        # plotando todos os dados:
        plt.hist(data, bins=bins, histtype='step', range=ranges,
                 label="All Data")
        plt.plot([rs, rs],
                 [0, np.histogram(data, bins=bins, range=ranges)[0].max()],
                 label="Original Data")
        plt.plot([0, 0],
                 [0, np.histogram(data, bins=bins, range=ranges)[0].max()],
                 label="O Correlation")
        plt.legend()

        # saving or plotting the distribution
        if hist == 'plot':
            plt.show()
        else:
            plt.savefig(hist + ".png")

    return reject_null, pvalue

def comp_spearman(data_u, data_v, print=False):
    data_u = data2rownp(data_u)
    data_v = data2rownp(data_v)
    nan_data = np.logical_or(np.isnan(data_u), np.isnan(data_v))
    real_data = nan_data == False
    real_data_u = data_u[real_data]
    real_data_v = data_v[real_data]
    if print:
        print('real_data_u:', real_data_u.tolist())
        print('real_data_v:', real_data_v.tolist())
        print('real_data:', real_data.tolist())
    if len(real_data_u) == 0:
        result = 0.
    elif (all(real_data_u == real_data_u[0]) or all(real_data_v == real_data_v[0])):
        result = 0.
    else:
        result = spearmanr(real_data_u, real_data_v)[0]
    return result


def scatter_colorbar(pd_df, mainprop, features, splitfeature, fdict='',
                     cdict='', x_labels='', y_labels='',
                     cblabel='Spearman Rank Correlation',
                     xmatrixlabel='Relaive Energy (eV)', xmatrixlabelr='$n$ of \ce{Ce_{$n$}Zr_{15-$n$}O_{30}}',
                     ymatrixlabel='Features', cbcomp=comp_spearman,
                     figure_name='figure', cbnorm=(-1., 1.), bootstrap=False,
                     alpha=0.10, nresamp=5000):
    """This function plot a lot of data
    infocb 'spearman' , 'kendall' , 'pearson' , 'mi' , 'entropy'
    mainprop: str.
              This feature will be the horizontal cell axes, shered
              whichin each column.
    features: list of str.
              The selected features will stay in the vertical axis, one feature
              per row in the scatternplot matrix.
    splitfeature: str.
                  Each colum on the scatternplot matrix will contain data for
                  one of the values of the variable to split.
    bootstrap: bollean, optional (default=False).
               If True, two bootstrap analysis will be performed, under de null
               and under the alternative hypothesis. confidence level is alpha,
               and the number of resamples is nresamp

    row_labels
    column_labels
    """

    print("Initializing plot")
    logging.info("Initializing plot")

    # Columns:
    if cdict == '':
        if x_labels == '':
            x_labels = []
            aux = list(np.array(np.unique(pd_df[splitfeature])))
            aux.sort()
            for xval in aux:
                x_labels.append(xval)
    else:
        unique_vals = np.unique(pd_df[splitfeature])
        x_labels = [cdict[val] for val in unique_vals]

    # Features:
    checkmissingkeys(features, pd_df.columns.to_list(), "the "
        "pandas.DataFrame does not present the following features")
    if fdict == '':
        if y_labels == '':
            y_labels = []
            for feature in features:
                y_labels.append(feature.replace('reg_', '').replace('_', '-'))
    else:
        checkmissingkeys(features, fdict, "Missing labels in dict fdict")
        y_labels = [fdict[feature] for feature in features]

    grouped = pd_df.groupby(splitfeature)
    height = len(features)
    width = len(grouped.groups)

    # Calcalationg:
    info_plot = np.zeros([height, width])
    for findex, feature in enumerate(features):
        for gindex, (_, group) in enumerate(grouped):
            # if findex == 32 and gindex == 0 and
            # feature == 'reg_av_exposition_incore':
            info_plot[findex, gindex] = cbcomp(group[feature],
                                               group[mainprop])
    info_plot = np.nan_to_num(info_plot)

    # Correlation Bootstrap
    if bootstrap:
        print('Bootstrap analysis')
        null_test = np.zeros([height, width])
        null_test_pval = np.zeros([height, width])
        alt_test = np.zeros([height, width])
        for findex, feature in enumerate(features):
            for gindex, (_, group) in enumerate(grouped):
                test, pval = bstnullrs(group[feature], group[mainprop],
                                       nresamp=nresamp, alpha=alpha)
                null_test[findex, gindex] = test
                null_test_pval[findex, gindex] = pval
                #alt_test[findex, gindex] = bstaltrs(group[feature],
                #                                    group[mainprop],
                #                                    nresamp=nresamp,
                #                                    alpha=alpha)
            print("completed: ", findex + 1, ' of ', len(features))

    # Iniciando a plotagem!
    plt.close('all')
    figwidth = int((width*1.)*2)
    figheight = int((height*1.)*2)
    fig, axis = plt.subplots(*info_plot.shape, sharex='col', sharey='row',
                             figsize=(figwidth, figheight))

    # crinado labels da figura no eixo horizontal da matrix
    # x_labels = ["Pt"+"{}".format(list(index_labels_dict.keys())[list(
    # index_labels_dict.values()).index(i)])
    # for i in index_labels_dict.values()]
    # Usando o dicionario acima pra obeter os labes dos eixos vertical da matri
    # y_labels = [feature_dic[feature] for feature in features]
    # mapeando os valores de correlacao/mi/entropy com cores
    # cmap = matplotlib.cm.get_cmap('coolwarm')
    # if infocb in ['spearman' , 'kendall' , 'pearson'] :
    # if infocb == 'mi' or infocb == 'entropy' :
    #     cmap = matplotlib.cm.get_cmap('brg')
    #     normalize = matplotlib.colors.Normalize(vmin=min(np.append(
    #         plot_in_colors_42.flatten(), plot_in_colors_13.flatten())),
    #         vmax=max(np.append(plot_in_colors_42.flatten(),
    #         plot_in_colors_13.flatten())))
    # colors42 = [cmap(normalize(value)) for value in plot_in_colors_42]
    cmap = matplotlib.cm.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=cbnorm[0], vmax=cbnorm[1])
    colors = [cmap(normalize(value)) for value in info_plot]

    # label sizes:
    axis_title_font_size = 38
    axis_label_font_size = 30
    tick_label_font_size = 25
    anotation_font_size = 13
    marker_size = 60

    marker = ['*', 'o']
    for indf, feature in enumerate(features):
        for indg, (_, group) in enumerate(grouped):

            if ((not all(group[feature] == 0.0))
                    and (not all(np.isnan(group[feature])))):
                if np.sum(np.array(np.isnan(group[feature].values) == False,
                                   dtype=bool)) > 1:
                    # Linear Regresion
                    fit_fn = np.poly1d(johnatan_polyfit(group[mainprop],
                                                        group[feature], 1))
                    # variavel auxiliar pra nao plotar o linha obtida na
                    # regressao alem dos dados do set (isso pode acontecer para
                    # as variaveisb2 onde nem todos os samples apresentam
                    # dados)
                    aux = np.array([list(group[mainprop]),
                                    fit_fn(group[mainprop])])
                    xfited_values = np.array([aux[:, aux[0].argmin()],
                                              aux[:, aux[0].argmax()]]).T[0]
                    yfited_values = np.array([aux[:, aux[0].argmin()],
                                              aux[:, aux[0].argmax()]]).T[1]
                    # plotando linha obtida com dados da regressao
                    axis[indf, indg].plot(xfited_values, yfited_values,
                                          marker=None, linestyle='-',
                                          color='k')
                # plotando dados da celula
                axis[indf, indg].scatter(group[mainprop], group[feature],
                                         marker=marker[0], s=marker_size,
                                         linestyle='None',
                                         color=colors[indf][indg])

                axis[indf, indg].xaxis.set_tick_params(direction='in',
                                                       length=5, width=0.9)
                axis[indf, indg].yaxis.set_tick_params(direction='in',
                                                       length=5, width=0.9)

            # Ajuste do alinhamento dos labels, quantidade de casa deciamais,
            # tamanho de fonte e etc
            axis[0, indg].xaxis.set_label_position("top")
            axis[0, indg].set_xlabel(x_labels[indg], va='center', ha='center',
                                     labelpad=40, size=axis_label_font_size)
            axis[indf, 0].set_ylabel(y_labels[indf], va='center', ha='center',
                                     labelpad=60, size=axis_label_font_size,
                                     rotation='horizontal')
            axis[indf, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            for tikslabel in axis[indf, 0].yaxis.get_ticklabels():
                tikslabel.set_fontsize(tick_label_font_size)
            axis[-1, indg].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            for tikslabel in axis[-1, indg].xaxis.get_ticklabels():
                tikslabel.set_fontsize(tick_label_font_size)
                tikslabel.set_rotation(60)

    # b=mlines.Line2D([], [], color='grey', marker=marker[1], linestyle='None',
    #                 markersize=15, label='\ce{Pt42TM13}')
    # a=mlines.Line2D([], [], color='grey', marker=marker[0], linestyle='None',
    #                 markersize=15, label='\ce{Pt13TM42')
    # axis[-1, -1].legend(handles=[a, b], loc=((-10.9, -0.8)), ncol=2,
    #                     fontsize=axis_label_font_size, handletextpad=-0.5,
    #                     columnspacing=0.7, frameon=False)
    # leg = Legend(axis[0,0], '-' , ['A', 'B'], loc='lower right',
    #              frameon=False)
    # axis[0,0].add_artist(leg);

    # Caso seja necessario modificar alguma coisa pontualmente, fazer isso aqui
    # axis[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # axis[2,0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # axis[3,0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # axis[4,0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # criando um colorbar, pra aprensetar as corres das correlacoes
    cax, _ = matplotlib.colorbar.make_axes(axis[0, 0], orientation='vertical',
                                           shrink=80., ancor=(2., 2.),
                                           pancor=False)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    cax.set_position([0.905, 0.1, 0.08, 0.8])
    cax.set_aspect(40)
    cbar.ax.tick_params(labelsize=tick_label_font_size, labelrotation=90)

    # Adicionando um segundo colorbar pra ficar sobre o primeiro e entao gera a
    # area cinza no meio do colorbar principal
    # cmap2 = matplotlib.colors.ListedColormap(['#DEDDDC', 'g'])
    # cax2, _ = matplotlib.colorbar.make_axes(axis[0,0],
    #                                         orientation='vertical',
    #                                         shrink=80., ancor=(2., 2.),
    #                                         pancor=False , pad=0.15 )
    # cbar2 = matplotlib.colorbar.ColorbarBase(cax2, cmap=cmap2,
    #     norm=matplotlib.colors.BoundaryNorm([-0.25,0.25 , 0.2500001],
    #     cmap.N),spacing='proportional')
    # cax2.set_position([0.905,0.4,0.08,0.2])
    # cax2.set_aspect( 10 )
    # cbar2.ax.tick_params(labelsize=0 )

    # margins e espacamentos entre as celulas da matrix de scatter plots
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.0,
                        hspace=0.0)

    # Caso queira, aqui adiciona-se as anotações na figura
    for indf, feature in enumerate(features):
        for indg, (_, group) in enumerate(grouped):
            if not all(group[feature] == 0.0):
                if null_test_pval[indf, indg] > 0.005:
                    labelstr = str(round(null_test_pval[indf, indg], 2))
                elif null_test_pval[indf, indg] > 0.:
                    labelstr = str('{:.0E}'.format(null_test_pval[indf, indg]))
                else:
                    labelstr = str('<{:.0E}'.format(1./nresamp))
                axis[indf, indg].text(0.06, 0.155, labelstr,
                                      fontsize=anotation_font_size,
                                      transform=axis[indf, indg].transAxes,
                                      bbox=dict(facecolor='yellow', alpha=0.1))

    # Adicionando os principais captions da figura.
    fig.text(0.04, 0.524, ymatrixlabel, ha='center', rotation='vertical',
             size=axis_title_font_size)
    fig.text(0.5, 0.95, xmatrixlabelr, ha='center', size=axis_title_font_size)
    fig.text(0.5, 0.045, xmatrixlabel, ha='center', size=axis_title_font_size)
    cbar.set_label(cblabel, size=axis_title_font_size)

    # Salvando a figura para um arquivo
    print("Wait...")
    plt.savefig(figure_name + ".png", dpi=300)

    return


def scatter_allvsall(pd_df, regs, splitfeature, axismarks=['ecn', 'dav', 'ori',
                                                           'exposition',
                                                           'qtn']):
    """ERRORS!!!!!!!!!..."""

    rc('text', usetex=False)

    if not os.path.isdir('qdfigures_allvsall'):
        os.mkdir('qdfigures_allvsall')

    sns.pairplot(pd_df, x_vars=regs, y_vars=regs, hue=splitfeature,
                 dropna=True)

    axiss = []
    for axismark in axismarks:
        axiss.append([])
        for name in regs:
            if axismark in name:
                axiss[-1].append(name)
    axiss = [[regs[5:]]]
    for axis, name in zip(axiss, axismarks):
        if axis:
            plt.close('all')
            print(axis)
            figure = sns.pairplot(pd_df, x_vars=axis, y_vars=axis,
                                                      dropna=True)
            print('saving...')
            figure.savefig('pairplot_' + name + '_' + name + '.png')

    for counter_1, (axis_1, name_1) in enumerate(zip(axiss, axismarks)):
        for counter_2, (axis_2, name_2) in enumerate(zip(axiss, axismarks)):
            if (counter_1 > counter_2) and axis_1 and axis_2:
                plt.close('all')
                figure = sns.pairplot(pd_df, x_vars=axis_1, y_vars=axis_2,
                                      hue=splitfeature, dropna=True)
                figure.savefig('pairplot_' + name_1 + '_' + name_2 + '.png')



FEATURES_DICT = {
    'homos1_final': 'HOMO',
    'lumos1_final': 'LUMO',
    'gap_final': '$E_{\\textrm{gap}}$',
    'cnav_total': '$CN_{av}$',
    'dav_total': '$d_{av}$',
    'total_bonds': '$N^{\\circ}$ bonds',
    'sigma': '$\sigma$',
    'rav': '$R_{av}$',
    'cnav_Ce': '$CN_{av}^{\\ce{Ce}}$',
    'cnav_Zr': '$CN_{av}^{\\ce{Zr}}$',
    'cnav_O': '$CN_{av}^{\\ce{O}}$',
    'dav_Ce': '$d_{av}^{\\ce{Ce}}$',
    'dav_Zr': '$d_{av}^{\\ce{Zr}}$',
    'dav_O': '$d_{av}^{\\ce{O}}$',
    'bonds_Ce_O': '$N$ \ce{Ce}-\ce{O}',
    'bonds_Zr_O': '$N$ \ce{Zr}-\ce{O}',
    'bonds_O_O': '$N$ \ce{O}-\ce{O}',
    'TM': '\\ce{TM}',
    'qtn_atoms_Pt': '$N$ \\ce{Pt}',
    'exc_energy': '$E_{\\textrm{exc}}$',
    'bound_energy': '$E_{b}$',
    'qtn_atoms_TM': '$N$ \\ce{TM}',
    'total_energy_final': '$E_{\\textrm{tot}}$',
    'E_fermi': '$E_{fermi}$',
    'cbm_final': '$E_{\\textrm{CBM}}$',
    'vbm_final': '$E_{\\textrm{VBM}}$',
    'mag_moment_final': '$\mu$',
    'dav_Pt': '$d_{av}^{\\ce{Pt}}$',
    'ecn_Pt': '$ECN_{av}^{\\ce{Pt}}$',
    'dav_TM': '$d_{av}^{\\ce{TM}}$',
    'ecn_TM': '$ECN_{av}^{\\ce{TM}}$',
    'bonds_Pt_Pt': '$N$ \ce{Pt}-\ce{Pt}',
    'bonds_TM_TM': '$N$ \ce{TM}-\ce{TM}',
    'bonds_Pt_TM': '$N$ \ce{Pt}-\ce{TM}',
    'surface_atoms': '$N$ Surf',
    'core_atoms': '$N$ Core',
    'Pt_surf': '$N$ \\ce{Pt}$_{surf}$',
    'Pt_core': '$N$ \\ce{Pt}$_{core}$',
    'TM_surf': '$N$ \ce{TM}$_{surf}$' ,
    'TM_core': '$N$ \\ce{TM}$_{core}$',
    'cluster_radius': '$R_{av}$',
    'total_surface_energy': '$E_{Surf}^{tot}$',
    'surface_energy': '$E_{Surf}$',
    'surface_area': '$A_{surf}$',
    'mag_moment_surf': '$\mu_{surf}$',
    'mag_moment_core': '$\mu_{core}$',
    'ecn_surf': '$ECN_{surf}$',
    'ecn_core': '$ECN_{core}$',
    'dav_surf': '$d_{av}^{surf}$',
    'dav_core': '$d_{av}^{core}$',
    'avarage_surface_exposition_surf': '$A_{exp}$',
    'charge_pt': '$Q_{\\ce{Pt}}$',
    'charge_tm': '$Q_{\\ce{TM}}$',
    'reg_total_energy_final': '$E_{\\textrm{tot}}$',
    'reg_gap_final': '$E_{\\textrm{gap}}$',
    'reg_E_fermi': '$E_{fermi}$',
    'reg_cbm_final': '$E_{\\textrm{CBM}}$',
    'reg_vbm_final': '$E_{\\textrm{VBM}}$',
    'reg_mag_moment_final': '$\mu$',
    'reg_exc_energy': '$E_{\\textrm{exc}}$',
    'reg_bound_energy': '$E_{b}$',
    'reg_total_surface_energy': '$E_{Surf}^{tot}$',
    'reg_surface_energy': '$E_{Surf}$',
    'reg_surface_area': '$A_{surf}$',
    'reg_cluster_radius': '$R_{av}$',
    'reg_sigma': '$\sigma$',
    'reg_bonds_Pt_Pt': '$N_{\ce{Pt}-\ce{Pt}}$',
    'reg_bonds_TM_TM': '$N_{\ce{TM}-\ce{TM}}$',
    'reg_bonds_Pt_TM': '$N_{\ce{Pt}-\ce{TM}}$',
    #
    'reg_av_b1_surf_charges': '$Q^{surf}$',
    'reg_av_b1_core_charges': '$Q^{core}$',
    'reg_av_b1_Pt_charges': '$Q^{\\ce{Pt}}$',
    'reg_av_b1_TM_charges': '$Q^{\\ce{TM}}$',
    'reg_av_b2_surf_Pt_charges': '$Q^{surf,\\ce{Pt}}$',
    'reg_av_b2_surf_TM_charges': '$Q^{surf,\\ce{TM}}$',
    'reg_av_b2_core_Pt_charges': '$Q^{core,\\ce{Pt}}$',
    'reg_av_b2_core_TM_charges': '$Q^{core,\\ce{TM}}$',
    'reg_av_b1_all_charges': '$Q^{all}$',
    'reg_av_b1_surf_mag_moments': '$\overline{m}^{surf}$',
    'reg_av_b1_core_mag_moments': '$\overline{m}^{core}$',
    'reg_av_b1_Pt_mag_moments': '$m^{\\ce{Pt}}$',
    'reg_av_b1_TM_mag_moments': '$\overline{m}^{\\ce{TM}}$',
    'reg_av_b2_surf_Pt_mag_moments': '$m^{surf,\\ce{Pt}}$',
    'reg_av_b2_surf_TM_mag_moments': '$m^{surf,\\ce{TM}}$',
    'reg_av_b2_core_Pt_mag_moments': '$m^{core,\\ce{Pt}}$',
    'reg_av_b2_core_TM_mag_moments': '$m^{core,\\ce{TM}}$',
    'reg_av_b1_all_mag_moments': '$m^{all}$',
    'reg_av_b1_surf_dav': '$d_{av}^{surf}$',
    'reg_av_b1_core_dav': '$d_{av}^{core}$',
    'reg_av_b1_Pt_dav': '$d_{av}^{\\ce{Pt}}$',
    'reg_av_b1_TM_dav': '$\overline{d_{av}}^{\\ce{TM}}$',
    'reg_av_b2_surf_Pt_dav': '$d_{av}^{surf,\\ce{Pt}}$',
    'reg_av_b2_surf_TM_dav': '$d_{av}^{surf,\\ce{TM}}$',
    'reg_av_b2_core_Pt_dav': '$d_{av}^{core,\\ce{Pt}}$',
    'reg_av_b2_core_TM_dav': '$d_{av}^{core,\\ce{TM}}$',
    'reg_av_b1_all_dav': '$d_{av}^{all}$',
    'reg_av_b1_surf_ecn': '$ECN^{surf}$',
    'reg_av_b1_core_ecn': '$ECN^{core}$',
    'reg_av_b1_Pt_ecn': '$ECN^{\\ce{Pt}}$',
    'reg_av_b1_TM_ecn': '$\overline{ECN}^{\\ce{TM}}$',
    'reg_av_b2_surf_Pt_ecn': '$ECN^{surf,\\ce{Pt}}$',
    'reg_av_b2_surf_TM_ecn': '$ECN^{surf,\\ce{TM}}$',
    'reg_av_b2_core_Pt_ecn': '$ECN^{core,\\ce{Pt}}$',
    'reg_av_b2_core_TM_ecn': '$ECN^{core,\\ce{TM}}$',
    'reg_av_b1_all_ecn': '$ECN^{all}$',
    #
    'reg_surface_atoms': '$N^{surf}$',
    'reg_core_atoms': '$N^{core}$',
    'reg_Pt_surf': '$N^{\\ce{Pt},surf}$',
    'reg_Pt_core': '$N^{\\ce{Pt},core}$',
    'reg_TM_surf': '$N^{\\ce{TM},surf}$',
    'reg_TM_core': '$N^{\\ce{TM},core}$',
    'reg_qtn_Pt': '$N^{\\ce{Pt}}$',
    'reg_qtn_TM': '$N^{\\ce{TM}}$'
    }


#
#def cost(ri, dij, A, C, baserad):
#   Rij = baserad.reshape([1, -1]) + baserad.reshape([-1, 1])
#   rij = ri.reshape([1, -1]) + ri.reshape([-1, 1])
#   size = len(x)
#   act = (1-1/(1 + 2.7**(10*(dij - rij))))*(np.ones([size,size]) - np.eye(size))
#   B=1-A
#   partA= A*np.sum(((dij - rij)**2)*act)
#   partB= B*np.sum(- act)
#   return A*np.sum(((dij - rij)**2)*act) + B*np.sum(- act) + C*np.sum(((rij - Rij)**2)*act) #(1-1/(1 + 2.7**(10*(dij - x))))*(-3 + 100*(dij - x)**2)
#optimize.minimize(cost, np.min(dij,axis=0), args=(dij,0.010,0.2,baserad), method="L-BFGS-B", bounds=((0.0, 1.7),)*len(positions), tol=1.E-7, options={"maxiter": 50, "disp": False})
