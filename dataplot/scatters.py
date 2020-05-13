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
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy import stats as scipystats
from sklearn.utils import resample
# from npeet import entropy_estimators
from quandarium.analy.aux import checkmissingkeys
rc('text', usetex=True)
rc('text.latex',
   preamble=r'\usepackage{mhchem} \usepackage{amsmath} \usepackage{amsfonts} \
              \usepackage{mathtools} \usepackage[T1]{fontenc}')

logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.WARNING)
logging.info('The logging level is INFO')


def data2rownp(data):
    """
    **summary line** It convert a data to a flat np.array.

    Data: (pd.series, np.array, and list)
          It it is none of then, an error will occur
    """
    #
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


def tonparray(*data, dropnan=True):
    """Converta data (pd series or non-flatten numpy array) to a flatten numpy
    array. Droping nans by default..."""
    if len(data) == 2:
        data1 = data[0]
        data2 = data[1]
        if isinstance(data1, pd.core.series.Series):
            data1 = data1.values.flatten()
        elif isinstance(data1, np.ndarray):
            data1 = data1.flatten()
        elif isinstance(data1, list):
            data1 = np.array(data1).flatten()
        else:
            print('ERROR: the type {} (for data1) is not suported in tonoparray.'.format(
                type(data1)))
            logging.error('ERROR: the type {} (for data1) is not suported in tonoparray.'.format(
                type(data1)))
        if isinstance(data2, pd.core.series.Series):
            data2 = data2.values.flatten()
        elif isinstance(data2, np.ndarray):
            data2 = data2.flatten()
        elif isinstance(data2, list):
            data2 = np.array(data2).flatten()
        else:
            print('ERROR: the type {} (for data2) is not suported in tonoparray.'.format(
                type(data2)))
            logging.error('ERROR: the type {} (for data2) is not suported in tonoparray.'.format(
                type(data2)))
        if dropnan:
            usefulldata = np.logical_and(np.isnan(data1) == False,
                                         np.isnan(data2) == False)
            return data1[usefulldata], data2[usefulldata]
        else:
            return data1, data2

    if len(data) == 1:
        data1 = data[0]
        if isinstance(data1, pd.core.series.Series):
            data1 = data1.values.flatten()
        elif isinstance(data1, np.ndarray):
            data1 = data1.flatten()
        elif isinstance(data1, list):
            data1 = np.array(data1).flatten()
        else:
            print('ERROR: the type {} is not suported in tonoparray.'.format(
                type(data1)))
            logging.error('ERROR: the type {} is not suported in tonoparray.'.format(
                type(data1)))
        if dropnan:
            usefulldata = np.isnan(data1) == False
            return data1[usefulldata]
        else:
            return data1


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


def comp_spearman(data_u, data_v):
    """Compute the spearmanr correlation index"""
    data_u, data_v = tonparray(data_u, data_v)
    if ((len(data_u) < 2) or all(data_u == data_u[0]) or all(data_v == data_v[0])):
        result = 0.
    else:
        result = spearmanr(data_u, data_v)[0]
    return result


def comp_kendall(data_u, data_v):
    """Compute the kendalltau correlation index"""
    data_u, data_v = tonparray(data_u, data_v)
    if ((len(data_u) < 2) or all(data_u == data_u[0]) or all(data_v == data_v[0])):
        result = 0.
    else:
        result = kendalltau(data_u, data_v)[0]
    return result


def comp_pearson(data_u, data_v):
    """Compute the pearsonr correlation index"""
    data_u, data_v = tonparray(data_u, data_v)
    if ((len(data_u) < 2) or all(data_u == data_u[0]) or all(data_v == data_v[0])):
        result = 0.
    else:
        result = pearsonr(data_u, data_v)[0]
    return result


def scatter_colorbar(pd_df, mainprop, features, colsplitfeature, cols,
                     celsplitfeature, cels,
                     show='',
                     label={'x': '', 'y': '', 'title': ''},
                     cbcomp=comp_spearman, cb_info={'min': -1., 'max': 1.,
                                                    'label': ''},
                     bootstrap_info={'n': 0, 'alpha': 0.25},
                     supress_plot=False,
                     figure_name='figure.png'):
    """This function plot a lot of data
    infocb 'spearman' , 'kendall' , 'pearson' , 'mi' , 'entropy'
    mainprop: str.
              This feature will be the horizontal cell axes, shered
              whichin each column.
    features: dict.
              Mapping the features names (keys) and its label (values). The
              order of the features in the figure follows the same order of the
              presented in this dict.
    colsplitfeature: str.
                     The scatternplot matrix will contain data splited per
                     column according to this feature values.
    cols: dict.
          Mapping the colsplitfeature feature values (keys) and its label (dict
          values and column labels). The columns in the figure follows the same
          order of this dict.
    celsplitfeature: str or None.
                     The scatternplot matrix will contain data splited per
                     cell according to this feature values. None wil desable
                     multi plot per scattermatrix cell.
    cels: dict or None.
          If dict, it map the celsplitfeature feature value (keys) and its
          labels (dict values and plot labels). The order of the plots in the
          cells in the figure follows the same order of this dict.
    labels: a dict ({'x': '', 'y': '', 'title': ''})
            The x, y, and the title labels.
    cbcomp: function,
            function that compute the information (correlation/mutual info)
            with the paired data.
    cb_info: dict with three values ({'min'=-1.,'max'=1.,'label'=''}).
             The max and min values for the colorbar values and its label.
    bootstrap_info: a dict with three ({n=0,alpha=0.25,show:test,corrcut=0}).
                    If 'n' value were larger then zero the bootstrap analysis
                    will be performed with (under the null and alternative
                    hypothesis) with number of bootstraped samples equal to 'n'
                    value and a conconfidence level 'alpha'.
                    Moreover, the it will make return pvalues and both
                    hypothese test results.
    show: str.
          Define what information will be shown in the scatterplot.
          If 'test' (and bootstrap_info['n'] > 0), the information which will
              be presented is whether if the correlation pass in the bootstrap
              hypothesis tests.
          If 'p' (and bootstrap_info['n'] > 0), the p-value of the correlation
              bootstrap under null hypothesis test will be shown.
          If 'ang', the angular value of the linear regression will be show.
          If '', nothing will be show.
    figure_name: str, (figure.png).
                 The name of the figure.

    Return:
    -------
    info_plot: np.array of floats with three dimentions.
               The correlations calculated with cbcomp function.

    alt_test, null_test: np.array of booleans (if bootstrap_info['n']>0)
                         the result of the hypothesis test
    null_test_pval: np.array of floats (if bootstrap_info['n']>0)
                    The pvalues
    """

    print("Initializing scatter plot")

    # Features:
    checkmissingkeys(list(features.keys()), pd_df.columns.to_list(), "the "
                     "pandas.DataFrame does not present the following features")

    # Cells:
    # if there only one plot per cell, a fake dimension will be crated
    if celsplitfeature is None:
        celsplitfeature = 'fake_celsplitfeature'
        pd_df[celsplitfeature] = np.ones(len(pd_df))
        cels = {1: ''}
    grouped = pd_df.groupby([colsplitfeature, celsplitfeature])
    depth = len(cels)
    height = len(features)
    width = len(cols)
    print(depth, height, width)

    # Calcalationg the property to be ploted in colors
    info_plot = np.zeros([depth, height, width])
    for findex, feature in enumerate(list(features.keys())):
        for colindex, colvals in enumerate(list(cols.keys())):
            for celindex, celvals in enumerate(list(cels.keys())):
                group = grouped.get_group((colvals, celvals))
                info_plot[celindex, findex, colindex] = cbcomp(
                    group[feature], group[mainprop])
    info_plot = np.nan_to_num(info_plot)

    # Correlation Bootstrap
    if bootstrap_info['n']:  # bootstraping the correlations
        print('Bootstrap analysis')
        null_test = np.zeros([depth, height, width], dtype=bool)
        null_test_pval = np.zeros([depth, height, width], dtype=float)
        alt_test = np.zeros([depth, height, width], dtype=bool)
        for findex, feature in enumerate(features):
            for colindex, colvals in enumerate(list(cols.keys())):
                for celindex, celvals in enumerate(list(cels.keys())):
                    group = grouped.get_group((colvals, celvals))
                    # null hypothesis
                    test, pval = bstnullrs(group[feature], group[mainprop],
                                           nresamp=bootstrap_info['n'],
                                           alpha=bootstrap_info['alpha'])
                    null_test[celindex, findex, colindex] = test
                    null_test_pval[celindex, findex, colindex] = pval
                    # alternative hypothesis
                    alt_test[celindex, findex, colindex] = bstaltrs(
                        group[feature], group[mainprop],
                        nresamp=bootstrap_info['n'],
                        alpha=bootstrap_info['alpha'])
            print("completed: ", findex + 1, ' of ', len(features))
        if show == 'test':  # passed in the tests?
            truefalse = {True: 'T',
                         False: 'F'}
            binfo_plot = np.zeros_like(alt_test, dtype='<U3')
            for findex, feature in enumerate(features):
                for colindex, colvals in enumerate(list(cols.keys())):
                    for celindex, celvals in enumerate(list(cels.keys())):
                        null = truefalse[null_test[celindex, findex, colindex]]
                        alt = truefalse[alt_test[celindex, findex, colindex]]
                        string = null + ',' + alt
                        binfo_plot[celindex, findex, colindex] = string

        if show == 'p':  # pvalue
            binfo_plot = np.array(np.round(null_test_pval, 3), dtype=str)

    if supress_plot:
        return info_plot, alt_test, null_test, null_test_pval

    # Iniciando a plotagem!
    plt.close('all')
    figwidth = int((width*1.)*2)
    figheight = int((height*1.)*2)
    fig, axis = plt.subplots(nrows=height, ncols=width, sharex='col',
                             sharey='row', figsize=(figwidth, figheight))

    # Creatin the colormap and mapping the correlation values in the colors
    cmap = matplotlib.cm.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=cb_info['min'],
                                            vmax=cb_info['max'])
    colors = [cmap(normalize(value)) for value in info_plot]
    colors = np.array(colors)

    # label sizes:
    axis_title_font_size = 30
    axis_label_font_size = 25
    tick_label_font_size = 25
    anotation_font_size = 15
    marker_size = 50

    slines = ['-', '--', ':', '-.']
    scolors = ['y', 'g', 'm', 'c', 'b', 'r']
    smarker = ['o', 's', 'D', '^', '*', 'o', 's', 'x', 'D', '+', '^', 'v', '>']
    angular_parameter = np.zeros([depth, height, width])
    for indf, feature in enumerate(list(features.keys())):
        for colindex, colvals in enumerate(list(cols.keys())):
            for celindex, celvals in enumerate(list(cels.keys())):
                group = grouped.get_group((colvals, celvals))
                # if ((not all(group[feature] == 0.0))
                #        and (not all(np.isnan(group[feature])))):
                datax, datay = tonparray(group[mainprop], group[feature])
                if datax.tolist():
                    if (len(datax) > 1) and (not np.all(datax == datax[0])):
                        # Linear Regresion
                        parameters = johnatan_polyfit(datax, datay, 1)
                        fit_fn = np.poly1d(parameters)
                        angular_parameter[celindex, indf, colindex] = parameters[0]
                        # variavel auxiliar pra nao plotar o linha obtida na
                        # regressao alem dos dados do set (isso pode acontecer
                        # para as variaveisb2 onde nem todos os samples
                        # apresentam dados)
                        yfited_values = np.array(fit_fn(datax))
                        argmin = np.argmin(datax)
                        argmax = np.argmax(datax)
                        trend_x = datax[[argmin, argmax]]
                        trend_y = yfited_values[[argmin, argmax]]
                        # plotando linha obtida com dados da regressao
                        axis[indf, colindex].plot(trend_x, trend_y,
                                                  marker=None,
                                                  linestyle=slines[celindex],
                                                  color='k')
                    # plotando dados da celula
                    axis[indf, colindex].scatter(datax, datay,
                                                 marker=smarker[celindex],
                                                 s=marker_size,
                                                 linestyle='None',
                                                 label=cels[celvals],
                                                 color=colors[celindex, indf,
                                                              colindex])
                    if cels[celvals]:
                        axis[indf, colindex].legend()

                axis[indf, colindex].xaxis.set_tick_params(direction='in',
                                                           length=5, width=0.9)
                axis[indf, colindex].yaxis.set_tick_params(direction='in',
                                                           length=5, width=0.9)

            # Ajuste do alinhamento dos labels, quantidade de casa deciamais,
            # tamanho de fonte e etc
            axis[0, colindex].xaxis.set_label_position("top")
            axis[0, colindex].set_xlabel(cols[colvals], va='center',
                                         ha='center', labelpad=40,
                                         size=axis_label_font_size)
            axis[indf, 0].set_ylabel(features[feature], va='center',
                                     ha='center', labelpad=30,
                                     size=axis_label_font_size,
                                     rotation='vertical')
            # axis[indf, 0].yaxis.set_major_formatter(
            #    FormatStrFormatter('%.1f'))
            for tikslabel in axis[indf, 0].yaxis.get_ticklabels():
                tikslabel.set_fontsize(tick_label_font_size)
            # axis[-1, colindex].xaxis.set_major_formatter(
            #    FormatStrFormatter('%.1f'))
            for tikslabel in axis[-1, colindex].xaxis.get_ticklabels():
                tikslabel.set_fontsize(tick_label_font_size)

                tikslabel.set_rotation(60)

    # Caso seja necessario modificar alguma coisa pontualmente, fazer aqui
    # axis[4,0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # Colorbar, pra aprensetar as corres das correlacoes
    cax, _ = matplotlib.colorbar.make_axes(axis[0, 0], orientation='vertical',
                                           shrink=80., ancor=(2., 2.),
                                           pancor=False)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    cax.set_position([0.905, 0.1, 0.08, 0.8])
    cax.set_aspect(40)
    cbar.ax.tick_params(labelsize=tick_label_font_size, labelrotation=90)

    # Margins e espacamentos entre as celulas da matrix de scatter plots
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.0,
                        hspace=0.0)

    # Adicionando anotações na figura
    if show == 'ang':
        binfo_plot = np.array(np.round(angular_parameter, 2), dtype=str)
    if bootstrap_info['n'] and show == ['p', 'test', 'ang']:

        for indf, feature in enumerate(features):
            for colindex in range(width):
                for celindex in range(depth):
                    if not all(group[feature] == 0.0):
                        bbox = dict(facecolor=scolors[celindex], alpha=0.1)
                        ypos = 0.155 + (depth - celindex - 1)*0.2
                        axis[indf, colindex].text(0.06, ypos,
                                                  binfo_plot[celindex, indf,
                                                             colindex],
                                                  fontsize=anotation_font_size,
                                                  transform=axis[
                                                     indf, colindex].transAxes,
                                                  bbox=bbox)

    # Adicionando os principais captions da figura.
    fig.text(0.04, 0.524, label['y'], ha='center', rotation='vertical',
             size=axis_title_font_size)
    fig.text(0.5, 0.95, label['title'], ha='center', size=axis_title_font_size)
    fig.text(0.5, 0.02, label['x'], ha='center', size=axis_title_font_size)
    cbar.set_label(cb_info['label'], size=axis_title_font_size)

    # Salvando a figura para um arquivo
    print("Wait...")
    plt.savefig(figure_name, dpi=300)

    if bootstrap_info['n']:
        return info_plot, alt_test, null_test, null_test_pval
    else:
        return info_plot


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
