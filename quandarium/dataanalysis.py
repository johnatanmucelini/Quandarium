"""This module present several functions to plot data."""

import sys
import seaborn as sns
import numpy as np

from matplotlib import rc, rcParams                         # For laxtex in ploting
from matplotlib.ticker import FormatStrFormatter  # For tick labels
import matplotlib
import matplotlib.pylab as plt
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from sklearn.utils import resample
from aux import checkmissingkeys
from aux import to_list
from aux import tonparray

rc('text', usetex=False)
#rc('text.latex',
#   preamble=r'\usepackage{mhchem} \usepackage{amsmath} \usepackage{amsfonts} \
#              \usepackage{mathtools} \usepackage[T1]{fontenc} \
#              \usepackage{siunitx}')


def comp_spearman(data_u, data_v):
    """Compute the spearmanr correlation index.
    Suggestion: use tonparray to converta data (pd.series, non-flatten
    numpy.array) to a flatten numpy.array and drop nans."""
    return spearmanr(data_u, data_v)[0]


def comp_kendall(data_u, data_v):
    """Compute the kendalltau correlation index.
    Suggestion: use tonparray to converta data (pd.series, non-flatten
    numpy.array) to a flatten numpy.array and drop nans."""
    return kendalltau(data_u, data_v)[0]


def comp_pearson(data_u, data_v):
    """Compute the pearsonr correlation index.
    Suggestion: use tonparray to converta data (pd.series, non-flatten
    numpy.array) to a flatten numpy.array and drop nans."""
    return pearsonr(data_u, data_v)[0]


def bstalt(data_x, data_y, corr_method=comp_spearman, alpha=0.05,
           nresamp=2000, hist=''):
    """This function bootstrap the Spearaman rank correlation.
    The bootstraped sample configurations that present all elements equal are
    not considered, because the correlations can\'t be calculated.

    Parameters
    ----------
    data_x, datay : numpy arrays (n,) shaped.
                    Paired data to analyse.
    corr_method : a function (default = comp_spearman).
                  A function that takes two sets of points (x,y in np arrays)
                  and return its correlation.

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
    >>> bootstraping_spearmancorr(data_x, data_y)
    True
    """

    rs = corr_method(data_x, data_y)  # original data correlation
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
        boostrapted_corr = corr_method(resampled_data_x, resampled_data_y)
        # storing the correlacao
        data[interation] = boostrapted_corr
        interation += 1
    # Sorting data
    data.sort()

    index_lower_confidence = int(round(nresamp*alpha/2.))
    index_upper_confidence = int(round(nresamp-nresamp*alpha/2.))
    confidence_data = data[index_lower_confidence:index_upper_confidence]
    confidence_interval = [confidence_data[0], confidence_data[-1]]
    # H0 is rejected if 0 is within the confidence interval:
    reject_null = np.all(np.sign(confidence_data) == np.sign(rs))

    if hist != '':
        plt.close('all')
        plt.ylabel('Frequancy')
        plt.xlabel('Correlation Coefficient')
        plt.xlim(-1.02, 1.02)
        bins = 201
        ranges = (-1., 1.)
        plt.hist(confidence_data, bins=bins, range=ranges,
                 label="Bootstraped Data in the CI")
        plt.hist(data, bins=bins, histtype='step', range=ranges,
                 label="Bootstraped Data")
        plt.plot([rs, rs],
                 [0, np.histogram(data, bins=bins, range=ranges)[0].max()],
                 label="Sample Correlation")
        plt.plot([0, 0],
                 [0, np.histogram(data, bins=bins, range=ranges)[0].max()],
                 label="Correlation = 0")
        plt.legend()

        # ploting or saving to a file:
        if hist == 'plot':
            plt.show()
        else:
            plt.savefig(hist + ".png")

    return reject_null, confidence_interval


def bstnull(data_x, data_y, corr_method=comp_spearman, alpha=0.05, nresamp=2000, hist=''):
    """This function bootstrap the data correlation under the null hypothesis.
    The bootstraped sample configurations that present all elements equal are
    not considered, because the correlations can\'t be calculated.

    Parameters
    ----------
    data_x, datay : numpy arrays (n,) shaped.
                    Paired data to analyse.
    corr_method : a function (default = comp_spearman).
                  A function that takes two sets of points (x,y in np arrays)
                  and return its correlation.
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
    pval: flot.
          The p-value. Keep in mind that to take accurate p-value the n
          should be large.

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

    rs = corr_method(data_x, data_y)  # original data correlation
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
        bootstraped_corr = corr_method(resampled_data_x, resampled_data_y)
        data[interation] = bootstraped_corr  # storing the correlation
        interation += 1
    # Sorting data
    data.sort()

    index_lower_confidence = int(round(nresamp*alpha/2.))
    index_upper_confidence = int(round(nresamp-nresamp*alpha/2.))
    confidence_data = data[index_lower_confidence:index_upper_confidence]

    # H0 is accepted (True) if rs is within the confidence interval:
    if rs >= 0:
        null = np.any(confidence_data > rs)
    else:
        null = np.any(confidence_data < rs)

    # p-value
    pvalue = sum(abs(data) > abs(rs))/len(data)

    # If requested a plot of the boostraped correlations will be created
    if hist != '':
        plt.close('all')
        plt.ylabel('Frequancy')
        plt.xlabel('Correlation Coefficient')
        plt.xlim(-1.02, 1.02)
        bins = 201
        ranges = (-1., 1.)
        # Ploting the data within the confidence alpha:
        plt.hist(confidence_data, bins=bins, range=ranges,
                 label="Bootstraped Data in the CI")
        # ploting all the data:
        plt.hist(data, bins=bins, histtype='step', range=ranges,
                 label="Bootstraped Data")
        plt.plot([rs, rs],
                 [0, np.histogram(data, bins=bins, range=ranges)[0].max()],
                 label="Sample Correlation")
        plt.plot([0, 0],
                 [0, np.histogram(data, bins=bins, range=ranges)[0].max()],
                 label="Correlation = 0")
        plt.legend()

        # saving or plotting the distribution
        if hist == 'plot':
            plt.show()
        else:
            plt.savefig(hist + ".png")

    return null, pvalue


def scatter_colorbar(pd_df, mainprop, features, colsplitfeature, cols,
                     celsplitfeature, cels,
                     show='',
                     label={'x': '', 'y': '', 'title': '', 'ticklabelssprecision':''},
                     cbcomp=comp_spearman, cb_info={'min': -1., 'max': 1.,
                                                    'label': ''},
                     bootstrap_info={'n': 0, 'alpha': 0.25},
                     supress_plot=False,
                     figure_name='figure.png',
                     uselatex=False,
                     scalefont=1.):
    """This function plot a scatterplot of correlations between properties and
    a target property.

    Parameters:
    -----------
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

    labels: a dict ({'x': '', 'y': '', 'title': '', ticklabelssprecision=''})
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

    show: str ['corr','pval', 'test', 'testred', 'ang', 'confint'].
          Define what information will be shown in the scatterplot, if
          bootstrap were employed.
          If 'corr', the correlation value will be printed.
          If 'test', the information whether the correlation pass in the
              bootstrap hypothesis tests or not will be presented.
          If 'testred', the following information will be printed: * if the
              correlation passed in both tests, + if the correlaiton passed in
              one test, and nothing if the correlation fail for both tests.
          If 'pval', the p-value of the correlation bootstrap under null
             hypothesis test will be shown.
          If 'confint', the confidence interval for the correlations will be
             show.
          If 'ang', the angular value of the linear regression will be show.
          If '', nothing will be show.

    figure_name: str, (figure.png).
                 The name of the figure.

    uselatex: bollean, (False)
              If True, the figure text will copiled with latex.

    scalefont: float, (1.)
               Scale the fontsize by this factor.

    Return:
    -------
    A dictionary with the folloyings keys:
        'corrs': np.array of floats with three dimentions.
                 The correlations calculated with cbcomp function.

        If the plot were calculated:
        'fig': pyplot figure.
               The figure.

        'axis': pyplat axis.
                The axis.

        'angular_parameter': np.array of floats with three dimentions.
                             The angular parameters of the linear model
                             fitting the data.

        If bootstrap were employed:
        'alt_test', 'null_test': np.array of booleans with three dimentions.
                                 The result of the hypothesis test, H1 and
                                 H0, respectively.

        'null_test_pval': np.array of floats with three dimentions.
                          The p-values.

        'alt_test_confimax', 'alt_test_confimin': np.array of booleans with
                                                  three dimentions.
                                                  Confidence maximum and
                                                  minimun.
    """

    print("Initializing scatter plot")

    # If were requested, latex will be employed to build the figure.
    #if uselatex:
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\usepackage[version=4]{mhchem} \usepackage{amsmath}'
                                           r'\usepackage{amsfonts} \usepackage{mathtools}'
                                           r'\usepackage[T1]{fontenc} \boldmath']
    rcParams['axes.titleweight'] = 'bold'

    # Features:
    if not isinstance(features, dict):
        print("Error: features should be a dictionary!")
        sys.exit(1)
    checkmissingkeys(list(features.keys()), pd_df.columns.to_list(), "the "
                     "pandas.DataFrame does not present the following "
                     "features")

    # Cells:
    # if there only one plot per cell, a fake dimension will be crated
    if celsplitfeature is None:
        celsplitfeature = 'fake_celsplitfeature'
        pd_df[celsplitfeature] = np.ones(len(pd_df))
        cels = {1: ''}
    if colsplitfeature is None:
        colsplitfeature = 'fake_colsplitfeature'
        pd_df[colsplitfeature] = np.ones(len(pd_df))
        cols = {1: ''}
    grouped = pd_df.groupby([colsplitfeature, celsplitfeature])
    depth = len(cels)
    height = len(features)
    width = len(cols)
    print('depth:', depth, 'height:', height, 'width:', width)

    # Calcalationg the property to be ploted in colors
    corrs_plot = np.zeros([depth, height, width])
    test_apply = np.zeros([depth, height, width], dtype=bool)
    for findex, feature in enumerate(list(features.keys())):
        for colindex, colvals in enumerate(list(cols.keys())):
            for celindex, celvals in enumerate(list(cels.keys())):
                group = grouped.get_group((colvals, celvals))
                datax, datay = tonparray(group[mainprop], group[feature])
                if (len(datax) > 1) and (not np.all(datax == datax[0])) and (not np.all(datay == datay[0])):
                    test_apply[celindex, findex, colindex] = True
                    corrs_plot[celindex, findex, colindex] = cbcomp(datax, datay)
                else:
                    corrs_plot[celindex, findex, colindex] = 0 
    corrs_plot = np.nan_to_num(corrs_plot)

    # Correlation Bootstrap
    if bootstrap_info['n']:  # bootstraping the correlations
        print('Bootstrap analysis')
        null_test = np.zeros([depth, height, width], dtype=bool)
        null_test_pval = np.zeros([depth, height, width], dtype=float)
        alt_test = np.zeros([depth, height, width], dtype=bool)
        alt_test_confimax = np.zeros([depth, height, width], dtype=float)
        alt_test_confimin = np.zeros([depth, height, width], dtype=float)
        for findex, feature in enumerate(list(features.keys())):
            for colindex, colvals in enumerate(list(cols.keys())):
                for celindex, celvals in enumerate(list(cels.keys())):
                    group = grouped.get_group((colvals, celvals))
                    datax, datay = tonparray(group[mainprop], group[feature])
                    if test_apply[celindex, findex, colindex]:
                        # null hypothesisi
                        test, pval = bstnull(datay, datax, corr_method=cbcomp,
                                             nresamp=bootstrap_info['n'],
                                             alpha=bootstrap_info['alpha'])
                        null_test[celindex, findex, colindex] = test
                        null_test_pval[celindex, findex, colindex] = pval
                        # alternative hypothesis
                        test, confi = bstalt(datay, datax, corr_method=cbcomp,
                                             nresamp=bootstrap_info['n'],
                                             alpha=bootstrap_info['alpha'])
                        alt_test[celindex, findex, colindex] = test
                        alt_test_confimax[celindex, findex, colindex] = confi[1]
                        alt_test_confimin[celindex, findex, colindex] = confi[0]
            print("completed: ", findex + 1, ' of ', len(features))


    # printing latex table
    if False: #TODO correct this if statement
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        s=' & '
        sup=np.array(np.round(alt_test_confimax, 2), str)
        inf=np.array(np.round(alt_test_confimin, 2), str)
        r=np.vectorize(lambda t1,t2: t1+','+t2)(sup,inf)
        if r.shape[0] > 1:
            firstrow = s + s
            for colindex, colvals in enumerate(list(cols.keys())):
                firstrow += cols[colvals] + s
            firstrow += ' \\\\'
            print(firstrow)
            for indf, feature in enumerate(list(features.keys())):
                for celindex, celvals in enumerate(list(cels.keys())):
                    rowtotalsup = features[feature] + s 
                    rowtotalinf =                     s 
                    rowtotalsup += cels[celvals] + s  + 'sup' + s
                    rowtotalinf +=                 s  + 'inf' + s
                    for colindex, colvals in enumerate(list(cols.keys())):
                        rowtotalsup += sup[celindex, indf, colindex] + s 
                        rowtotalinf += inf[celindex, indf, colindex] + s
                    print(rowtotalsup + ' \\\\')
                    print(rowtotalinf + ' \\\\')
        else :
            firstrow = s
            for colindex, colvals in enumerate(list(cols.keys())):
                firstrow += cols[colvals] + s + s
            firstrow += ' \\\\'
            print(firstrow)
            for indf, feature in enumerate(list(features.keys())):
                rowsup = features[feature] + s + 'sup' + s
                rowinf =                     s + 'inf' + s
                for colindex, colvals in enumerate(list(cols.keys())):
                    celtotalsup = sup[0, indf, colindex]
                    celtotalinf = inf[0, indf, colindex]
                    rowsup += celtotalsup + s 
                    rowinf += celtotalinf + s
                print(rowsup + ' \\\\')
                print(rowinf + ' \\\\')
    if True: #TODO correct this if statement...
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        s=' & '
        r = np.array(np.round(corrs_plot, 2), str)
        corr = show
        if corrs_plot.shape[0] > 1:
            firstrow = "Features" + s + s
            for colindex, colvals in enumerate(list(cols.keys())):
                firstrow += cols[colvals] + s
            firstrow += ' \\\\'
            print(firstrow)
            for indf, feature in enumerate(list(features.keys())):
                for celindex, celvals in enumerate(list(cels.keys())):
                    rowtotal = features[feature] + s + cels[celvals] + s + corr + s
                    for colindex, colvals in enumerate(list(cols.keys())):
                        rowtotal += r[celindex, indf, colindex] + s
                    print(rowtotal + ' \\\\')
        else :
            firstrow = "Features" + s 
            for colindex, colvals in enumerate(list(cols.keys())):
                firstrow += cols[colvals] + s 
            firstrow += ' \\\\'
            print(firstrow)
            for indf, feature in enumerate(list(features.keys())):
                row = features[feature] + s + corr + s
                for colindex, colvals in enumerate(list(cols.keys())):
                    celtotal = r[0, indf, colindex]
                    row += celtotal + s
                print(row + ' \\\\')



    # If requested, the plot is supressed now
    if supress_plot:
        if bootstrap_info['n']:
            return {'corrs': corrs_plot,
                    'alt_test': alt_test,
                    'alt_test_confimax': alt_test_confimax,
                    'alt_test_confimin': alt_test_confimin,
                    'null_test': null_test,
                    'null_test_pval': null_test_pval,
                    'test_apply' : test_apply
                    }
        return {'corrs': corrs_plot,
                'test_apply' : test_apply}

    # creating empth plot!
    plt.close('all')
    figwidth = int((width*1.)*2)
    figheight = int((height*1.)*2)
    fig, axis = plt.subplots(nrows=height, ncols=width, sharex='col',
                             sharey='row', figsize=(figwidth, figheight))

    # if width or height == 1 (means 1 column, or 1 row) axis has only one
    # dimension, which make some problems, so it must be reshaped
    if height == 1 or width == 1:
        axis = axis.reshape(height, width)

    # Creatin the colormap and mapping the correlation values in the colors
    cmap = matplotlib.cm.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=cb_info['min'],
                                            vmax=cb_info['max'])
    colors = [cmap(normalize(value)) for value in corrs_plot]
    colors = np.array(colors)

    # label sizes:
    axis_title_font_size = 30*scalefont
    axis_label_font_size = 25*scalefont
    tick_label_font_size = 25*scalefont
    anotation_font_size = 20
    marker_size = 50

    # Symbols/markers, lines,
    slines = ['-', '--', ':', '-.']
    scolors = ['y', 'g', 'm', 'c', 'b', 'r']
    smarker = ['o', 's', 'D', '^', '*', 'o', 's', 'x', 'D', '+', '^', 'v', '>']

    # Adding the scatterplos and linear models
    angular_parameter = np.zeros([depth, height, width])
    for indf, feature in enumerate(list(features.keys())):
        for colindex, colvals in enumerate(list(cols.keys())):
            for celindex, celvals in enumerate(list(cels.keys())):
                group = grouped.get_group((colvals, celvals))
                # if ((not all(group[feature] == 0.0))
                #        and (not all(np.isnan(group[feature])))):
                datax, datay = tonparray(group[mainprop], group[feature])
                if datax.tolist():
                    if test_apply[celindex, indf, colindex]:
                        # Linear Regresion
                        degreee = 1
                        parameters = np.polyfit(datax, datay, degreee)
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
                                                              colindex],
                                                 alpha=0.8)
                    if cels[celvals]:
                        axis[indf, colindex].legend()

                axis[indf, colindex].xaxis.set_tick_params(direction='in',
                                                           length=5, width=0.9)
                axis[indf, colindex].yaxis.set_tick_params(direction='in',
                                                           length=5, width=0.9)

            # Ajuste do alinhamento dos labels, quantidade de casa deciamais,
            axis[0, colindex].xaxis.set_label_position("top")
            axis[0, colindex].set_xlabel(cols[colvals], va='center',
                                         ha='center', labelpad=40,
                                         size=axis_label_font_size)
            axis[indf, 0].set_ylabel(features[feature], va='center',
                                     ha='center', labelpad=40,
                                     size=axis_label_font_size,
                                     rotation=60)
            for tikslabel in axis[indf, 0].yaxis.get_ticklabels():
                tikslabel.set_fontsize(tick_label_font_size)
            for tikslabel in axis[-1, colindex].xaxis.get_ticklabels():
                tikslabel.set_fontsize(tick_label_font_size)
                tikslabel.set_rotation(60)

            y = label['ticklabelssprecision'][1][indf]
            x = label['ticklabelssprecision'][0]
            print(list(axis[indf, colindex].xaxis.get_ticklabels()))
            axis[indf, colindex].xaxis.set_major_formatter(FormatStrFormatter('%0.'+str(x)+'f'))        
            axis[indf, colindex].yaxis.set_major_formatter(FormatStrFormatter('%0.'+str(y)+'f'))
            print(list(axis[indf, colindex].xaxis.get_ticklabels()))
            axis[indf, colindex].xaxis.set_major_locator(plt.MaxNLocator(3))
            axis[indf, colindex].yaxis.set_major_locator(plt.MaxNLocator(3))

            print(list(axis[indf, colindex].xaxis.get_ticklabels()))

            #axis[indf, colindex].xaxis.set_ticklabels(axis[indf, colindex].xaxis.get_ticklabels(), {'fontweight':'bold'})
            #axis[indf, colindex].yaxis.set_ticklabels(axis[indf, colindex].yaxis.get_ticklabels(), {'fontweight':'bold'})
            #axis[indf, colindex].xaxis.set_major_formatter(width='bold')
            #axis[indf, colindex].yaxis.set_major_formatter(width='bold')


                


    # Colorbar, pra aprensetar as corres das correlacoes
    cax, _ = matplotlib.colorbar.make_axes(axis[0, 0], orientation='vertical',
                                           shrink=80., ancor=(2., 2.),
                                           pancor=False)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    cax.set_position([0.93, 0.1, 0.04, 0.8])
    cax.set_aspect(40)  # boxY/boxX
    cbar.ax.tick_params(labelsize=tick_label_font_size, labelrotation=90)

    plt.subplots_adjust(left=0.125,
                        right=0.92,
                        bottom=0.1,
                        top=0.9,
                        wspace=0.0,
                        hspace=0.0)

    # Defining what will be ploted
    if show == 'test':  # The result of the test
        truefalse = {True: 'T',
                     False: 'F'}
        binfo_plot = np.vectorize(lambda x, y: truefalse[x] + ',' +
                                  truefalse[y])(null_test, alt_test)
    if show == 'testred':
        def auxfunc(x, y):
            if not x and y:
                return 'x'
            if x and not y:
                return 'o'
            return '+'
        binfo_plot = np.vectorize(auxfunc)(null_test, alt_test)
    if show == 'confint':  # The confidence intervals
        binfo_plot = np.vectorize(lambda x, y: str(round(x, 2)) + ',' +
                                  str(round(y, 2)))(alt_test_confimax,
                                                    alt_test_confimin)
    if show == 'pval':  # the p-value
        binfo_plot = np.array(np.round(null_test_pval, 5), dtype=str)
    if show == 'ang':  # the angle of the linear model
        binfo_plot = np.array(np.round(angular_parameter, 2), dtype=str)

    if bootstrap_info['n'] and show in ['pval', 'test', 'ang', 'confint']:
        for findex, feature in enumerate(features):
            for colindex, colvals in enumerate(list(cols.keys())):
                for celindex, celvals in enumerate(list(cels.keys())):
                    if not test_apply[celindex, findex, colindex]:
                        binfo_plot[celindex, findex, colindex] = ''
        print(binfo_plot)
        for indf, feature in enumerate(features):
            for colindex in range(width):
                for celindex in range(depth):
                    if test_apply[celindex, indf, colindex]:
                        bbox = dict(facecolor=scolors[celindex], alpha=0.1)
                        ypos = 0.155 + (depth - celindex - 1)*0.2
                        axis[indf, colindex].text(0.06, ypos,
                                                  binfo_plot[celindex, indf,
                                                             colindex],
                                                  fontsize=anotation_font_size,
                                                  transform=axis[
                                                      indf, colindex
                                                      ].transAxes,
                                                  bbox=bbox)

    if bootstrap_info['n'] and show in ['testred']:
        for findex, feature in enumerate(features):
            for colindex, colvals in enumerate(list(cols.keys())):
                for celindex, celvals in enumerate(list(cels.keys())):
                    if not test_apply[celindex, findex, colindex]:
                        binfo_plot[celindex, findex, colindex] = ' '
        print(binfo_plot)
        for indf, feature in enumerate(features):
            for colindex in range(width):
                text = ''
                for celindex in range(depth):
                    text += binfo_plot[celindex, indf, colindex]
                    if celindex < depth-1:
                        text += ','
                if np.any(test_apply[:, indf, colindex]):
                    bbox = dict(facecolor=scolors[0], alpha=0.01)
                    ypos = 0.155 + (depth - celindex - 1)*0.2
                    axis[indf, colindex].text(0.06, ypos, text,
                                              fontsize=anotation_font_size,
                                              transform=axis[
                                                  indf, colindex].transAxes,
                                              bbox=bbox)

    # Adicionando os principais captions da figura.
    fig.text(0.01, 0.524, label['y'], ha='center', rotation='vertical',
             size=axis_title_font_size)
    fig.text(0.5, 0.95, label['title'], ha='center', size=axis_title_font_size)
    fig.text(0.5, 0.01, label['x'], ha='center', size=axis_title_font_size)
    cbar.set_label(cb_info['label'], size=axis_title_font_size)

    print(list(axis[0, 0].yaxis.get_ticklabels()))
 
    # Salvando a figura para um arquivo
    plt.savefig(figure_name, dpi=300)

    if bootstrap_info['n']:
        return {'fig': fig,
                'axis': axis,
                'corrs': corrs_plot,
                'alt_test': alt_test,
                'alt_test_confimax': alt_test_confimax,
                'alt_test_confimin': alt_test_confimin,
                'null_test': null_test,
                'null_test_pval': null_test_pval,
                'angular_parameter': angular_parameter,
                'test_apply' : test_apply
                }
    return {'fig': fig,
            'axis': axis,
            'corrs': corrs_plot,
            'angular_parameter': angular_parameter,
            'test_apply' : test_apply
            }


def histbag(figname, bag, grupbybag):
    """ It create a histogram of the values in a bag
    Parameter:
    ----------
    figname: str.
             The name of the figure.

    bag: some structured data object.
         The data of the bag.

    grupbybag: the classes to groupby the values of the bag

    Return:
    -------
    figure: seaborn figure object
            The figure."""

    print('Initializing histbag.')

    bag = to_list(bag)
    grupbybag = to_list(grupbybag)
    data = []
    for i, _ in enumerate(bag):
        data += to_list(bag[i])
    data = np.array(data)

    splitfeaturedata = []
    for i in range(len(bag)):
        splitfeaturedata += to_list(grupbybag[i])
    splitfeaturedata = np.array(splitfeaturedata)

    split_vals = np.unique(splitfeaturedata)

    plt.close('all')
    for val in split_vals:
        dataval = data[splitfeaturedata == val]
        g = sns.distplot(dataval, hist=False, kde=True,
                         kde_kws={'shade': True, 'linewidth': 3},
                         label=val)
    g.get_figure().savefig(figname)

    return g


def boxplot_info(x, p=False):
    size = len(x)

    if size < 3 or np.isnan(np.sum(x)) :
        return [0, 0, 0, 0, 0, 0]

    low_qua, median, up_qua = np.percentile(x, [25,50,75], interpolation='linear')
    IQR = up_qua - low_qua
    up_ext_m = up_qua + 1.5*IQR
    low_ext_m =  low_qua - 1.5*IQR
    
    x = np.sort(x)
    up_ext = x[x<=up_ext_m][-1]
    low_ext = x[x>=low_ext_m][0]

    n_out = np.sum(x>up_ext) + np.sum(x<low_ext)

    if p:
        print("up_ext (max) {:.4f} ({:4f})".format(up_ext, up_ext_m))
        print("up_qua  {:.4f}".format(up_qua))
        print("median  {:.4f}".format(median))
        print("low_qua {:.4f}".format(low_qua))
        print("low_ext (max) {:.4f} ({:4f})".format(low_ext, low_ext_m))
        print("n_out", n_out)

    return [up_ext,up_qua,median,low_qua,low_ext,size,n_out]



def boxplot(pd_df, features, colsplitfeature, cols, celsplitfeature, cels, 
            label={'x': '', 'y': '', 'title': '', 'ticklabelssprecision':''},
            order=None,
            supress_plot=False,
            figure_name='figure.png',
            uselatex=False,
            scalefont=1.):
    """This function plot a matrix of boxplot of correlations for several properties.

    Parameters:
    -----------
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

    labels: a dict ({'x': '', 'y': '', 'title': '', ticklabelssprecision=''})
            The x, y, and the title labels.


    figure_name: str, (figure.png).
                 The name of the figure.

    uselatex: bollean, (False)
              If True, the figure text will copiled with latex.

    scalefont: float, (1.)
               Scale the fontsize by this factor.

    Return:
    -------
    XXX
    A dictionary with the folloyings keys:
        'corrs': np.array of floats with three dimentions.
                 The correlations calculated with cbcomp function.

        If the plot were calculated:
        'fig': pyplot figure.
               The figure.

        'axis': pyplat axis.
                The axis.

        'angular_parameter': np.array of floats with three dimentions.
                             The angular parameters of the linear model
                             fitting the data.

        If bootstrap were employed:
        'alt_test', 'null_test': np.array of booleans with three dimentions.
                                 The result of the hypothesis test, H1 and
                                 H0, respectively.

        'null_test_pval': np.array of floats with three dimentions.
                          The p-values.

        'alt_test_confimax', 'alt_test_confimin': np.array of booleans with
                                                  three dimentions.
                                                  Confidence maximum and
                                                  minimun.
    """

    print("Initializing scatter plot")

    # If were requested, latex will be employed to build the figure.
    #if uselatex:
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\usepackage[version=4]{mhchem} \usepackage{amsmath}'
                                           r'\usepackage{amsfonts} \usepackage{mathtools}'
                                           r'\usepackage[T1]{fontenc} \boldmath']
    rcParams['axes.titleweight'] = 'bold'

    # Features:
    if not isinstance(features, dict):
        print("Error: features should be a dictionary!")
        sys.exit(1)
    checkmissingkeys(list(features.keys()), pd_df.columns.to_list(), "the "
                     "pandas.DataFrame does not present the following "
                     "features")

    # Cells:
    # if there only one plot per cell, a fake dimension will be crated
    if celsplitfeature is None:
        celsplitfeature = 'fake_celsplitfeature'
        pd_df[celsplitfeature] = np.ones(len(pd_df))
        cels = {1: ''}
    if colsplitfeature is None:
        colsplitfeature = 'fake_colsplitfeature'
        pd_df[colsplitfeature] = np.ones(len(pd_df))
        cols = {1: ''}
    grouped = pd_df.groupby([colsplitfeature, celsplitfeature])
    depth = len(cels)
    height = len(features)
    width = len(cols)
    print('depth:', depth, 'height:', height, 'width:', width)

    # creating empth plot!    
    plt.close('all')
    figwidth = int((width*1.)*2)
    figheight = int((height*1.)*2)
    fig, axis = plt.subplots(nrows=height, ncols=width, sharex='col',
                             sharey='row', figsize=(figwidth, figheight))

    # label sizes:
    axis_title_font_size = 30*scalefont
    axis_label_font_size = 25*scalefont
    tick_label_font_size = 25*scalefont
    anotation_font_size = 20
    marker_size = 50

    # Symbols/markers, lines,
    slines = ['-', '--', ':', '-.']
    scolors = ['y', 'g', 'm', 'c', 'b', 'r']
    smarker = ['o', 's', 'D', '^', '*', 'o', 's', 'x', 'D', '+', '^', 'v', '>']

    # Adding the scatterplos and linear models
    size = len(pd_df)
    nouts = np.zeros((height, width, depth))
    ndata = np.zeros((height, width, depth))
    #print(nouts)
    for indf, feature in enumerate(list(features.keys())):
        for colindex, colvals in enumerate(list(cols.keys())):
            if len(cels) > 1:
                # plot
                boxplot = sns.boxplot(x=celsplitfeature, y=feature, data=pd_df[ pd_df[colsplitfeature] == colvals ], orient='v', order=order, ax=axis[indf, colindex])
                # info
                for celindex, celvals in enumerate(list(cels.keys())):
                    aux = np.logical_and( np.array(pd_df[colsplitfeature].to_numpy()) == colvals, np.array(pd_df[celsplitfeature].to_numpy()) == celvals )
                    aux2 = pd_df[aux]
                    data = np.array(aux2[feature].to_numpy())
                    output = boxplot_info(data)
                    ndata[indf,colindex,celindex] = output[-2] 
                    nouts[indf,colindex,celindex] = output[-1] 
            else:
                # plot
                boxplot = sns.boxplot(x=feature, data=pd_df[ pd_df[colsplitfeature] == colvals ], orient='v', order=order, ax=axis[indf, colindex])
                # info
                data = np.array(pd_df[pd_df[colsplitfeature] == colvals][feature].to_numpy() )
                output = boxplot_info(data)
                ndata[indf,colindex,0] = output[-2]
                nouts[indf,colindex,0] = output[-1]                
            # removind individual plots labels
            boxplot.set_xlabel('')
            boxplot.set_ylabel('')

    #print
    # per feature
    for indf, feature in enumerate(list(features.keys())):
        print("{}: {:.3f}%".format(features[feature], 100*np.sum(nouts[indf,:,:])/np.sum(ndata[indf,:,:])))
    # per column
    for colindex, colvals in enumerate(list(cols.keys())):
        print("{}: {:.3f}%".format(cols[colvals], 100*np.sum(nouts[:,colindex,:])/np.sum(ndata[:,colindex,:])))
    # per cell
    if len(cels) > 1:
        for celindex, celvals in enumerate(list(cels.keys())):
            print("{}: {:.3f}%".format(cels[celvals], 100*np.sum(nouts[:,:,celindex])/np.sum(ndata[:,:,celindex])))
    # total
    print('Total: {:.3f}%'.format(100*np.sum(nouts)/np.sum(ndata)))


    # adding labels
    for indf, feature in enumerate(list(features.keys())):
        for colindex, colvals in enumerate(list(cols.keys())):
            if True:
                axis[indf, colindex].xaxis.set_tick_params(direction='in', length=5, width=0.9)
                axis[indf, colindex].yaxis.set_tick_params(direction='in', length=5, width=0.9)

                # Ajuste do alinhamento dos labels, quantidade de casa deciamais,
                axis[0, colindex].xaxis.set_label_position("top")
                axis[0, colindex].set_xlabel(cols[colvals], va='center', ha='center', labelpad=40, size=axis_label_font_size)
                axis[indf, 0].set_ylabel(features[feature], va='center', ha='center', labelpad=40, size=axis_label_font_size, rotation=60)
                for tikslabel in axis[indf, 0].yaxis.get_ticklabels():
                    tikslabel.set_fontsize(tick_label_font_size)
                for tikslabel in axis[-1, colindex].xaxis.get_ticklabels():
                    tikslabel.set_fontsize(tick_label_font_size)

                    #tikslabel.set_rotation(60)
                #y = label['ticklabelssprecision'][1][indf]
                #x = label['ticklabelssprecision'][0]
                #print(list(axis[indf, colindex].xaxis.get_ticklabels()))
                #axis[indf, colindex].xaxis.set_major_formatter(FormatStrFormatter('%0.'+str(x)+'f'))            
                #axis[indf, colindex].yaxis.set_major_formatter(FormatStrFormatter('%0.'+str(y)+'f'))
                #print(list(axis[indf, colindex].xaxis.get_ticklabels()))
                #axis[indf, colindex].xaxis.set_major_locator(plt.MaxNLocator(3))
                #axis[indf, colindex].yaxis.set_major_locator(plt.MaxNLocator(3))        
                #axis[indf, colindex].xaxis.set_ticklabels(axis[indf, colindex].xaxis.get_ticklabels(), {'fontweight':'bold'})
                #axis[indf, colindex].yaxis.set_ticklabels(axis[indf, colindex].yaxis.get_ticklabels(), {'fontweight':'bold'})
                #axis[indf, colindex].xaxis.set_major_formatter(width='bold')
                #axis[indf, colindex].yaxis.set_major_formatter(width='bold')

    plt.subplots_adjust(left=0.125,
                        right=0.92,
                        bottom=0.1,
                        top=0.9,
                        wspace=0.0,
                        hspace=0.0)

    # Adicionando os principais captions da figura.
    fig.text(0.01, 0.524, label['y'], ha='center', rotation='vertical',
             size=axis_title_font_size)
    fig.text(0.5, 0.95, label['title'], ha='center', size=axis_title_font_size)
    fig.text(0.5, 0.01, label['x'], ha='center', size=axis_title_font_size)

    # Salvando a figura para um arquivo
    plt.savefig(figure_name, dpi=300)

    return
