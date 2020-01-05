from scipy.signal import  find_peaks, find_peaks_cwt
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from math import sqrt
from collections import Counter 
import sys

############################
norms = ('raw', 'l1', 'l2')
############################

def pipeline( a, norm ) :
    from seaborn import heatmap

    fig = pp.figure(figsize=(10,4))
    print('array shape:', a.shape)
    
    if not len(a):
        print('No data')
        return
    
    peaks = PeakTool(a, norm=norm)

    
    print(peaks.drop_nonproper(False))
    print(len(peaks))

    #ax = fig.add_subplot( 2, 2, 1 )
    #peaks.overlay_plot( ax=ax )
    #ax.plot( peaks.get_median() )

    #ax = fig.add_subplot( 2, 2, 2 )
    #heatmap(peaks.df, ax=ax)


    ax = fig.add_subplot( 1, 2, 1 )
    print('dropped by strict check:',len(peaks.drop_nonproper(True)))
    peaks.overlay_plot(ax=ax)
    print('current # of peaks',len(peaks))
    ax.plot( peaks.get_median() )

    peaks.trim_ends()

    ax = fig.add_subplot( 1, 2, 2 )
    heatmap( peaks.df, ax=ax )
    pp.show()


    
    peaks.run_kmeans(n_clusters=2, plot_peaks=pp.figure( figsize=(10, 4)), )
                     #plot_dist=pp.figure( figsize=(20,8))
                    #)
    print( Counter(peaks.kmeans.labels_) )
    pp.show()

    fig = pp.figure( figsize=(10, 4))
    ax = fig.add_subplot(1,2,1)
    peaks.run_pca( ax=ax )
    
    ax = fig.add_subplot(1,2,2)
    peaks.run_tsne( ax=ax )
    
    pp.show()
    return peaks


def pair_pipeline( fnbase ):
    
    print('*'*30)
    print('* Analyzing', fnbase, '...')
    print('*'*30)

    a = np.fromfile(fnbase+'01', dtype='>i2' )
    b = np.fromfile(fnbase+'02', dtype='>i2' )
    
    if len(a)==0 or len(b)==0 :
        print('No results!')
        return None, None
    
    peaks1 = pipeline(a, 'raw')
    peaks2 = pipeline(b, 'raw')
    
    aa = np.zeros( len(peaks1.data) )
    bb = np.zeros( len(peaks1.data) )

    for v in peaks1.values():
        aa[v.center-10:v.center+10] = 1

    for v in peaks2.values():
        bb[v.center-10:v.center+10] = 1

    sum(aa)/20, sum(bb)/20
    x = {}
    for i in np.arange(-500, 500):
        x[i] = sum(np.roll(aa, i) * bb)

    x = pd.DataFrame([(i,j) for i,j in x.items()])
    pp.plot(x[0], x[1])
    pp.show()
    print('\n')
    
    return peaks1, peaks2
