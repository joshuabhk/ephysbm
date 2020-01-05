class PeakTool(PeakSet) :
    
    def __init__( self, data=[], smoothing_window=20, neo_delta_t=10, norm='raw' ) :
        self.data = data
        self.original_data = None #after smoothing
        self.neo = []
        
        self.smoothing_window = smoothing_window
        self.neo_delta_t = neo_delta_t
        self.dropped = {}
        self.norm = norm
        self.kmeans = []
        self.pca = []
        self.tsne = []
        self.dists = []

    
        if not len(data) :
            return
        
        if smoothing_window :
            self.run_smoothing( smoothing_window )
            
        if neo_delta_t :
            self.run_neo( neo_delta_t )
            
        self.find_peaks()
            
    
    def run_smoothing( self, window=None, triangle=True ) :
        if window == None :
            window = self.smoothing_window
        hw = window//2
        #moving average setup
        pre = np.arange( 1, hw, 1.0 )
        post = np.arange( hw, 0, -1.0)
        offset=len(pre)
        w = np.array( list(pre) + list(post) )
        w = w/sum(w) #normalization!

        lw = len(w)
        
        a = self.data
        aa = np.zeros(len(a))
        
        if triangle :
            for i in np.arange(0,len(a)-len(w)):
                aa[i+offset] = (a[i:i+lw]*w).sum()
        else :
            for i in np.arange(0,len(a)-len(w)):
                aa[i+offset] = a[i:i+lw].mean()
        
        self.original_data = self.data
        self.data = aa
        
    def run_neo( self, dt=None ) :
    
        if dt == None :
            dt = self.neo_delta_t
            
        if dt == None :
            return self.neo
            
        #neo setup
        a = self.data
        neoa = np.zeros(len(a))

        for i in range(dt, len(a)-dt) :
            neoa[i] = a[i]**2 - a[i-dt]*a[i+dt]
            
        self.neo = neoa
    
    def find_peaks( self, a=[], distance=20, quantile=0.9 ): 
        if not len(a) :
            if len(self.neo) :
                a = self.neo
            else :
                a = self.data
                
        p, d = signal.find_peaks(a, distance=20, height=np.quantile(a, quantile) )
        self.peak_positions = p
        self.peak_metadata = d
        
        self.peaks = super().__init__(self.data, p, autofix=True, getmodel=True, norm=self.norm)
    
        return self.peaks

    def drop_nonproper( self, strict=True ):
        dropped = super().drop_nonproper(strict=strict)
        self.dropped.update(dropped)
        
        return dropped
    
    def run_pca( self, plot=True, ax=pp ) :
        from sklearn.decomposition import PCA
        
        x = self.df.dropna(axis=1)
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ['component 1', 'component 2'])

        self.pca = principalDf
        if plot :
            ax.set_xlabel('Principal Component 1', fontsize = 15)
            ax.set_ylabel('Principal Component 2', fontsize = 15)
            ax.set_title('2 component PCA', fontsize = 20)
            
            if self.kmeans :
                kmeans = self.kmeans
                principalDf['kmeans'] = kmeans.labels_
                for i in sorted(Counter(kmeans.labels_)) :
                    ax.scatter(principalDf[kmeans.labels_==i]['component 1'], 
                               principalDf[kmeans.labels_==i]['component 2'], 
                               label=i)
                ax.legend()
            else :
                ax.scatter( principalDf['component 1'], principalDf['component 2'] )

    def run_tsne( self, plot=True, ax=pp ):
    
        from sklearn.manifold import TSNE
        x = self.df.dropna(axis=1)
        tsne = TSNE(n_components=2)
        principalComponents = tsne.fit_transform(x)
        
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ['component 1', 'component 2'])
        
        self.tsne = principalDf
        
        if plot :
            ax.set_xlabel('TSNE Component 1', fontsize = 15)
            ax.set_ylabel('TSNE Component 2', fontsize = 15)
            ax.set_title('2 component TSNE', fontsize = 20)
            if self.kmeans :
                kmeans = self.kmeans
                principalDf['kmeans'] = kmeans.labels_
                for i in sorted(Counter(kmeans.labels_)) :
                    ax.scatter( principalDf[kmeans.labels_==i]['component 1'], 
                               principalDf[kmeans.labels_==i]['component 2'],
                              label=i)
                ax.legend()
            else :
                ax.scatter( principalDf['component 1'], principalDf['component 2'] )

                
    def cal_dists( self ) :
        if len(self.dists) :
            return
        
        kv = [(k,v) for k,v in self.items()]
        dists = []
        for i, (k,v) in enumerate(kv) :
            #dists.append([k,k,0])
            for j, (l,w) in zip( range(i+1,len(kv)), kv[i+1:] ) :
                l2d = v.dist(w)
                dists.append( [k,l, l2d] )
                dists.append( [l,k, l2d] )
        
        self.dists = dists
        return True
        
    def run_kmeans( self, n_clusters=3, plot_peaks=None, plot_dist=None ) :
        from sklearn.cluster import KMeans
        X = self.df.dropna(axis=1)
        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        #print(kmeans.labels_)

        #kmeans.predict(x)
        #kmeans.cluster_centers_


        #Counter(kmeans.labels_)

        self.kmeans = kmeans
        
        if plot_peaks :
            for i in sorted(Counter(kmeans.labels_)) :
                ax = plot_peaks.add_subplot(2,2,i+1)
                ax.set_title('Peaks in cluter %s'%i, fontsize = 20)
                for c, (k,p) in zip(kmeans.labels_, self.items()):
                    if c == i :
                        ax.plot( p.to_series(), alpha=0.2)
        
        if plot_dist :
            #print('before cal dist', file=sys.stderr)
            self.cal_dists()
            dists = self.dists
            #print('after cal dist', file=sys.stderr)

            i2c = {}
            for i,c in zip(self.keys(), kmeans.labels_):
                i2c[i] = c

            clustdists = [[i2c[i],i2c[j],k] for i,j,k in dists]

            plotnum = 0
            for i in sorted(Counter(kmeans.labels_)) :
                plotnum += 1
                ax = plot_dist.add_subplot(2,2,plotnum)
                distplot( [k for ci,cj,k in clustdists if ci==i and cj==i], 
                         ax=ax, 
                         label='within %s'%i )
                
                ax.set_title( "cluster %s"%i)
                
                for j in sorted(Counter(kmeans.labels_)):
                    if i != j :
                        distplot( [k for ci,cj,k in clustdists if ci==i and cj==j], 
                                 ax=ax, 
                                 label='between %s & %s'%(i,j) )
                    
                ax.legend()

           
