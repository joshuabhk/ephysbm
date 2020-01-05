    
    
class PeakSet :
    def __init__(self, 
                 data, positions=[], 
                 autofix=True, getmodel=True, norm='raw') :
        
        self.data = data
        self.set = {pos:Peak(pos, data=data, norm=norm) for pos in positions }
        self.df = []
        self.model = []
        self.norm = norm
        
        if not len(positions) :
            self.find_peak( **kwargs )
        
        if autofix :
            for v in self.set.values() :
                v.set_peak()
        
        if getmodel :
            self.get_model()

            
    def add_peak( self, key, peak ):
        if key in self.set :
            assert False
        self.set[key] = peak
        
    def overlay_plot( self, alpha=0.2, ax=pp ):
        for k, v in self.set.items():
            v.overlay_plot(alpha=alpha, ax=ax)
            
    def __len__( self ):
        return len(self.set)
    
    def __getitem__(self, key):
        return self.set[key]
    
    def get_median( self ):
        if len(self.df) : 
            df = self.df
        else :
            df = {}
            for k, v in self.set.items() :
                if v.is_proper() :
                    df[k] = v.to_series()
        
            df = pd.DataFrame(df).transpose()
            self.df = df
            
        return df.median()
    
    def get_model( self ) :
        if not self.model :
            
            v = np.array(self.get_median())
            self.model = Peak( data=v )
            self.model.set_peak( self.model.data.argmax() )
            
        return self.model
    
    def drop_nonproper( self, strict=False ):
        dropped = {k for k,v in self.set.items() if not v.is_proper(strict)}
        
        #update dataframe of peaks!
        filtered_droplist = [ k for k in dropped if k in self.df.index ] #this is for adjusting pre-removed when building dataframe
        self.df = self.df.drop(filtered_droplist)
        
        return {k:self.set.pop(k) for k in dropped}
    
    def trim_ends( self ):
        col = self.df.dropna(axis=1).columns
        mn, mx = min(col), max(col)
    
        for v in self.values() :
            v.adjust_boundary( mn - v.cindex[0], mx - v.cindex[-1] )
        
        
            
    def items( self ):
        return self.set.items()
    
    def keys( self ) :
        return self.set.keys()

    def values(self) :
        return self.set.values()
    
   
