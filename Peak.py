class Peak :
    def __init__(self, 
                 maxpos=None, 
                 maxval=None, 
                 minpos=None,
                 minval=None,
                 begin=None,
                 end=None,
                 data=None,
                 prepeak=20, 
                 postpeak=50, 
                 norm='raw'
                ):
        
        self.prepeak = prepeak
        self.postpeak = postpeak
        self.maxpos = maxpos
        self.maxval = maxval
        self.minpos = minpos
        self.minval = minval
        self.begin = begin
        self.end = end
        self.data = data
        self.center = None
        
        self.series = []
        self.nf = None
        self.norm = norm
        
        if self.maxpos != None and self.minpos !=None :
            self.center = ( self.maxpos + self.minpos )//2
        
       
        
    def __len__(self) :
        return self.end - self.begin
        
    def is_proper( self, strict=False, verbose=False ):
        simple_check = (self.center != None and self.maxpos != None and self.minpos != None and self.begin != None and self.end != None)
        if not simple_check :
            return False
        
        if not strict :
            return simple_check
        else :
            maxcheck = self.series.idxmax() < 0
            mincheck = self.series.idxmin() > 0
            
            if verbose :
                print(self.series.idxmax(), self.series.idxmin() )
                if not maxcheck :
                    print( 'Peak [%s,%s) failed strict maximum condition!'%(self.begin, self.end))
                    pp.plot( self.series )
                    pp.show()
                if not mincheck :
                    print( 'Peak [%s,%s) failed strict minimum condition!'%(self.begin, self.end))
                    pp.plot( self.series )
                    pp.show()
            return maxcheck and mincheck
            
    def set_peak( self, pos=None, distance=20, plot=False ) :
        #Get the new adjusted peak center and call normalization function
        if pos== None :
            pos = self.maxpos
            
        assert pos != None, "Center value should be provided!" 
        assert len(self.data) > 0, 'Data should have been defined!'
        
        new_cur = pos
        c = 0
        d = distance
        
        while True :
            c += 1
            cur = new_cur
            
            mx = self.data[cur-d:cur]
            mn = self.data[cur:cur+d]
            
            if not len(mx) or not len(mn):
                return 0
            
            argmax = cur-d + mx.argmax()
            argmin = cur + mn.argmin()
            new_cur = (argmax + argmin)//2

            if plot:
                print(cur, 'x', argmax, ':', self.data[argmax], '|', argmin, ':', self.data[argmin])
                pp.plot( cur, 0, 'x', color='black')
                pp.plot( argmax, self.data[argmax], 'x')
                pp.plot( argmin, self.data[argmin], 'o')
        
                print(cur, '->', new_cur)
            
            if new_cur == cur :
                break
        
            if c > 1000 :
                assert False, 'Iteration maximum reached!'
                
        #checking boundary conditions
        if argmax - self.prepeak < 0 :
            return 0
        elif argmin + self.postpeak >= len(self.data) :
            return 0
        else :
            pass
        
        self.maxpos = argmax
        self.minpos = argmin
        
        self.begin = argmax - self.prepeak
        self.end = argmin + self.postpeak
            
        self.maxval = self.data[argmax]
        self.minval = self.data[argmin]
        self.center = (self.maxpos + self.minpos) // 2
        self.normalize()
        return self.center
        
    def normalize( self ) :
        #get normalized values
        self.series = pd.Series( self.data[self.begin:self.end], index=np.arange(self.begin-self.center, self.end-self.center))
        self.cindex = pd.array(self.series.index)
        
        #initialization of rormalization factors
        nf = self.get_norm()
        
        #make sure normalization factor has been set correctly!
        assert nf == self.nf
    
        self.series = self.series/self.nf
        
        return nf
    
    def overlay_plot( self, alpha=0.2, ax=pp, **kwargs ):
        #norm == None will print raw data
        if self.begin != None and self.end != None :
            ax.plot(self.series, alpha=alpha, **kwargs )
                
                
    def plot( self, ax=pp, **kwargs ):
        ax.plot( np.arange(self.begin, self.end), self.data[self.begin:self.end], **kwargs )
    
    def get_recentered_index_and_values(self) :   
        assert len(self.series['raw']) == len(self.data[self.begin:self.end])
        
        return self.cindex, self.data[self.begin:self.end]
    
    def diff( self, peak ):
        return np.array(self.series) - np.array(peak.series)
        
    def sim( self, peak ):
        return (self.series * peak.series).dropna()
    
    def sim_square_sum( self, peak ):
        return sum( [( float(i))**2 for i in self.sim(peak)] )
    
    def diff_abs_sum( self, peak ) :
        dv = self.diff(peak)
        return np.abs( dv ).sum()
    
    def diff_square_sum( self, peak ):
        dv = self.diff(peak)
        return sum([ (float(i))**2 for i in dv] )
    
    def get_norm( self ):
        
        norm = self.norm
            
        if self.nf == None :
            self.set_norm( norm )
            
        return self.nf #pre calculated normalized factor

        
        
    def set_norm( self, norm ):
        #setup a new norm
        if norm == 'raw' :
            self.nf = 1
        elif norm == 'l1':
            self.nf = np.abs( self.data[self.begin:self.end] ).sum()
        elif norm == 'l2':
            l = [(int(v))**2 for v in self.data[self.begin:self.end] ]
            s = sum( l )
            
            assert s >= 0 
            
            self.nf = sqrt( s )
        else :
            assert False
            
        return self.nf
        
    
    def dist( self, peak ):
        assert self.norm == peak.norm
        
        if self.norm == 'raw':
            return self.diff_abs_sum(peak)
        elif self.norm == 'l1' :
            return self.diff_abs_sum(peak)
        elif self.norm == 'l2' :
            return sqrt( self.diff_square_sum(peak) )
        else :
            assert False
            
    def to_series( self ):
        return self.series 
        
    def adjust_boundary( self, left_offset, right_offset ):
        self.begin = self.begin + left_offset
        self.end = self.end + right_offset
        self.normalize()
        
        assert self.begin < self.center < self.end
    
        return self.begin, self.end
    
    
