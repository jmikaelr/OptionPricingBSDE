import numpy as np




class MCLSQ():
    """ Pricing options with monte carlo least square method using BSDEs """
    def __init__(self, S, K, r, z, T, p, m, opt_type, european = True):
        self.S = S      # Iniital Stock Price
        self.K = K      # Strike Price
        self.r = r      # Risk free interest rate
        self.z = z      # Stock Volatility  
        self.T = T      # Time to maturity
        self.p = p      # Periods (Time steps)
        self.m = m      # Number of simulations in MC
        self.dt = T/p

        self.opt_type = self._check_opt_type(opt_type)



    def _check_opt_type(self, opt_type):
        try:
            if opt_type.lower() == 'call':
                return 'call'
            elif opt_type.lower() == 'put':
                return 'put'
            else:
                raise TypeError('Invalid option type, should be call or put!')
        except:
            raise TypeError('Invalid option  type, should be call or put!')


    def _MC_simulator(self):
        stock_prices = np.zeros((self.p + 1, self.m))
        stock_prices[0] = self.S

    
        for t in range(1, self.p + 1):
            dw = np.random.normal(0, 1, self.m)
            stock_prices[t]= stock_prices[t-1] * np.exp((self.r - 0.5 * self.z**2)*self.dt + self.z * np.sqrt(self.dt) * dw)
    
        return stock_prices    
        

    def run(self):
        stock_prices = self._MC_simulator()
