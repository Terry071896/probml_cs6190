import pandas as pd
import numpy as np
import time
import mlflow
import math
import sympy
#from itertools import combinations
import os
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

file_path = os.path.realpath(__file__)
print(file_path)
file_dir = '/'.join(file_path.split('/')[:-1])



class BaseSINDy(object):

    def __init__(self, config, poly_order=5,
                    #max_derivatives_order=4,
                    include_t=False,
                    include_poly_predictor=False, 
                    order_predict=1, 
                    thresh=1e-6, 
                    alpha=1.0, 
                    use_preprocessing = True,
                    k_spline = 3,
                    interpolated_dt = 0.1,
                    use_mlflow = True,
                    experiment_name='sindy_model', 
                    run_name='sindy_base') -> None:
        self.config = config
        self.poly_order = poly_order
        #self.max_derivatives_order = max_derivatives_order
        self.include_t = include_t
        self.include_poly_predictor = include_poly_predictor
        self.order_predict = order_predict
        self.thresh = thresh
        self.alpha = alpha
        self.use_preprocessing = use_preprocessing
        self.k_spline = k_spline
        self.interpolated_dt = interpolated_dt
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.use_mlflow = use_mlflow
        self.x_splines = None
        if self.use_mlflow:
            mlflow.log_param('poly_order', poly_order)
            #mlflow.log_param('max_derivatives_order', max_derivatives_order)
            mlflow.log_param('include_t', include_t)
            mlflow.log_param('include_poly_predictor', include_poly_predictor)
            mlflow.log_param('thresh', thresh)
            mlflow.log_param('order_predict', order_predict)
            mlflow.log_param('alpha', alpha)
            mlflow.log_param('use_preprocessing', use_preprocessing)
            if use_preprocessing:
                mlflow.log_param('interpolated_dt', interpolated_dt)


    def get_interpolated_t(self, t):
        if self.interpolated_dt is False:
            return t
        if self.interpolated_dt < 1:
            t_s = np.array(list(np.linspace(t[0], t[-1], int((t[-1] - t[0])//self.interpolated_dt))))
        elif self.interpolated_dt > 1:
            t_s = np.array(list(np.linspace(t[0], t[-1], int(self.interpolated_dt))))
        else:
            t_s = t
        return t_s
    
    def smooth(self, t, x, t_s=None, coef=None):
        if coef is None:
            coef=self.config['smooth_coef']
        spl = UnivariateSpline(t, x)
        spl.set_smoothing_factor(coef)
        #print('smoothing:', coef)
        if t_s is None:
            return t, spl(t)
        else:
            return t_s, spl(t_s)
    
    def preprocesser(self, t, X, n=None):
        print('\t\t start spline...')
        t_s = self.get_interpolated_t(t)
        self.x_splines = {}
        X_new = np.zeros((len(X), len(t_s)))
        for i, x in enumerate(X):
            start = time.time()
            _, x = self.smooth(t, x, t_s=t_s)
            end = start - time.time()
            X_new[i] = x
            if self.use_mlflow:
                mlflow.log_metric(f'preprocessing/x_{i}_solve_time', end)
        print('\t\t spline done.')
        return t_s, X_new

    def take_first_derivative(self, t, x, one_pt=True):
        if one_pt:
            dx = np.diff(x)/np.diff(t) 
            t=t[:-1]
        else:
            dx = (-1*x[4:]+8*x[3:-1]-8*x[1:-3]+x[:-4])/(12*np.diff(t)[1:-2])
            t = t[1:-3]
        if self.config['derivative_smooth_coef'] is not False:
            t, dx = self.smooth(t, dx, coef=self.config['derivative_smooth_coef'])
        return dx


    def take_second_derivative(self, t, x, one_pt=True):
        if one_pt:
            dxx = np.diff(self.take_first_derivative(t,x))/np.diff(t[:-1])
            t=t[:-2]
        else:
            dxx = (-1*x[4:]+16*x[3:-1]-30*x[2:-2]+16*x[1:-3]-x[:-4])/(12*np.diff(t)[1:-2]**2)
            t=t[1:-3]
        if self.config['derivative_smooth_coef'] is not False:
            t, dxx = self.smooth(t, dxx, coef=self.config['derivative_smooth_coef'])
        return dxx

    def take_third_derivative(self, t, x, one_pt=True):
        #return (x[4:] - 2*x[3:-1] + 2*x[1:-3] - x[:-4])/(2*np.diff(t)[1:-2]**3)
        if one_pt:
            dxxx = np.diff(self.take_second_derivative(t,x))/np.diff(t[:-2])
            t=t[:-3]
        else:
            dxxx = (x[4:] - 2*x[3:-1] + 2*x[1:-3] - x[:-4])/(2*np.diff(t)[1:-2]**3)
            t=t[1:-3]
        if self.config['derivative_smooth_coef'] is not False:
            t, dxxx = self.smooth(t, dxxx, coef=self.config['derivative_smooth_coef'])
        return dxxx

    def take_forth_derivative(self, t, x, one_pt=True):
        #return (x[4:] - 4*x[3:-1] + 6*x[2:-2] - 4*x[1:-3] + x[:-4])/(np.diff(t)[1:-2]**4)
        if one_pt:
            dxxxx = np.diff(self.take_third_derivative(t,x))/np.diff(t[:-3])
            t=t[:-4]
        else:
            dxxxx = (x[4:] - 4*x[3:-1] + 6*x[2:-2] - 4*x[1:-3] + x[:-4])/(np.diff(t)[1:-2]**4)
            t=t[1:-3]
        if self.config['derivative_smooth_coef'] is not False:
            t, dxxxx = self.smooth(t, dxxxx, coef=self.config['derivative_smooth_coef'])
        return dxxxx

    def take_derivative(self, t, x, derv=1, one_pt=True):
        if derv == 1:
            return self.take_first_derivative(t, x, one_pt=one_pt)
        elif derv == 2:
            return self.take_second_derivative(t, x, one_pt=one_pt)
        elif derv == 3:
            return self.take_third_derivative(t, x, one_pt=one_pt)
        elif derv == 4:
            return self.take_forth_derivative(t, x, one_pt=one_pt)
        else:
            if one_pt:
                return x
            else:
                return x[1:-3]

    def build_ThetaX(self, t, X):
        t = np.array(t)
        self.t = t
        predictors = {}
        labels = {}
        #print(t.shape)
        if self.use_preprocessing:
            t, X = self.preprocesser(t, X)
            #print(t.shape)

        for i, x in enumerate(X):
            x = np.array(x)
            t, x = self.smooth(t,x)
            t_s = self.get_interpolated_t(t)
            self.t = np.array(t)
            #print(self.t.shape)

            if self.use_preprocessing:
                dervs = []
                for d in range(self.k_spline):
                    if self.config['derivative_method'] == 'one_pt':
                        shift = (-1*(self.k_spline-1-d))
                        if shift == 0:
                            shift = None
                        
                        dx = self.take_derivative(t, x, d, one_pt=True)[:shift]
                        self.t = t[:-1*(self.k_spline-1)]
                    elif self.config['derivative_method'] == 'five_pt':
                        dx = self.take_derivative(t, x, d, one_pt=False)
                        self.t = t[1:-3]
                    else:
                        print('something went wrong!')
                        return
                    
                    dervs.append(dx)
                    if d == 0:
                        lab = 'x_%s'%i
                    else:
                        lab = 'd'+'x'*d+'_%s'%i
                    #print(self.t.shape, dx.shape)
                    # plt.plot(self.t, dx, '.--')
                    # plt.title(lab)
                    # plt.show()

                    if self.order_predict == d:
                        predictors[lab] = dx
                    else:
                        labels[lab] = dx
                

            else:
                dx = self.take_first_derivative(t, x)[:-2]
                dxx = self.take_second_derivative(t, x)[:-1]
                dxxx = self.take_third_derivative(t, x)[:]
                #dxxxx = self.take_forth_derivative(t, x)
                x = x[:-3]
                self.t = t[:-3]

                if self.order_predict == 0:
                    predictors['x_%s'%i] = x
                    labels['dx_%s'%i] = dx
                    labels['dxx_%s'%i] = dxx
                    labels['dxxx_%s'%i] = dxxx
                    #labels['dxxxx_%s'%i] = dxxxx
                elif self.order_predict == 1:
                    labels['x_%s'%i] = x
                    predictors['dx_%s'%i] = dx
                    labels['dxx_%s'%i] = dxx
                    labels['dxxx_%s'%i] = dxxx
                    #labels['dxxxx_%s'%i] = dxxxx
                elif self.order_predict == 2:
                    labels['x_%s'%i] = x
                    labels['dx_%s'%i] = dx
                    predictors['dxx_%s'%i] = dxx
                    labels['dxxx_%s'%i] = dxxx
                    #labels['dxxxx_%s'%i] = dxxxx
                elif self.order_predict == 3:
                    labels['x_%s'%i] = x
                    labels['dx_%s'%i] = dx
                    labels['dxx_%s'%i] = dxx
                    predictors['dxxx_%s'%i] = dxxx
                    #labels['dxxxx_%s'%i] = dxxxx
                elif self.order_predict == 4:
                    labels['x_%s'%i] = x
                    labels['dx_%s'%i] = dx
                    labels['dxx_%s'%i] = dxx
                    labels['dxxx_%s'%i] = dxxx
                    #predictors['dxxxx_%s'%i] = dxxxx
                else:
                    print('order_predict must be 0, 1, or 2.')
                    return

        if self.include_t:
            labels['t'] = t

        if self.include_poly_predictor:
            labels = {**labels, **predictors}
        
        orig_labels = list(labels.keys())
        for l in orig_labels:
            for i in range(2,self.poly_order+1):
                labels['%s^%s'%(l,i)] = labels[l]**i


        if self.include_poly_predictor:
            labels = {l:v for l, v in labels.items() if l not in predictors.keys()}

        self.ThetaX = np.array(list(labels.values()))
        self.labs = np.array(list(labels.keys()))
        self.b = np.array(list(predictors.values()))
        self.b_labs = np.array(list(predictors.keys()))
        
        #print(self.ThetaX.shape, self.labs.shape, self.b.shape, self.b_labs.shape)
        self.labels = labels
        self.predictors = predictors
        
        return (self.ThetaX, self.labs), (self.b, self.b_labs)

    
    def normalize(self, df_X, b):
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        normalized_X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        normalized_b = (b - np.mean(b)) / np.std(b)
    
        return pd.DataFrame(normalized_X, columns=df_X.columns), normalized_b

    def STLS(self, df_X, b, optimizer, opti_vars=None, iterates=3):
        coeffs_all = {col:0 for col in df_X.columns}
        
        for i in range(iterates):
            #norm_X, norm_b = self.normalize(df_X=df_X, b = b)
            if opti_vars:
                coeffs, model = optimizer(df_X, b, **opti_vars)
            else:
                coeffs, model = optimizer(df_X, b)
            remain = []
            try:
                it = 0
                modified_coefs = []
                for m, std in zip(coeffs, np.array(model.sigma_).diagonal()):
                    if m < 0 and m+2*std > 0:
                        modified_coefs.append(0)
                    elif m > 0 and m-2*std < 0:
                        modified_coefs.append(0)
                    else:
                        modified_coefs.append(m)
                    it += 1
                coeffs = modified_coefs
                print('incorporating sigmas')
            except:
                pass
            print('coefs:', coeffs)
            if np.sum(np.abs(coeffs)) < self.thresh:
                return {col:0 for col in coeffs_all.keys()}
            percents = np.abs(coeffs)/np.sum(np.abs(coeffs))
            print('precents:', percents, sum(percents))
            i = 0
            for col, beta in zip(df_X.columns, coeffs):
                if np.abs(beta) < self.thresh or percents[i] < self.config['percent_thresh']:
                    beta = 0
                else:
                    remain.append(col)
                coeffs_all[col] = beta
                i += 1
            df_X = df_X[remain]
        if opti_vars:
            coeffs, model = optimizer(df_X, b, **opti_vars)
        else:
            coeffs, model = optimizer(df_X, b)
        for col, beta in zip(df_X.columns, coeffs):
            coeffs_all[col] = beta
        return coeffs_all

    
    def bayesian_ridge_regression(self, df_X, b, lambda_=1.0):
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        start = time.time()
        model = BayesianRidge(alpha_init=self.alpha, lambda_init=lambda_)
        model.fit(X, b)
        x = model.coef_
        print(x.shape)
        print('Sigma', np.array(model.sigma_).diagonal())
        print('Mean coefs', x)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
            mlflow.log_metric('sigma', model.sigma_)
            mlflow.log_param('bayesian_ridge/alpha', self.alpha)
            mlflow.log_param('bayesian_ridge/lambda', lambda_)
        return x, model
    

    def lasso(self, df_X, b):
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        start = time.time()
        r = Lasso(self.alpha).fit(X,b)
        x = r.coef_
        print(x.shape)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
            mlflow.log_param('lasso/alpha', self.alpha)
        return x, r

    def ridge(self, df_X, b):
        X = np.asmatrix(df_X.to_numpy())
        b = np.array(b)
        start = time.time()
        r = Ridge(self.alpha).fit(X,b)
        x = r.coef_
        print(x.shape)
        end = time.time()
        if self.use_mlflow:
            mlflow.log_metric('optimizer_time (sec)', end-start)
            mlflow.log_param('ridge/alpha', self.alpha)
        return x, r


    def mlflow_save_results(self, results):
        for x, err, i in zip(results['x_k'], results['error'], results['iterations']):
            mlflow.log_metrics({f'beta/{k}':v for k, v in zip(self.labs, x)}, step=i)
            mlflow.log_metric('error', err, step=i)

            
    def fit(self, t, X, optimizer=None, opti_vars=None, iterates=3):
        if optimizer is None:
            optimizer = self.ridge


        X_vars, bs = self.build_ThetaX(t, X)


        df_X = pd.DataFrame(X_vars[0].T, columns=X_vars[1])
        df_X.index = self.t
        self.df_X = df_X
        coefs_all = {k: None for k in bs[1]}
        non_zero_coefs_all = {k: None for k in bs[1]}
        for b, b_lab in zip(bs[0], bs[1]):
            print(b_lab)
            coefs = self.STLS(df_X, b, optimizer, opti_vars=opti_vars, iterates=iterates)
            coefs_all[b_lab] = coefs#{k:coefs[k] for k in bs[1]}
            #print(coefs)
            non_zero_coefs_all[b_lab] = {k:coefs[k] for k in coefs.keys() if np.abs(coefs[k]) > self.thresh}

        self.coefs = coefs_all
        self.non_zero_coefs = non_zero_coefs_all

        return self.coefs

    def predict(self, t, X):
        X_vars, bs = self.build_ThetaX(t, X)
        return {k:np.asarray(X_vars[0])*np.array(self.coefs[k].values()) for k in self.coefs.keys()}

import random

class E_SINDy(object):

    def __init__(self, config, n_models = 50,
                    point_ratio=0.1,
                    optimizer = 'ridge',
                    random_seed = 42,
                    use_mlflow=True,
                    experiment_name='sindy_model', 
                    run_name='sindy_base') -> None:
        self.config = config
        self.n_models = n_models
        if point_ratio <= 0:
            print('point_ratio doesn\'t make sence.  Setting to 0.1')
            point_ratio = 0.1
        self.point_ratio = point_ratio
        self.optimizer = optimizer
        self.random_seed = random_seed
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name

        if use_mlflow:
            mlflow.log_param('n_model', n_models)
            mlflow.log_param('point_ratio', point_ratio)
            mlflow.log_param('optimizer', optimizer)
            mlflow.log_param('random_seed', random_seed)
        return

    def preprocesser(self, t, X, n=None):
        t_s = np.array(list(np.linspace(t[0], t[-1], int((t[-1] - t[0])//self.interpolated_dt))))
        #print(t_s[0], t_s[-1])
        for i, x in enumerate(X):
            start = time.time()
            _, x = self.smooth(t, x, t_s=t_s)
            end = start - time.time()
            X[i] = x
            if self.use_mlflow:
                mlflow.log_metric(f'preprocessing/x_{i}_solve_time', end)
        return t_s, X

    def run_base_sindy(self, t, X, poly_order=5,
                    include_t=False,
                    include_poly_predictor=False, 
                    order_predict=1, 
                    thresh=1e-6, 
                    alpha=1.0,
                    use_preprocessing = True,
                    k_spline = 3,
                    interpolated_dt = 0.1,
                    stls_iterates=3):

        sindy = BaseSINDy(config = self.config, poly_order=poly_order, 
                            include_t=include_t, 
                            include_poly_predictor=include_poly_predictor, 
                            order_predict=order_predict, 
                            thresh=thresh, 
                            alpha=alpha, 
                            use_preprocessing = use_preprocessing,
                            k_spline = k_spline,
                            interpolated_dt = interpolated_dt,
                            use_mlflow=self.use_mlflow)

        if self.optimizer == 'ridge':
            optimizer = sindy.ridge
            opti_vars = None
        elif self.optimizer == 'lasso':
            optimizer = sindy.lasso
            opti_vars = None
        elif self.optimizer == 'bayesian_ridge_regression':
            optimizer = sindy.bayesian_ridge_regression
            opti_vars = self.config['bayesian_ridge_regression']
        else:
            optimizer = self.optimizer
            opti_vars = None
        
        print(t.shape, X.shape)
        coefs = sindy.fit(t, X, optimizer=optimizer, opti_vars=opti_vars, iterates=stls_iterates)
        self.all_spline.append(sindy.x_splines)
        return coefs, sindy.df_X

    def smooth(self, t, x, t_s=None):
        coef=self.config['smooth_coef']
        spl = UnivariateSpline(t, x)
        spl.set_smoothing_factor(coef)
        #print('smoothing:', coef)
        if t_s is None:
            return t, spl(t)
        else:
            return t_s, spl(t_s) 

    def preprocess(self, t, X):
        t_s = np.linspace(t[0], t[-1], 100)

        #print(t_temp[0], t_temp[-1])
        X_temp = []
        for x in X:
            _, x = self.smooth(t, x, t_s)
            X_temp.append(x)
        X_temp = np.array(X_temp)
        return t_s, X_temp

    def fit(self, t, X, 
                    poly_order=5,
                    include_t=False,
                    include_poly_predictor=False, 
                    order_predict=1, 
                    thresh=1e-6, 
                    alpha=1.0,
                    use_preprocessing = True,
                    k_spline = 3,
                    interpolated_dt = 0.1,
                    stls_iterates = 3):
        self.poly_order = poly_order
        self.include_t = include_t
        self.include_poly_predictor = include_poly_predictor
        self.thresh = thresh
        self.alpha = alpha
        self.use_preprocessing = use_preprocessing
        self.k_spline = k_spline
        self.interpolated_dt = interpolated_dt
        self.stls_iterates = stls_iterates
        
        all_coefs = {}
        all_dfs = {}
        self.all_spline = []

        # if self.use_preprocessing:
        #     t, X = self.preprocess(t, X)
        self.t = t
        self.X = X
        for i in range(self.n_models):
            print('running iteration: ', i)
            #random.seed(self.random_seed)
            if self.point_ratio < 1:
                sample_index = random.sample(list(range(len(t))), int(len(t)*self.point_ratio))
                sample_index.sort()
            elif self.point_ratio > 1:
                sample_index = random.sample(list(range(len(t))), int(self.point_ratio))
                sample_index.sort()
            else:
                sample_index = list(range(len(t)))

            t_temp = np.array(t[sample_index])
            #print(t_temp[0], t_temp[-1])
            X_temp = []
            for x in X:
                X_temp.append(np.array(x[sample_index]))
            X_temp = np.array(X_temp)
            print('\t start sindy..')
            coefs, df_X = self.run_base_sindy(t_temp, X_temp, poly_order=poly_order, 
                                        include_t=include_t, 
                                        include_poly_predictor=include_poly_predictor, 
                                        order_predict=order_predict, 
                                        thresh=thresh, 
                                        alpha=alpha,
                                        use_preprocessing = use_preprocessing,
                                        k_spline = k_spline,
                                        interpolated_dt = interpolated_dt, 
                                        stls_iterates=stls_iterates)
            print('\t sindy end.')
            
            all_dfs[i] = df_X
            for k, v in coefs.items():
                if k not in all_coefs.keys():
                    all_coefs[k] = [v]
                else:
                    all_coefs[k].append(v)
                if self.use_mlflow:
                    mlflow.log_metric(f'coef_average_{k}', np.mean(all_coefs[k]))
                    mlflow.log_metric(f'coef_standard_dev_{k}', np.std(all_coefs[k]))
        #print(all_coefs)
        self.coefs =  {k : {'mean' : pd.DataFrame(v).mean().to_dict(), 'std' : pd.DataFrame(v).std().to_dict(), 'df' : pd.DataFrame(v)} for k, v in all_coefs.items()}
        self.all_dfs = all_dfs
        if self.use_mlflow:
            mlflow.log_artifact(self.coefs, 'coefs.json')
        return self.coefs

    def predict(self):
        return
        