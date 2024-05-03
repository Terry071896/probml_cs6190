import yaml

from base_sindy import E_SINDy, BaseSINDy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open("../configs/config_sim.yaml", 'r') as stream:
    try:
        config=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if config['data_path'].endswith('massSpringData_blue_2_5.csv'):
    df = pd.read_csv(config['data_path'])[:100]
    m = df.to_numpy()
    t = m[:, 1]
    X = np.array([m[:, 3]])
elif 'tea' in config['data_path']:
    df = pd.read_csv(config['data_path'])
    df = df[df.time < 500]
    df = df[df.time > 7]
    df = df.iloc[list(range(0, len(df), 5))]
    m = df.to_numpy()
    t = m[:, 0]
    X = m[:, 1:].T
elif 'singleGreen200' in config['data_path']:
    df = pd.read_csv(config['data_path'])
    m = df.to_numpy()
    t = m[:, 0]
    X = m[:, 1:].T
else:
    
    t = np.linspace(0,25, 400)
    c0 = 1
    c1 = 1
    k = 0.25
    
    X = np.array([c0*np.cos(np.sqrt(k)*t)+c1*np.sin(np.sqrt(k)*t)])
    df = pd.DataFrame(np.array([t, X[0]]).T, columns=['t', 'x'])


# optimizer = config['E_SINDy']['optimizer']
# sindy = BaseSINDy(config=config, **config['SINDy'], **config['mlflow'])

# if optimizer == 'ridge':
#     optimizer = sindy.ridge
#     opti_vars = None
# elif optimizer == 'lasso':
#     optimizer = sindy.lasso
#     opti_vars = None
# elif optimizer == 'bayesian_ridge_regression':
#     optimizer = sindy.bayesian_ridge_regression
#     opti_vars = config['bayesian_ridge_regression']
# else:
#     optimizer = sindy.ridge
#     opti_vars = None

# coefs = sindy.fit(t, X, optimizer=optimizer, opti_vars=opti_vars, iterates=config['stls_iterates'])
# print(sindy.coefs)



esindy = E_SINDy(config, **config['E_SINDy'], **config['mlflow'])
esindy.fit(t, X, **config['SINDy'], stls_iterates = config['stls_iterates'])

a = np.array(list(esindy.coefs['x_0']['mean'].values()))
print(np.abs(a)/np.sum(np.abs(a)))

df_coefs = esindy.coefs['x_0']['df'].dropna()
df_coefs[[a for a in df_coefs.columns if np.sum(np.abs(df_coefs[a])) > 0.0001]].hist(figsize=(10,20))
print(df_coefs)
