import argparse
from config.config import BasicConfig ,TabularConfig


parser = argparse.ArgumentParser(prog='test')
test = vars(TabularConfig('CartPole-v0', 500))
params = TabularConfig.__annotations__
params.update(BasicConfig.__annotations__)
for key, value in params.items():
    if value.__class__.__module__ == 'builtins':
        if type(value) == bool:
            parser.add_argument('--'+key, type=bool, action='store_true')
        else:
            parser.add_argument('--'+key, type=value, default=test[key])
        
args = parser.parse_args() 
arg_dict = vars(args)
print(args)
