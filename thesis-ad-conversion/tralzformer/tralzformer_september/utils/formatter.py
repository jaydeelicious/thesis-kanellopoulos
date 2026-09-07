from typing import Any
from numpy.typing import NDArray
import numpy as np

class Formatter:
    ''' ... '''
    def __init__(self, 
        modalities: dict[str, dict[str, Any]],
    ) -> None:
        ''' ... '''
        self.modalities = modalities

    def __call__(self,
        smp: dict[str, Any],
    ) -> dict[str, int | NDArray[np.float32] | None]:
        ''' ... '''
        new = dict()

        # loop through all data modalities
        for k, info in self.modalities.items():
            # the value is missing or equals None
            if k not in smp or smp[k] is None:
                if info['type'] == 'categorical':
                    new[k] = -1
                else:
                    new[k] = None
                continue

            # get value
            v = smp[k]

            # Replace "," with "." e.g. First_Age = 84,5724
            if isinstance(v, str):
                v = v.replace(",", ".")
                try:
                    v = float(v)
                except ValueError:
                    raise ValueError(f'"{k}" has an invalid value: {v}')
                
            # validate the value by using numpy's intrinsic machanism 
            try:
                v_np = np.array(v, dtype=np.float32)
            except:
                raise ValueError('\"{}\" has unexpected value {}'.format(k, v))
            
            # additional validation for categorical value
            if info['type'] == 'categorical':
                # print(k, v_np.shape)
                if v_np.shape != ():
                    raise ValueError('Categorical data \"{}\" has unexpected value {}.'.format(k, v))
                elif v == -1:
                    new[k] = -1
                    continue
                elif int(v) != v:
                    raise ValueError('Categorical data \"{}\" has unexpected value {}.'.format(k, v))
                elif v < 0: # or v >= info['num_categories']:
                    raise ValueError('Categorical data \"{}\" has unexpected value {}.'.format(k, v))
            
            # additional validation for numerical value
            elif info['type'] == 'numerical':
                if info['shape'] == [1] and v_np.shape != () and v_np.shape != (1,):
                    raise ValueError('Numerical data \"{}\" has unexpected shape {}.'.format(k, v_np.shape))
                elif info['shape'] != [1] and tuple(info['shape']) != v_np.shape:
                    raise ValueError('Numerical data \"{}\" has unexpected shape {}.'.format(k, v_np.shape))

            
                
            # format categorical value
            if info['type'] == 'categorical':
                new[k] = int(v)

            # format numerical value
            elif info['type'] == 'numerical':
                if info['shape'] == [1] and v_np.shape == ():
                    # unsqueeze the data
                    new[k] = np.array([v], dtype=np.float32)
                else:
                    new[k] = v_np
            

        return new