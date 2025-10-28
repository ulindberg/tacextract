""" TAC functions """
from pathlib import Path
import numpy as np
import csv

def tacwrite(FrameTimesStart:np.ndarray,FrameDuration:np.ndarray, tac:np.ndarray, tacfile:Path, unit:str='Bq/ml', label=None):
    """ Write time activity curve as PMOD (.tac) file format
    
    Args:
            FrameTimesStart:
            FrameDuration:
            tac:
            tacfile:
            unit:
            label:

        Returns:
            tacfile
        """
    
    # Create Header
    if label is None:
        label = ['tac1']
        if tac.ndim > 1:
            for i in range(2,np.size(tac,1)+1):
                label += ['tac'+str(i)]
    header = ['start[seconds]', f'end[{unit}]'] + label[:]
    # header.extend(('start[seconds]', f'end[{unit}]', label))
    
    # Concatenate columns of time and signal(s)
    outarray = np.stack((FrameTimesStart,FrameTimesStart+FrameDuration,tac),axis=1)
    
    with open(tacfile, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # write the header
        writer.writerow(header)
        
        # write multiple rows
        writer.writerows(outarray)
    
    return tacfile