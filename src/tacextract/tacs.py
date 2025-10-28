""" TAC functions """
from pathlib import Path
import numpy as np
import csv
import re

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

def tacread(filename: str):
	""" Read Time Activity Curve (tac) file

	Parameters
	----------
	filename : str
	   filename of tacfile

	"""

	# Read header row
	f = open(filename)
	header = f.readline().split('\t')

	# Read data
	tac = np.loadtxt(filename, delimiter='\t', skiprows=1, comments='#')
	
	if tac.ndim == 1:
		tac= tac[None,:]
	units = {}

	# Get timeunit from first column header
	units[0] = re.search(r"(?<=\[)[^)]*(?=\])",header[0]).group(0)

	# Get concentration unit from second column header
	units[1] = re.search(r"(?<=\[)[^)]*(?=\])",header[1]).group(0)

	return tac, header, units