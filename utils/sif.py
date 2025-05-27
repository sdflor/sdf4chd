import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from vtk_utils.vtk_utils import *
import io_utils
import argparse

def face_intersection(fn):
    tmp_dir = os.path.join(os.path.dirname(fn), 'temp')
    try:
        os.makedirs(tmp_dir)
    except:
        pass

    mesh = load_vtk_mesh(fn)
    region_ids = np.unique(vtk_to_numpy(mesh.GetPointData().GetArray('RegionId'))).astype(int)
    total_num = 0
    bad_num = 0
    percentage_list = []
    for i in region_ids:
        poly_i = thresholdPolyData(mesh, 'Scalars_', (i, i),'cell')
        if poly_i.GetNumberOfPoints() == 0:
            poly_i = thresholdPolyData(mesh, 'RegionId', (i, i),'point')
        name_i = os.path.basename(fn).split('.')[0] + '_{}'.format(i)
        out_fn_i = os.path.join(tmp_dir, name_i + '.stl')
        write_vtk_polydata(poly_i, out_fn_i)
        os.system('/Users/fanweikong/Documents/Modeling/tetgen/build/tetgen -d ' + out_fn_i)
        if os.path.isfile(os.path.join(tmp_dir, name_i+'_skipped.face')):
            with open(os.path.join(tmp_dir, name_i+'_skipped.face')) as f:
                line = f.readline()
                bad_num = int(line.split(' ')[0])
        total_num = poly_i.GetNumberOfCells()
        percentage_list.append(float(bad_num)/float(total_num)*100)
    return percentage_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',  help='File name of the mesh.')
    args = parser.parse_args()
    fns = glob.glob(os.path.join(dir_n, '*_spstd0.5_r*.vtp'))
    p_list = []
    for fn in fns:
        percentage_list = face_intersection(fn)
        p_list.append(percentage_list)
    io_utils.write_scores(os.path.join(dir_n, 'sif_scores.csv'), p_list)
