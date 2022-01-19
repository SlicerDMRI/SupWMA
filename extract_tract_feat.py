import utils.tract_feat as tract_feat

import whitematteranalysis as wma
import numpy as np

import argparse
import os
import h5py


def gen_features():
    print(script_name, 'Computing feauture:', args.feature)
    if args.feature == 'RAS':
        feat_RAS = tract_feat.feat_RAS(pd_tract, number_of_points=args.numPoints)

        # Reshape from 3D (num of fibers, num of points, num of features) to 4D (num of fibers, num of points, num of features, 1)
        # The 4D array considers the input has only one channel (depth = 1)
        feat_shape = np.append(feat_RAS.shape, 1)
        feat = np.reshape(feat_RAS, feat_shape)

    elif args.feature == 'RAS-3D':

        feat_RAS_3D = tract_feat.feat_RAS_3D(pd_tract, number_of_points=args.numPoints, repeat_time=args.numRepeats)

        feat = feat_RAS_3D

    elif args.feature == 'RASCurvTors':

        feat_curv_tors = tract_feat.feat_RAS_curv_tors(pd_tract, number_of_points=args.numPoints)

        feat_shape = np.append(feat_curv_tors.shape, 1)

        feat = np.reshape(feat_curv_tors, feat_shape)

    elif args.feature == 'CurvTors':

        feat_curv_tors = tract_feat.feat_curv_tors(pd_tract, number_of_points=args.numPoints)

        feat_shape = np.append(feat_curv_tors.shape, 1)

        feat = np.reshape(feat_curv_tors, feat_shape)

    else:
        raise ValueError('Please enter valid feature names.')

    print(script_name, 'Feature matrix shape:', feat.shape)

    return feat


if __name__ == "__main__":
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(description="Compute FiberMap of input vtk file.",
                                     epilog="Written by Tengfei Xue, Fan Zhang, txue3@bwh.harvard.edu, fzhang@bwh.harvard.edu")

    parser.add_argument('inputVTK', help='input tractography data as vtkPolyData file(s).')
    parser.add_argument('outputDir', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-outPrefix', type=str, help='A prefix string of all output files.')

    parser.add_argument('-feature', action="store", type=str, help="Name of feature`")   # RAS: Right-Anterior-Superior
    parser.add_argument('-numPoints', action="store", type=int, default=15, help='Number of points per fiber to extract feature.')
    parser.add_argument('-numRepeats', action="store", type=int, default=15, help='Number of repiteation times.')

    args = parser.parse_args()
    script_name = '<extract_tract_feat>'

    if not os.path.exists(args.inputVTK):
        print(script_name, "Error: Input tractography ", args.inputVTK, "does not exist.")
        exit()

    if not os.path.exists(args.outputDir):
        print(script_name, "Output directory", args.outputDir, "does not exist, creating it.")
        os.makedirs(args.outputDir)

    print(script_name, 'Reading input tractography:', args.inputVTK)
    pd_tract = wma.io.read_polydata(args.inputVTK)

    # generate features
    feat = gen_features()

    # Save feat
    with h5py.File(os.path.join(args.outputDir, args.outPrefix + '_featMatrix.h5'), "w") as f:
        f.create_dataset('feat', data=feat)

        print(script_name, 'Feature matrix shape:', feat.shape)
