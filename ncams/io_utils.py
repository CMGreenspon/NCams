#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019-2022 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Utilities for input and output of data with different formats.
"""

import math
import csv
import ntpath


############################### CSV
def import_csv(filename):
    '''Imports csv into a simple structure.

    Arguments:
        filename {str} -- filename to import.
    Returns a tuple of:
        column_names {list of M str} -- list of all column names.
        values {list [M][N] of float if possible, str otherwise} -- list of all column values. First
            index corresponds to column number.
    '''
    with open(filename, 'r') as f:
        rdr = csv.reader(f)

        line = next(rdr)
        column_names = [i.strip() for i in line]

        values = [[] for _ in column_names]

        for li in rdr:
            for idof, vdof in enumerate(li):
                try:
                    v = float(vdof)
                except ValueError:
                    v = vdof
                values[idof].append(v)

    # clear empty
    for idof in reversed(range(len(column_names))):
        if (len(column_names[idof]) == 0 and
                all(isinstance(v, str) and len(v) == 0 for v in values[idof])):
            del column_names[idof]
            del values[idof]
    return column_names, values


def export_csv(filename, column_names, values):
    '''Exports from a structure into a csv file.

    Arguments:
        filename {str} -- filename to write.
        column_names {list of M str} -- list of all column names.
        values {list [M][N]} -- list of all column values. First index corresponds to
            column number.
    '''
    with open(filename, 'w', newline='') as f:
        wrr = csv.writer(f)

        wrr.writerow(column_names)

        for itrial in range(len(values[0])):
            lo = [values[k][itrial] for k in range(len(column_names))]
            wrr.writerow(lo)


############################### MOT - OpenSim motion files
def import_mot(fname):
    '''Import OpenSim motion file into a python structure.

    Arguments:
        fname {str} -- motion file.

    Returns a list:
        dof_names {list of str} -- names of DOFs.
        times {list of numbers} -- time series.
        dofs {list} -- each item corresponds to values for that DOF for each frame.
            dofs[iDOF][iTime]
    '''
    with open(fname, 'r') as f:
        rdr = csv.reader(f, dialect='excel-tab')

        l = next(rdr)
        while len(l) < 1 or not l[0].strip().lower() == 'time':
            l = next(rdr)

        dof_names = [i.strip() for i in l[1:]]

        times = []
        dofs = [[] for _ in dof_names]

        for li in rdr:
            times.append(float(li[0]))
            for idof, vdof in enumerate(li[1:]):
                dofs[idof].append(float(vdof))
    return dof_names, times, dofs


def export_mot(fname, dof_names, times, dofs):
    '''Exports python structures into a motion file for OpenSim.

    Arguments:
        fname {str} -- filename of the mot file to output into.
        dof_names {list of str} -- each element is the DOF string name.
        times {list of numbers} -- time series.
        dofs {list} -- each item corresponds to values for that DOF for each frame.
    '''
    with open(fname, 'w', newline='') as f:
        wrr = csv.writer(f, dialect='excel-tab')

        wrr.writerow(['Coordinates'])
        wrr.writerow(['version=1'])
        wrr.writerow(['nRows={}'.format(len(times))])
        wrr.writerow(['nColumns={}'.format(len(dof_names)+1)])
        wrr.writerow(['inDegrees=yes'])
        wrr.writerow([])
        wrr.writerow(['Units are S.I. units (second, meters, Newtons, ...)'])
        wrr.writerow(['Angles are in degrees.'])
        wrr.writerow([])
        wrr.writerow(['endheader'])
        wrr.writerow(['time'] + dof_names)

        for itime, time in enumerate(times):
            wrr.writerow([time] + [dof_vals[itime] for dof_vals in dofs])


############################### TRC - OpenSim joint angle traces
def import_trc(filename):
    '''Import OpenSim trc file into a Python structure format.

    Arguments:
        filename {str} -- trc file name.
    Output:
        bodyparts {list of str} -- names of markers.
        frame_numbers {list of ints} -- Frame # column.
        times {list of floats} -- Time column
        points {array NFrames X NBodyparts X 3} -- [iframe][ibodypart][0:x,1:y,2:z]
        rate {float} -- DataRate.
        units {str} - units of the data.
    '''
    with open(filename, 'r') as fin:
        rdr = csv.reader(fin, delimiter='\t', dialect='excel-tab')

        li = next(rdr)  # flavor text
        li = next(rdr)  # flavor text

        li = next(rdr)
        if not li[0] == li[1] or not li[0] == li[5]:
            warnings.warn('DataRate, CameraRate or OrigDataRate do not match. Using DataRate.')
        rate = float(li[0])
        units = li[4]

        li = next(rdr)
        bodyparts = li[slice(2, len(li), 3)]

        li = next(rdr)
        li = next(rdr)

        frame_numbers = []
        times = []
        points = []
        for li in rdr:
            if len(li) == 0:
                continue
            frame_numbers.append(int(li[0]))
            times.append(float(li[1]))
            points.append([])
            for ibp in range(len(bodyparts)):
                points[-1].append([])
                if li[2+ibp*3] == '':
                    points[-1][-1].append(math.nan)
                    points[-1][-1].append(math.nan)
                    points[-1][-1].append(math.nan)
                else:
                    points[-1][-1].append(float(li[2+ibp*3]))
                    points[-1][-1].append(float(li[2+ibp*3+1]))
                    points[-1][-1].append(float(li[2+ibp*3+2]))
    return bodyparts, frame_numbers, times, points, rate, units


def export_trc(filename, bodyparts, points, rate, frame_numbers=None, times=None, units='mm'):
    '''Exports marker data into OpenSim-compatible trc file.

    Arguments:
        filename {str} -- output file name.
        bodyparts {list of str} -- names of markers.
        points {array NFrames X NBodyparts X 3} -- [iframe][ibodypart][0:x,1:y,2:z]
        rate {float} -- DataRate.
    Keyword Arguments:
        frame_numbers {list of ints} -- Frame # column. If None, generated from rate and length of
            points starting at 1.
        times {list of floats} -- Time column. If None, generated from rate and length of
            points starting at 0.
        units {str} - units of the data. {default: 'mm'}
    '''
    if frame_numbers is None:
        frame_numbers = list(range(1, len(points) + 1))
    if times is None:
        period = 1. / rate
        times = [i * period for i in range(len(frame_numbers))]

    n_bodyparts = len(bodyparts)
    n_frames = len(frame_numbers)

    with open(filename, 'w', newline='') as fou:
        wrr = csv.writer(fou, delimiter='\t', dialect='excel-tab')

        # header
        wrr.writerow(['PathFileType', '4', '(X/Y/Z)', ntpath.basename(filename)])
        wrr.writerow(['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units', 'OrigDataRate',
                      'OrigDataStartFrame', 'OrigNumFrames'])
        wrr.writerow([rate, rate, n_frames, n_bodyparts, units, rate, 1, 1])

        # bodyparts
        lo = ['Frame#', 'Time']
        for bp in bodyparts:
            lo += [bp, '', '']
        wrr.writerow(lo)

        # XYZ columns
        lo = ['', '']
        for ibp in range(n_bodyparts):
            lo += ['X{}'.format(ibp+1), 'Y{}'.format(ibp+1), 'Z{}'.format(ibp+1)]
        wrr.writerow(lo)
        wrr.writerow([])  # necessary

        # data
        for frame_number, time, point in zip(frame_numbers, times, points):
            lo = [frame_number, time]
            for ibp in range(n_bodyparts):
                if math.isnan(point[ibp][0]):
                    lo += ['', '', '']
                else:
                    lo += point[ibp]

            # OpenSim4.0 cannot read the line properly when the last value is
            # empty and wants an additional tab:
            if lo[-1] == '':
                lo.append('')

            wrr.writerow(lo)


############################### 3D points in CSV
def import_triangulated_csv(filename):
    '''Returns data as dictionary:
        bodypart: nFrames X 3
    '''
    data = {}
    with open(filename, 'r') as f:
        rdr = csv.reader(f)

        li1 = next(rdr)
        li2 = next(rdr)

        # read the csv
        frame_numbers = []
        data_raw = [[] for _ in range(len(li1) - 1)]
        for li in rdr:
            frame_numbers.append(int(li[0]))
            for i, el in enumerate(li[1:]):
                data_raw[i].append(float(el))

    # transform into a dictionary
    for i, (i1, i2) in enumerate(zip(li1[1:], li2[1:])):
        v = data.get(i1, [])
        v.append(data_raw[i])
        data[i1] = v

    # transpose the dictionary
    for k in data.keys():
        data[k] = list(zip(*data[k]))

    return frame_numbers, data
