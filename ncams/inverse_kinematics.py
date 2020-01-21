#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Functions related to setting up and analyzing inverse kinematics using OpenSim (SimTK).
"""
import ntpath
import csv
import math


def triangulated_to_trc(triang_csv, trc_file, marker_name_dict, data_unit_convert=None,
                        rate=50, repeat=0, zero_marker='scapula_anterior', frame_range=None,
                        runtime_data_check=None):
    '''Transforms triangulated data from NCams/DLC format into OpenSim trc.

    Arguments:
        triang_csv {string} -- get the triangulated data from this file.
        trc_file {string} -- output filename.
        marker_name_dict {dict} -- dictionary relating names of markers in triangulated file to the
            names in the output trc file.

    Keyword Arguments:
        data_unit_convert {lambda x} -- transform values in units. OSim usually expects meters.
            By default, transforms decimeters into m. (default: {lambda x: x*100})
        rate {number} -- framerate of the data. (default: {50})
        repeat {number} -- add copies of the datapoints to the end of the output file. Useful when
            OSim has a problem visualising too few points. (default: {0})
        zero_marker {str or None} -- shift all data so that the marker with this name is (0,0,0) at
            frame 0. If None, no shift of the data is happening (default: {'scapula_anterior'})
        frame_range {2-list of numbers or None} -- frame range to export from the file. If a tuple
            then indicates the start and end frame number, including both as an interval. If None
            then all frames will be used. If frame_range[1] is None, continue until the last frame.
            (default: None)
        runtime_data_check {function} -- print custom information about the data while it is being
            transferred. The function should accept the following positional arguments:
            (frame_number, value_dict)
                frame_number {int} -- frame number id of the datum being processed.
                value_dict {dict} -- dictionary relating the maker name in NCams/DLC style to the
                    marker locations (x, y, z) in units after the data_unit_convert.
            (default: {pass})
    '''
    if data_unit_convert is None:
        data_unit_convert = lambda x: x*100  # dm to mm
    period = 1./rate

    if frame_range is None:
        with open(triang_csv, 'r') as f:
            n_frames = len(f.readlines()) - 2
    elif frame_range[1] is None:
        with open(triang_csv, 'r') as f:
            n_frames = len(f.readlines()) - 2 - frame_range[0]
    else:
        n_frames = frame_range[1] - frame_range[0] + 1

    with open(triang_csv, 'r') as fin, open(trc_file, 'w', newline='') as fou:
        rdr = csv.reader(fin)
        wrr = csv.writer(fou, delimiter='\t', dialect='excel-tab')

        li = next(rdr)
        n_bodyparts = int((len(li)-1)/3)
        bp_xyz_indcs = []
        bodyparts = []
        for i in range(n_bodyparts):
            if li[1+i*3] in marker_name_dict.keys():
                bodyparts.append(li[1+i*3])
                bp_xyz_indcs.append([1+i*3, 1+i*3+1, 1+i*3+2])
        n_bodyparts = len(bodyparts)

        wrr.writerow(['PathFileType', '4', '(X/Y/Z)', ntpath.basename(trc_file)])
        wrr.writerow(['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units', 'OrigDataRate',
                      'OrigDataStartFrame', 'OrigNumFrames'])
        wrr.writerow([rate, rate, n_frames*(repeat+1), n_bodyparts, 'mm', rate, 1, 1])
        lo = ['Frame#', 'Time']
        for bp in bodyparts:
            lo += [marker_name_dict[bp], '', '']
        wrr.writerow(lo)

        lo = ['', '']
        for ibp in range(n_bodyparts):
            lo += ['X{}'.format(ibp+1), 'Y{}'.format(ibp+1), 'Z{}'.format(ibp+1)]
        wrr.writerow(lo)
        wrr.writerow([])

        zero_index = None
        if repeat > 0:
            data_copy = []

        next(rdr)
        if frame_range is not None and frame_range[0] > 0:
            while int(next(rdr)[0]) < frame_range[0] - 1:
                pass

        num_dats = dict(zip(bodyparts, [0]*n_bodyparts))

        for i, li in enumerate(rdr):
            # when to stop based on frame_range
            if (frame_range is not None and frame_range[1] is not None and
                    int(li[0]) > frame_range[1]):
                break

            # first iteration, set up zeros
            if zero_index is None:
                if zero_marker is None:
                    zero_index = -1
                    zero_x = 0
                    zero_y = 0
                    zero_z = 0
                else:
                    zero_index = bodyparts.index(zero_marker)
                    zero_x = float(li[bp_xyz_indcs[zero_index][0]])
                    zero_y = float(li[bp_xyz_indcs[zero_index][1]])
                    zero_z = float(li[bp_xyz_indcs[zero_index][2]])

            # set up the dictionary for all values
            value_dict = {}
            for bp, (ix, iy, iz) in zip(bodyparts, bp_xyz_indcs):
                if 'nan' in (li[ix].lower(), li[iy].lower(), li[iz].lower()):
                    value_dict[bp] = ['', '', '']
                else:
                    value_dict[bp] = [data_unit_convert(float(li[ix])-zero_x),
                                      data_unit_convert(float(li[iy])-zero_y),
                                      data_unit_convert(float(li[iz])-zero_z)]
                    num_dats[bp] += 1

            lo = [i+1, i*period]
            if repeat > 0:
                data_copy.append([])
            for bp in bodyparts:
                lo += value_dict[bp]
                if repeat > 0:
                    data_copy[-1] += value_dict[bp]

            wrr.writerow(lo)

            # print a runtime report
            if runtime_data_check is not None:
                runtime_data_check(i, value_dict)

        print('Portion of the data being data and not NaNs:')
        print('\n'.join('\t{}: {:.3f}'.format(marker_name_dict[bp], num_dats[bp]/n_frames)
                        for bp in bodyparts))

        # add copies
        frame_start_copy = i+2
        for i in range(repeat):
            for j, dc in enumerate(data_copy):
                frame_n = frame_start_copy+i*len(data_copy)+j
                lo = [frame_n, (frame_n-1)*period]
                lo += dc
                wrr.writerow(lo)
        if repeat > 0:
            print('Added {} copies of data.'.format(repeat))


def rdc_touchpad3d(frame_number, value_dict, dist_desired=(105, 79, 105, 79), warn_threshold=5,
                   marker_name_dict=None):
    '''Runtime data check example, checks if TouchPad3D markers are present and measures deviation
    from the physical measurements of the TouchPad3D.

    Arguments:
        frame_number {int} -- frame number id of the datum being processed.
        value_dict {dict} -- dictionary relating the maker name in NCams/DLC style to the marker
            locations (x, y, z) in units after the data_unit_convert.

    Keyword Arguments:
        dist_desired {tuple} -- measurements of each side (left-front, front-right, roght-back,
            back-left) of the touchpad. (default: {(105, 79, 105, 79) mm})
        warn_threshold {number} -- threshold in mm for warning a user if the touchpad dimensions
            seem wrong. (default: {5} mm)
        marker_name_dict {dict} -- dictionary relating DLC/NCams names of markers to the OpenSim
            names. (default: {None})
    '''
    lf_desired, fr_desired, rb_desired, bl_desired = dist_desired
    def dist_f(fs, ss):
        if fs not in value_dict.keys() or ss not in value_dict.keys():
            return math.nan
        if len(value_dict[fs][0]) == 0 or len(value_dict[ss][0]) == 0:
            return math.nan
        return ((value_dict[fs][1] - value_dict[ss][1])**2 +
                (value_dict[fs][2] - value_dict[ss][2])**2 +
                (value_dict[fs][3] - value_dict[ss][3])**2) ** 0.5

    if marker_name_dict is None:
        marker_name_dict = dict(zip(('tp_left_c', 'tp_front_c', 'tp_right_c', 'tp_back_c'),
                                    ('tp_left_c', 'tp_front_c', 'tp_right_c', 'tp_back_c')))

    lf = dist_f('tp_left_c', 'tp_front_c')
    fr = dist_f('tp_front_c', 'tp_right_c')
    rb = dist_f('tp_right_c', 'tp_back_c')
    bl = dist_f('tp_back_c', 'tp_left_c')
    warnings = ''
    if abs(lf-lf_desired) > warn_threshold and abs(fr-fr_desired) > warn_threshold:
        warnings += ' Recommend dropping {} marker from IK.'.format(
            marker_name_dict['tp_front_c'])
    if abs(fr-fr_desired) > warn_threshold and abs(rb-rb_desired) > warn_threshold:
        warnings += ' Recommend dropping {} marker from IK.'.format(
            marker_name_dict['tp_right_c'])
    if abs(rb-rb_desired) > warn_threshold and abs(bl-bl_desired) > warn_threshold:
        warnings += ' Recommend dropping {} marker from IK.'.format(
            marker_name_dict['tp_back_c'])
    if abs(bl-bl_desired) > warn_threshold and abs(lf-lf_desired) > warn_threshold:
        warnings += ' Recommend dropping {} marker from IK.'.format(
            marker_name_dict['tp_left_c'])
    print('Frame {} distances: lf: {} mm, fr: {} mm, rb: {} mm, bl: {} mm.{}'.format(
        frame_number, lf, fr, rb, bl, warnings))
