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
import xml.etree.ElementTree as ET


IK_XML_STR = r'''<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <InverseKinematicsTool>
        <!--Directory used for writing results.-->
        <results_directory>./</results_directory>
        <!--Directory for input files-->
        <input_directory />
        <!--Name of the model file (.osim) to use for inverse kinematics.-->
        <model_file>Unassigned</model_file>
        <!--A positive scalar that weights the relative importance of satisfying constraints. A weighting of 'Infinity' (the default) results in the constraints being strictly enforced. Otherwise, the weighted-squared constraint errors are appended to the cost function.-->
        <constraint_weight>Inf</constraint_weight>
        <!--The accuracy of the solution in absolute terms. Default is 1e-5. It determines the number of significant digits to which the solution can be trusted.-->
        <accuracy>1.0000000000000001e-05</accuracy>
        <!--Markers and coordinates to be considered (tasks) and their weightings. The sum of weighted-squared task errors composes the cost function.-->
        <IKTaskSet>
            <objects />
            <groups />
        </IKTaskSet>
        <!--TRC file (.trc) containing the time history of observations of marker positions obtained during a motion capture experiment. Markers in this file that have a corresponding task and model marker are included.-->
        <marker_file>Unassigned</marker_file>
        <!--The name of the storage (.sto or .mot) file containing the time history of coordinate observations. Coordinate values from this file are included if there is a corresponding model coordinate and task. -->
        <coordinate_file>Unassigned</coordinate_file>
        <!--The desired time range over which inverse kinematics is solved. The closest start and final times from the provided observations are used to specify the actual time range to be processed.-->
        <time_range> 0 1</time_range>
        <!--Flag (true or false) indicating whether or not to report marker errors from the inverse kinematics solution.-->
        <report_errors>true</report_errors>
        <!--Name of the resulting inverse kinematics motion (.mot) file.-->
        <output_motion_file>out_inv_kin.mot</output_motion_file>
        <!--Flag indicating whether or not to report model marker locations. Note, model marker locations are expressed in Ground.-->
        <report_marker_locations>false</report_marker_locations>
    </InverseKinematicsTool>
</OpenSimDocument>
'''


def triangulated_to_trc(triang_csv, trc_file, marker_name_dict, data_unit_convert=None,
                        rate=50, repeat=0, zero_marker='scapula_anterior', frame_range=None,
                        runtime_data_check=None, rotation=None,
                        ik_file=None, ik_weight_type='nans',
                        ik_xml_str=None, ik_out_mot_file='out_inv_kin_mot'):
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
        rotation {function} -- is applied to every point (x,y,z). Is supposed to accept a list with
            3 numbers (vector in NCams coordinate system) and return a list with three numbers
            (vector in OSim coordinate system). (default: {returns same vector})
        ik_file {str or None} -- makes a config file to run inverse kinematics in OSim. If None, the
            it is not created. (default: None)
        ik_weight_type {'nans', 'ones', 'likelihood'} -- an algorithm to pick a weight for each
            marker:
            'nans' -- weight for a marker equals to 1 - portion of points where it was NaN.
            'ones' -- all weights are 1.
            'likelihood' -- not implemented.
            (default: 'nans')
        ik_xml_str {str} -- XML structure of the output inverse kinematic file. See
            ncams.inverse_kinematics.IK_XML_STR for an example of input. (default:
            ncams.inverse_kinematics.IK_XML_STR)
        ik_out_mot_file {str} --  filename of the output inverse kinematics file.
            {default: 'out_inv_kin_mot'}
    '''
    if data_unit_convert is None:
        data_unit_convert = lambda x: x*100  # dm to mm
    period = 1./rate
    if rotation is None:
        rotation = lambda x: x

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
        # skip the first frame until the desired frame_range
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
                    value_dict[bp] = rotation([[data_unit_convert(float(li[ix])-zero_x),
                                                data_unit_convert(float(li[iy])-zero_y),
                                                data_unit_convert(float(li[iz])-zero_z)]])
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
        time_range = [0, (i-1)*period]

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

    # make inverse kinematic config file for OSim
    if ik_file is not None:
        if ik_xml_str is None:
            ik_xml_str = IK_XML_STR
        print('Making IK file {}'.format(ik_file))
        root = ET.fromstring(ik_xml_str)
        if root.tag != 'OpenSimDocument':
            raise ValueError('Wrong structure of the IK string. OpenSimDocument is not present at '
                             'top-level.')

        ikt = root.find('InverseKinematicsTool')
        if ikt is None:
            # default structure of the IKT, like in the IK_XML_STR
            ikt = ET.Element('InverseKinematicsTool')
            ikt.append(ET.Element('results_directory', text='./'))
            ikt.append(ET.Element('input_directory'))
            ikt.append(ET.Element('model_file', text='Unassigned'))
            ikt.append(ET.Element('constraint_weight', text='Inf'))
            ikt.append(ET.Element('accuracy', text='1.e-05'))
            ikt.append(ET.Element('coordinate_file', text='Unassigned'))
            ikt.append(ET.Element('report_errors', text='true'))
            ikt.append(ET.Element('report_marker_locations', text='false'))
            root.append(ikt)

        ikts = ikt.find('IKTaskSet')
        if ikts is None:
            ikts = ET.Element('IKTaskSet')
            ikts.append(ET.Element('groups'))

        iktso = ikts.find('objects')
        if iktso is None:
            iktso = ET.Element('objects')
            ikts.append(iktso)

        if iktso.text is None or len(iktso.text) == 0:
            iktso.text = '\n' + ' '*16

        if iktso.tail is None or len(iktso.tail) == 0:
            pass
        iktso.tail = '\n' + ' '*12

        for bp in bodyparts:
            bpe = ET.Element('IKMarkerTask')
            bpe.set('name', marker_name_dict[bp])
            bpe.text = '\n' + ' '*20
            bpe.tail = '\n' + ' '*16

            # calculate the weight of the marker
            if ik_weight_type == 'nans':
                bp_weight = num_dats[bp]/n_frames
            elif ik_weight_type == 'ones':
                bp_weight = 1
            elif ik_weight_type == 'likelihood':
                raise NotImplementedError('Using likelihood for estimation of the marker weight.')

            if bp_weight < 1e-8:
                bp_apply = 'false'
            else:
                bp_apply = 'true'

            bpe_a = ET.Element('apply')
            bpe_a.text = bp_apply
            bpe_a.tail = '\n' + ' '*20
            bpe.append(bpe_a)
            bpe_w = ET.Element('weight')
            bpe_w.text = str(bp_weight)
            bpe_w.tail = '\n' + ' '*16
            bpe.append(bpe_w)

            iktso.append(bpe)

        ikt_tr = ikt.find('time_range')
        if ikt_tr is None:
            ikt_tr = ET.Element('time_range')
            ikt.append(ikt_tr)
        ikt_tr.text = '{} {}'.format(time_range[0], time_range[1])

        ikt_mf = ikt.find('marker_file')
        if ikt_mf is None:
            ikt_mf = ET.Element('marker_file')
            ikt.append(ikt_mf)
        ikt_mf.text = trc_file

        ikt_omf = ikt.find('output_motion_file')
        if ikt_omf is None:
            ikt_omf = ET.Element('output_motion_file')
            ikt.append(ikt_omf)
        ikt_omf.text = ik_out_mot_file

        tree = ET.ElementTree(element=root)
        tree.write(ik_file, encoding='UTF-8', xml_declaration=True)


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
