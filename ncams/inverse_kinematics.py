#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019-2022 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Functions related to setting up and analyzing inverse kinematics using OpenSim (SimTK).
"""
import math
from copy import deepcopy
import xml.etree.ElementTree as ET
import warnings

import numpy as np
import scipy.signal

from . import io_utils


# Default accuracy of 1e-5 does not produce precise enough results for hand and finger movements.
IK_XML_STR = '''\
<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <InverseKinematicsTool>
        <!--Directory used for writing results.-->
        <results_directory>./</results_directory>
        <!--Directory for input files-->
        <input_directory />
        <!--Name of the model file (.osim) to use for inverse kinematics.-->
        <model_file>{model_file}</model_file>
        <!--A positive scalar that weights the relative importance of satisfying constraints. A weighting of 'Infinity' (the default) results in the constraints being strictly enforced. Otherwise, the weighted-squared constraint errors are appended to the cost function.-->
        <constraint_weight>Inf</constraint_weight>
        <!--The accuracy of the solution in absolute terms. Default is 1e-5. It determines the number of significant digits to which the solution can be trusted.-->
        <accuracy>1.0000000000000001e-06</accuracy>
        <adaptiveAccuracy>true</adaptiveAccuracy>
        <ignoreConvergenceErrors>true</ignoreConvergenceErrors>
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

SC_XML_STR = '''\
<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <ScaleTool name="{tool_name}">
        <!--Mass of the subject in kg.  Subject-specific model generated by scaling step will have this total mass.-->
        <mass>0</mass>
        <!--Height of the subject in mm.  For informational purposes only (not used by scaling).-->
        <height>-1</height>
        <!--Age of the subject in years.  For informational purposes only (not used by scaling).-->
        <age>-1</age>
        <!--Notes for the subject.-->
        <notes>Unassigned</notes>
        <!--Specifies the name of the unscaled model (.osim) and the marker set.-->
        <GenericModelMaker>
            <!--Model file (.osim) for the unscaled model.-->
            <model_file>Unassigned</model_file>
            <!--Set of model markers used to scale the model. Scaling is done based on distances between model markers compared to the same distances between the corresponding experimental markers.-->
            <marker_set_file>Unassigned</marker_set_file>
        </GenericModelMaker>
        <!--Specifies parameters for scaling the model.-->
        <ModelScaler>
            <!--Whether or not to use the model scaler during scale-->
            <apply>true</apply>
            <!--Specifies the scaling method and order. Valid options are 'measurements', 'manualScale', singly or both in any sequence.-->
            <scaling_order> measurements</scaling_order>
            <!--Specifies the measurements by which body segments are to be scaled.-->
            <MeasurementSet>
                <objects />
                <groups />
            </MeasurementSet>
            <!--Scale factors to be used for manual scaling.-->
            <ScaleSet>
                <objects />
                <groups />
            </ScaleSet>
            <!--TRC file (.trc) containing the marker positions used for measurement-based scaling. This is usually a static trial, but doesn't need to be.  The marker-pair distances are computed for each time step in the TRC file and averaged across the time range.-->
            <marker_file>{marker_file}</marker_file>
            <!--Time range over which to average marker-pair distances in the marker file (.trc) for measurement-based scaling.-->
            <time_range>{time_range}</time_range>
            <!--Flag (true or false) indicating whether or not to preserve relative mass between segments.-->
            <preserve_mass_distribution>false</preserve_mass_distribution>
            <!--Name of OpenSim model file (.osim) to write when done scaling.-->
            <output_model_file>Unassigned</output_model_file>
            <!--Name of file to write containing the scale factors that were applied to the unscaled model (optional).-->
            <output_scale_file>Unassigned</output_scale_file>
        </ModelScaler>
        <!--Specifies parameters for placing markers on the model once a model is scaled. -->
        <MarkerPlacer>
            <!--Whether or not to use the marker placer during scale-->
            <apply>false</apply>
            <!--Task set used to specify weights used in the IK computation of the static pose.-->
            <IKTaskSet>
                <objects />
                <groups />
            </IKTaskSet>
            <!--TRC file (.trc) containing the time history of experimental marker positions (usually a static trial).-->
            <marker_file />
            <!--Name of file containing the joint angles used to set the initial configuration of the model for the purpose of placing the markers. These coordinate values can also be included in the optimization problem used to place the markers. Before the model markers are placed, a single frame of an inverse kinematics (IK) problem is solved. The IK problem can be solved simply by matching marker positions, but if the model markers are not in the correct locations, the IK solution will not be very good and neither will marker placement. Alternatively, coordinate values (specified in this file) can be specified and used to influence the IK solution. This is valuable particularly if you have high confidence in the coordinate values. For example, you know for the static trial the subject was standing will all joint angles close to zero. If the coordinate set (see the CoordinateSet property) contains non-zero weights for coordinates, the IK solution will try to match not only the marker positions, but also the coordinates in this file. Least-squared error is used to solve the IK problem. -->
            <coordinate_file>Unassigned</coordinate_file>
            <!--Time range over which the marker positions are averaged.-->
            <time_range> -1 -1</time_range>
            <!--Name of the motion file (.mot) written after marker relocation (optional).-->
            <output_motion_file>Unassigned</output_motion_file>
            <!--Output OpenSim model file (.osim) after scaling and maker placement.-->
            <output_model_file>Unassigned</output_model_file>
            <!--Output marker set containing the new marker locations after markers have been placed.-->
            <output_marker_file>Unassigned</output_marker_file>
            <!--Maximum amount of movement allowed in marker data when averaging frames of the static trial. A negative value means there is not limit.-->
            <max_marker_movement>-1</max_marker_movement>
        </MarkerPlacer>
    </ScaleTool>
</OpenSimDocument>
'''

SC_EMPTY_MEASUREMENT = '''
                    <Measurement name="{name}">
                        <apply>true</apply>
                        <MarkerPairSet>
                            <objects />
                            <groups />
                        </MarkerPairSet>
                        <BodyScaleSet>
                            <objects>
{body_scales}
                            </objects>
                            <groups />
                        </BodyScaleSet>
                    </Measurement>
'''
SC_BODY_SCALE = '''
                                <BodyScale name="{}">
                                    <axes> X Y Z</axes>
                                </BodyScale>
'''


def triangulated_to_trc(triang_csv, trc_file, marker_name_dict, data_unit_convert=None,
                        rate=50, zero_marker=None, frame_range=None,
                        rotation=None, verbose=0, reflect=False):
    '''Transforms triangulated data from NCams/DLC format into OpenSim trc.

    Arguments:
        triang_csv {string} -- get the triangulated data from this file.
        trc_file {string} -- output filename.
        marker_name_dict {dict} -- dictionary relating names of markers in triangulated file to the
            names in the output trc file.

    Keyword Arguments:
        data_unit_convert {lambda x} -- transform values in units, e.g., from meters to decimeters:
            x*100. OSim usually expects meters. Should be applicable to the numpy matrix of the
            data. No transform by default.
        rate {number} -- framerate of the data. (default: {50})
        zero_marker {str or None} -- shift all data so that the marker with this name is (0,0,0) at
            frame 0. If None, no shift of the data is happening (default: None)
        frame_range {2-list of numbers or None} -- frame range to export from the file. If a tuple
            then indicates the start and end frame number, including both as an interval. If None
            then all frames will be used. If frame_range[1] is None, continue until the last frame.
            Note that frame# starts with 0. The output trc file requires frames to start from 1.
            (default: None)
        rotation {function} -- is applied to every point (x,y,z). Is supposed to accept a (P, 3)
            vectors (vector in NCams coordinate system) and return a list of (P, 3)
            (in OSim coordinate system). See scipy.spatial.transform.rotation.Rotation.apply for
            reference. No rotation by default.
        verbose {int} -- verbosity level. Higher verbosity prints more output {default: 0}.
        reflect {bool} -- reflects the data along an axis (x = -x). Needed when processing data
            from a left-handed experiment to right-handed model. {default: False}
    '''
    # import triangulated file
    # frame numbers are only to take the subset using frame_range
    frame_numbers, triang_data = io_utils.import_triangulated_csv(triang_csv)

    # change into numpy arrays
    frame_numbers = np.array(frame_numbers)
    for k in triang_data.keys():
        triang_data[k] = np.array(triang_data[k])

    # trim the range based on frame_range and estimate number of frames
    if frame_range is not None:
        if frame_range[1] is None:
            frame_mask = frame_numbers >= frame_range[0]
        else:
            frame_mask = frame_numbers >= frame_range[0] & frame_numbers <= frame_range[1]
        # trim
        frame_numbers = frame_numbers[frame_mask]
        for k in triang_data.keys():
            triang_data[k] = triang_data[k][frame_mask, :]

    # collect the data based on the marker name dictionary
    bodyparts = []
    marker_names = []  # OpenSim names
    points = []
    for td_k, td in triang_data.items():
        if td_k in marker_name_dict.keys():
            bodyparts.append(td_k)
            marker_names.append(marker_name_dict[td_k])
            points.append(td)
    points = np.array(points)
    # change the indices from NBodyparts X NFrames X 3 to NFrames X NBodyparts X 3
    points = np.swapaxes(points, 0, 1)

    # transform the data
    # remove the 0 point
    if zero_marker is not None:
        # just break it if it is not there
        zero_index = bodyparts.index(zero_marker)
        zero_xyz = points[0, zero_marker, :]
        if math.isnan(zero_xyz[0]):
            warnings.warn('Zero marker is NaN. All data will be NaNs.')
        points = points - zero_xyz

    # convert
    if data_unit_convert is not None:
        points = data_unit_convert(points)

    # rotate
    if rotation is not None:
        for ibp, _ in enumerate(bodyparts):
            points[:, ibp, :] = rotation(points[:, ibp, :])

    # reflect
    if reflect:
        points[:, :, 0] = - points[:, :, 0]

    # calculate how much data is nans
    n_frames = len(points)
    num_dats = {}
    for ibp, bp in enumerate(bodyparts):
        num_dats[bp] = n_frames - np.sum(np.isnan(points[:, ibp, 0]))
    marker_weights = {marker_name_dict[bp]: num_dats[bp]/n_frames for bp in bodyparts}
    if verbose > 0:
        print('Portion of the data being data and not NaNs:')
        print('\n'.join('\t{}: {:.3f}'.format(marker_name, marker_weight)
                        for marker_name, marker_weight in marker_weights.items()))

    # export
    io_utils.export_trc(trc_file, marker_names, points.tolist(), rate)

    # estimate time_range
    period = 1./rate
    time_range = [0, period * len(points)]

    # return variables for creation of IK files
    return marker_weights, time_range


def _add_xml_element(parent, name, text=None, tail=None, unique=True):
    '''Adds an element to the parent. If unique, checks if such element already exists in the parent
    and does not add it if it does. Otherwise always adds.
    '''
    if unique:
        el = parent.find(name)
    if not unique or el is None:
        el = ET.Element(name)
        parent.append(el)
    if text is not None:
        el.text = text
    if tail is not None:
        el.tail = tail
    return el


def make_ik_file(filename, ik_xml_str, marker_weights, trc_file, ik_out_mot_file, time_range,
                 verbose=0):
    if ik_xml_str is None:
        ik_xml_str = IK_XML_STR.format(model_file="Unassigned")
    if verbose > 0:
        print('Making IK file {}'.format(filename))

    # check the basic elements
    root = ET.fromstring(ik_xml_str)
    if root.tag != 'OpenSimDocument':
        raise ValueError('Wrong structure of the IK string. OpenSimDocument is not present at '
                         'top-level.')

    ikt = root.find('InverseKinematicsTool')
    if ikt is None:
        raise ValueError('Wrong structure of the IK string. InverseKinematicsTool is not present.')

    # add the IK task and objects (markers) if missing
    ikts = _add_xml_element(ikt, 'IKTaskSet')
    _add_xml_element(ikts, 'groups')
    iktso = _add_xml_element(ikts, 'objects', text='\n' + ' '*16, tail='\n' + ' '*12)

    # add each marker with weights
    for marker_name, marker_weight in marker_weights.items():
        mare = _add_xml_element(iktso, 'IKMarkerTask',
                                text='\n' + ' '*20, tail='\n' + ' '*16,
                                unique=False)
        mare.set('name', marker_name)

        _add_xml_element(mare, 'weight', text=str(marker_weight), tail='\n' + ' '*20)

        if marker_weight < 1e-8:
            _add_xml_element(mare, 'apply', text='false', tail='\n' + ' '*16)
        else:
            _add_xml_element(mare, 'apply', text='true', tail='\n' + ' '*16)

    # other elements
    _add_xml_element(ikt, 'time_range', text='{} {}'.format(time_range[0], time_range[1]))
    _add_xml_element(ikt, 'marker_file', text=trc_file)
    _add_xml_element(ikt, 'output_motion_file', text=ik_out_mot_file)

    tree = ET.ElementTree(element=root)
    tree.write(filename, encoding='UTF-8', xml_declaration=True)


def make_sc_file(filename, tool_name, measurements, marker_file, time_range,
                 verbose=0):
    sc_xml_str = SC_XML_STR.format(
        tool_name=tool_name,
        time_range='{} {}'.format(time_range[0], time_range[1]),
        marker_file=marker_file)
    if verbose > 0:
        print('Making SC file {}'.format(filename))

    # check the basic elements
    root = ET.fromstring(sc_xml_str)
    sct = root.find('ScaleTool')

    # add the SC ModelScaler and objects (measurements) if missing
    ms = _add_xml_element(sct, 'ModelScaler')
    mset = _add_xml_element(ms, 'MeasurementSet')
    _add_xml_element(mset, 'groups')
    mseto = _add_xml_element(mset, 'objects', text='\n' + ' '*20, tail='\n' + ' '*16)

    for measurement, body_scales in measurements.items():
        sc_meas = SC_EMPTY_MEASUREMENT.format(
            name=measurement,
            body_scales=''.join([SC_BODY_SCALE.format(body_scale) for body_scale in body_scales]))
        meas = ET.fromstring(sc_meas)
        meas.tail = '\n' + ' ' * 20
        mseto.append(meas)
    meas.tail = '\n' + ' ' * 16

    tree = ET.ElementTree(element=root)
    tree.write(filename, encoding='UTF-8', xml_declaration=True)


def remove_empty_markers_from_trc(trc_filename):
    '''Removes markers from a trc file that do not have any data points.

    Arguments:
        trc_filename {str} -- filename to process.
    '''
    raise DeprecationWarning()
    bodyparts, frame_numbers, times, points, rate, units = io_utils.import_trc(trc_filename)

    for ibp in reversed(range(len(bodyparts))):
        if all([np.isnan(points[iframe][ibp][0]) for iframe in range(len(points))]):
            print('Removing bodypart {} from trc file {}.'.format(bodyparts[ibp], trc_filename))
            del bodyparts[ibp]
            for iframe in range(len(points)):
                del points[iframe][ibp]

    print('{} bodyparts left: {}.'.format(len(bodyparts), ', '.join(bodyparts)))

    io_utils.export_trc(trc_filename, bodyparts, frame_numbers, times, points, rate, units)


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
    raise DeprecationWarning()
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
    warning_s = ''
    if abs(lf-lf_desired) > warn_threshold and abs(fr-fr_desired) > warn_threshold:
        warning_s += ' Recommend dropping {} marker from IK.'.format(
            marker_name_dict['tp_front_c'])
    if abs(fr-fr_desired) > warn_threshold and abs(rb-rb_desired) > warn_threshold:
        warning_s += ' Recommend dropping {} marker from IK.'.format(
            marker_name_dict['tp_right_c'])
    if abs(rb-rb_desired) > warn_threshold and abs(bl-bl_desired) > warn_threshold:
        warning_s += ' Recommend dropping {} marker from IK.'.format(
            marker_name_dict['tp_back_c'])
    if abs(bl-bl_desired) > warn_threshold and abs(lf-lf_desired) > warn_threshold:
        warning_s += ' Recommend dropping {} marker from IK.'.format(
            marker_name_dict['tp_left_c'])
    print('Frame {} distances: lf: {} mm, fr: {} mm, rb: {} mm, bl: {} mm.{}'.format(
        frame_number, lf, fr, rb, bl, warning_s))


def smooth_motion(in_fname, ou_fname, median_kernel_size=11, ou_rate=None, filter_1d=None):
    '''Filters the motion from a file and saves it.

    Arguments:
        in_fname {str} -- filename with inverse kinematics motion to filter.
        ou_fname {str} -- filename for output of the filtered kinematics.

    Keyword Arguments:
        median_kernel_size {odd int} -- size of the kernel for median filter. Has to be odd.
            (default: 11)
        ou_rate {number} -- output rate. If not equal to in_rate, the signal is going to be
            resampled before median filter. (default: {same as input rate measured from input motion
            file})
        filter_1d {callable} -- custom filter to run on each DOF. Should be an executeable that
            accepts two arguments: time series, dof values. (default: {None})
    '''
    raise DeprecationWarning()
    # load
    dof_names, times, dofs = import_mot(in_fname)

    in_rate = np.mean(np.diff(times))
    if ou_rate is None:
        ou_rate = in_rate

    # resample
    if in_rate != ou_rate:
        num_ou = int(len(times)*ou_rate/in_rate)
        for idof in range(len(dofs)):
            dofs[idof] = scipy.signal.resample(dofs[idof], num_ou, t=times, window=None)[0]
        times = scipy.signal.resample(dofs[0], num_ou, t=times, window=None)[1]
        times = [t/ou_rate*in_rate for t in times]

    # median filter
    for idof in range(len(dofs)):
        dofs[idof] = scipy.signal.medfilt(dofs[idof], kernel_size=median_kernel_size)

    # custom filter
    if filter_1d is not None:
        for idof in range(len(dofs)):
            dofs[idof] = filter_1d(times, dofs[idof])

    # output
    export_mot(ou_fname, dof_names, times, dofs)


def set_opensim_model_default_position(osim_model_in, osim_model_ou, positions, lock=False):
    tree = ET.parse(osim_model_in)
    root = tree.getroot()

    for dof_name, position in positions.items():
        coordinate = root.find(".//Coordinate[@name='{}']".format(dof_name))
        c_defval = coordinate.find("default_value")
        c_defval.text = str(position)
        if lock:
            c_locked = coordinate.find("locked")
            c_locked.text = 'true'

    tree.write(osim_model_ou, encoding='UTF-8', xml_declaration=True)
