from scipy import io
import numpy as np
import pdb
import scipy.signal


import matplotlib.pyplot as plt


if __name__ == '__main__':
    pdb.set_trace()
    raw = io.loadmat(f"raw/ultrasound_data.mat")['ultrasound_data'][0]

    name_all = []
    v_all = []
    a_all = []

    pwv_all = []
    age_all = []
    comp_all = []
    z0_all = []
    pp_all = []
    bp_shape_all = []

    map_all = []
    map_alt_all = []

    resample_len_per_beat = 100
    id_all = []
    dbp_all = []
    heartrate_all = []
    area_all = []

    gender_all = []
    height_all = []
    weight_all = []
    diameter_complete_all = []
    diameter_complete_mean_all = []
    diameter_complete_min_all = []
    diameter_complete_max_all = []
    diameter_complete_time_frame_all = []
    diameter_complete_avg_all = []
    diameter_complete_avg_time_frame_all = []
    diameter_complete_avg_beats_all = []

    area_complete_avg_all = []
    area_complete_avg_beats_all = []
    area_complete_mean_all = []
    area_complete_min_all = []
    area_complete_max_all = []
    bp_shape_complete_avg_all = []
    bp_shape_complete_mean_all = []
    bp_shape_complete_min_all = []
    bp_shape_complete_max_all = []
    bp_shape_complete_avg_beats_all = []
    map_complete_avg_beats_all = []
    map_alt_complete_avg_beats_all = []

    velocity_complete_all = []
    velocity_complete_mean_all = []
    velocity_complete_min_all = []
    velocity_complete_max_all = []
    velocity_complete_time_frame_all = []
    velocity_complete_avg_all = []
    velocity_complete_avg_time_frame_all = []
    velocity_complete_avg_beats_all = []

    for subject_id in range(67):
        if subject_id == 45 or subject_id == 55 or subject_id == 60:
            continue
        print(subject_id)
        data = raw[subject_id][-1][0][0]
        id_all.append(subject_id)

        velocity = data[1].flatten()
        velocity = velocity[np.logical_not(np.isnan(velocity))]

        diameter = data[3].flatten()
        diameter = diameter[np.logical_not(np.isnan(diameter))]
        # x = scipy.signal.find_peaks(velocity)

        diameter = scipy.signal.resample(diameter, resample_len_per_beat)
        velocity = scipy.signal.resample(velocity, resample_len_per_beat)

        diameter_max_index = np.argmax(diameter)
        diameter = np.concatenate((diameter[diameter_max_index:], diameter[:diameter_max_index]))

        velocity_max_index = np.argmax(velocity)
        velocity = np.concatenate(
            (velocity[velocity_max_index:], velocity[:velocity_max_index]))

        name = raw[subject_id][0][0]

        data_supine_bp = raw[subject_id][3]
        SBP_avg = data_supine_bp[0][0][0][0][0][0].mean()
        DBP_avg = data_supine_bp[0][0][0][0][0][1].mean()
        ppressure = SBP_avg - DBP_avg
        heartrate_avg = data_supine_bp[0][0][0][0][0][2].mean()

        area = np.pi * np.square(diameter) / 4
        compliance = (max(area) - min(area)) / ppressure
        bp_shape = (area - area.mean()) / compliance

        map = DBP_avg + (area.mean() - min(area)) / (max(area) - min(area)) * (SBP_avg - DBP_avg)
        map_alt = (SBP_avg + 2*DBP_avg) / 3
        map_alt_all.append(map_alt)

        diameter_complete = raw[subject_id][1][0][0][0]
        diameter_complete_all.append(diameter_complete)

        diameter_complete_time_frame = raw[subject_id][1][0][0][2]
        diameter_complete_time_frame_all.append(diameter_complete_time_frame)

        diameter_complete_avg = np.nanmean(diameter_complete, axis=1)
        diam_avg_inds = np.logical_not(np.isnan(diameter_complete_avg))
        diameter_complete_avg = diameter_complete_avg[diam_avg_inds]
        diameter_complete_avg_all.append(diameter_complete_avg)
        diameter_complete_min_all.append(min(diameter_complete_avg))
        diameter_complete_mean_all.append(np.mean(diameter_complete_avg))
        diameter_complete_max_all.append(max(diameter_complete_avg))
        diameter_complete_avg_time_frame = diameter_complete_time_frame[diam_avg_inds]
        diameter_complete_avg_time_frame_all.append(diameter_complete_avg_time_frame)

        area_complete_avg = np.pi * np.square(diameter_complete_avg) / 4
        compliance = (max(area_complete_avg) - min(area_complete_avg)) / ppressure
        bp_shape_complete_avg = (area_complete_avg - area_complete_avg.mean()) / compliance
        area_complete_avg_all.append(area_complete_avg)
        area_complete_min_all.append(min(area_complete_avg))
        area_complete_mean_all.append(np.mean(area_complete_avg))
        area_complete_max_all.append(max(area_complete_avg))
        bp_shape_complete_avg_all.append(bp_shape_complete_avg)
        bp_shape_complete_min_all.append(min(bp_shape_complete_avg))
        bp_shape_complete_mean_all.append(np.mean(bp_shape_complete_avg))
        bp_shape_complete_max_all.append(max(bp_shape_complete_avg))

        diameter_complete_avg_beats = []
        # split into even individual beats based on avg heartrate measure
        num_beats = heartrate_avg/60 * (diameter_complete_avg_time_frame[-1] - diameter_complete_avg_time_frame[0])
        beat_len = int(len(diameter_complete_avg)/num_beats)
        for i in range(0, len(diameter_complete_avg)-beat_len + 1, beat_len):
            beat = diameter_complete_avg[i:i+beat_len]
            beat = scipy.signal.resample(beat, resample_len_per_beat)
            beat_max_index = np.argmax(beat)
            beat = np.concatenate((beat[beat_max_index:], beat[:beat_max_index]))
            diameter_complete_avg_beats.append(beat)

        area_complete_avg_beats = []
        # split into even individual beats based on avg heartrate measure - same beat len as in diameter waveform
        for i in range(0, len(area_complete_avg) - beat_len + 1, beat_len):
            beat = area_complete_avg[i:i + beat_len]
            beat = scipy.signal.resample(beat, resample_len_per_beat)
            beat_max_index = np.argmax(beat)
            beat = np.concatenate((beat[beat_max_index:], beat[:beat_max_index]))
            area_complete_avg_beats.append(beat)

        # # compute map for each beat calibrated on first beat
        # map_complete_avg_beats = []
        # first_beat_min, first_beat_max = min(area_complete_avg_beats[0]), max(area_complete_avg_beats[0])
        # for area_beat in area_complete_avg_beats:
        #     map = DBP_avg + (area_beat.mean() - first_beat_min) / (first_beat_max - first_beat_min) * (SBP_avg - DBP_avg)
        #     map_complete_avg_beats.append(map)

        # compute map for each beat calibrated on first beat
        map_complete_avg_beats = []
        map_alt_complete_avg_beats = []
        first_beat_min, first_beat_mean = min(area_complete_avg_beats[0]), area_complete_avg_beats[0].mean()

        area_beat_means = []
        for area_beat in area_complete_avg_beats:
            area_beat_means.append(area_beat.mean())
            map_beat = (((map - DBP_avg) * area_beat.mean() + (first_beat_mean*DBP_avg - first_beat_min*map))
                        / (first_beat_mean-first_beat_min))
            map_alt_beat = (((map_alt - DBP_avg) * area_beat.mean() + (first_beat_mean*DBP_avg - first_beat_min*map_alt))
                            / (first_beat_mean-first_beat_min))
            map_complete_avg_beats.append(map_beat)
            map_alt_complete_avg_beats.append(map_alt_beat)
        print("area mean beats range", max(area_beat_means) - min(area_beat_means))

        bp_shape_complete_avg_beats = []
        # split into even individual beats based on avg heartrate measure - same beat len as in diameter waveform
        for i in range(0, len(area_complete_avg) - beat_len + 1, beat_len):
            beat = bp_shape_complete_avg[i:i + beat_len]
            beat = scipy.signal.resample(beat, resample_len_per_beat)
            beat_max_index = np.argmax(beat)
            beat = np.concatenate((beat[beat_max_index:], beat[:beat_max_index]))
            bp_shape_complete_avg_beats.append(beat)

        velocity_complete = raw[subject_id][2][0][0][0]
        velocity_complete_all.append(velocity_complete)

        velocity_complete_time_frame = raw[subject_id][2][0][0][2]
        velocity_complete_time_frame_all.append(velocity_complete_time_frame)

        velocity_complete_avg = np.nanmean(velocity_complete, axis=1)
        vel_avg_inds = np.logical_not(np.isnan(velocity_complete_avg))
        velocity_complete_avg = velocity_complete_avg[vel_avg_inds]
        velocity_complete_avg_all.append(velocity_complete_avg)
        velocity_complete_min_all.append(min(velocity_complete_avg))
        velocity_complete_mean_all.append(np.mean(velocity_complete_avg))
        velocity_complete_max_all.append(max(velocity_complete_avg))

        velocity_complete_avg_time_frame = velocity_complete_time_frame[vel_avg_inds]
        velocity_complete_avg_time_frame_all.append(velocity_complete_avg_time_frame)

        velocity_complete_avg_beats = []
        # split into even individual beats based on avg heartrate measure
        num_beats = heartrate_avg / 60 * (velocity_complete_avg_time_frame[-1] - velocity_complete_avg_time_frame[0])
        beat_len = int(len(velocity_complete_avg) / num_beats)
        for i in range(0, len(velocity_complete_avg) - beat_len + 1, beat_len):
            beat = velocity_complete_avg[i:i + beat_len]
            beat = scipy.signal.resample(beat, resample_len_per_beat)
            beat_max_index = np.argmax(beat)
            beat = np.concatenate((beat[beat_max_index:], beat[:beat_max_index]))
            velocity_complete_avg_beats.append(beat)

        # truncate to shorter length
        min_beats = min(len(diameter_complete_avg_beats), len(velocity_complete_avg_beats))
        diameter_complete_avg_beats = diameter_complete_avg_beats[:min_beats]
        velocity_complete_avg_beats = velocity_complete_avg_beats[:min_beats]
        area_complete_avg_beats = area_complete_avg_beats[:min_beats]
        bp_shape_complete_avg_beats = bp_shape_complete_avg_beats[:min_beats]
        map_complete_avg_beats = map_complete_avg_beats[:min_beats]
        map_alt_complete_avg_beats = map_alt_complete_avg_beats[:min_beats]
        diameter_complete_avg_beats_all.append(np.array(diameter_complete_avg_beats))
        velocity_complete_avg_beats_all.append(np.array(velocity_complete_avg_beats))
        area_complete_avg_beats_all.append(np.array(area_complete_avg_beats))
        bp_shape_complete_avg_beats_all.append(np.array(bp_shape_complete_avg_beats))
        map_complete_avg_beats_all.append(np.array(map_complete_avg_beats))
        map_alt_complete_avg_beats_all.append(np.array(map_alt_complete_avg_beats))

        v_all.append(velocity)
        bp_shape_all.append(bp_shape)
        name_all.append(name)

        map_all.append(map)
        dbp_all.append(DBP_avg)
        heartrate_all.append(heartrate_avg)
        area_all.append(area)

        anthro = raw[subject_id][-2][0][0]
        age = anthro[0][0][0]
        if age == 0:
            age_all.append(30)
        else:
            age_all.append(age)

        if anthro[1][0] == 'male':
            gender_all.append(0)
        else:
            gender_all.append(1)

        height_all.append(anthro[2][0][0])
        weight_all.append(anthro[3][0][0])

    np.save(f'./npy/measured_mit_v1_part1_shape_all.npy', np.array(bp_shape_all))
    np.save(f'./npy/measured_mit_v1_part1_map_all.npy', np.array(map_all))
    np.save(f'./npy/measured_mit_v1_part1_v_all.npy', np.array(v_all))
    np.save(f'./npy/measured_mit_v1_part1_name_all.npy', np.array(name_all))
    np.save(f'./npy/measured_mit_v1_part1_id_all.npy', np.array(id_all))
    np.save(f'./npy/measured_mit_v1_part1_dbp_all.npy', np.array(dbp_all))
    np.save(f'./npy/measured_mit_v1_part1_area_all.npy', np.array(area_all))

    np.save(f'./npy/measured_mit_v1_part1_age_all.npy', np.array(age_all))
    np.save(f'./npy/measured_mit_v1_part1_gender_all.npy', np.array(gender_all))
    np.save(f'./npy/measured_mit_v1_part1_height_all.npy', np.array(height_all))
    np.save(f'./npy/measured_mit_v1_part1_weight_all.npy', np.array(weight_all))

    np.save(f'./npy/measured_mit_v1_part1_diameter_complete_all.npy', np.array(diameter_complete_all))
    np.save(f'./npy/measured_mit_v1_part1_diameter_complete_avg_all.npy', np.array(diameter_complete_avg_all))
    np.save(f'./npy/measured_mit_v1_part1_bp_shape_complete_avg_all.npy', np.array(bp_shape_complete_avg_all))
    np.save(f'./npy/measured_mit_v1_part1_diameter_complete_time_frame_all.npy', np.array(diameter_complete_time_frame_all))
    np.save(f'./npy/measured_mit_v1_part1_diameter_complete_avg_time_frame_all.npy', np.array(diameter_complete_avg_time_frame_all))
    np.save(f'./npy/measured_mit_v1_part1_velocity_complete_all.npy', np.array(velocity_complete_all))
    np.save(f'./npy/measured_mit_v1_part1_velocity_complete_avg_all.npy', np.array(velocity_complete_avg_all))
    np.save(f'./npy/measured_mit_v1_part1_velocity_complete_time_frame_all.npy', np.array(velocity_complete_time_frame_all))
    np.save(f'./npy/measured_mit_v1_part1_velocity_complete_avg_time_frame_all.npy', np.array(velocity_complete_avg_time_frame_all))

    np.save(f'./npy/measured_mit_v1_part1_heartrate_all.npy', np.array(heartrate_all))
    np.save(f'./npy/measured_mit_v1_part1_velocity_complete_avg_beats_all.npy', np.array(velocity_complete_avg_beats_all))
    np.save(f'./npy/measured_mit_v1_part1_diameter_complete_avg_beats_all.npy', np.array(diameter_complete_avg_beats_all))
    np.save(f'./npy/measured_mit_v1_part1_area_complete_avg_beats_all.npy', np.array(area_complete_avg_beats_all))
    np.save(f'./npy/measured_mit_v1_part1_bp_shape_complete_avg_beats_all.npy', np.array(bp_shape_complete_avg_beats_all))
    np.save(f'./npy/measured_mit_v1_part1_map_complete_avg_beats_all.npy', np.array(map_complete_avg_beats_all))

    np.save(f'./npy/measured_mit_v1_part1_diameter_complete_min_all.npy', np.array(diameter_complete_min_all))
    # mean is a single mean value for whole waveform, complete_avg is the 3 complete waveforms averaged out
    np.save(f'./npy/measured_mit_v1_part1_diameter_complete_mean_all.npy', np.array(diameter_complete_mean_all))
    np.save(f'./npy/measured_mit_v1_part1_diameter_complete_max_all.npy', np.array(diameter_complete_max_all))
    np.save(f'./npy/measured_mit_v1_part1_velocity_complete_min_all.npy', np.array(velocity_complete_min_all))
    np.save(f'./npy/measured_mit_v1_part1_velocity_complete_mean_all.npy', np.array(velocity_complete_mean_all))
    np.save(f'./npy/measured_mit_v1_part1_velocity_complete_max_all.npy', np.array(velocity_complete_max_all))

    np.save(f'./npy/measured_mit_v1_part1_area_complete_min_all.npy', np.array(area_complete_min_all))
    np.save(f'./npy/measured_mit_v1_part1_area_complete_mean_all.npy', np.array(area_complete_mean_all))
    np.save(f'./npy/measured_mit_v1_part1_area_complete_max_all.npy', np.array(area_complete_max_all))
    np.save(f'./npy/measured_mit_v1_part1_bp_shape_complete_min_all.npy', np.array(bp_shape_complete_min_all))
    np.save(f'./npy/measured_mit_v1_part1_bp_shape_complete_mean_all.npy', np.array(bp_shape_complete_mean_all))
    np.save(f'./npy/measured_mit_v1_part1_bp_shape_complete_max_all.npy', np.array(bp_shape_complete_max_all))

    np.save(f'./npy/measured_mit_v1_part1_map_alt_all.npy', np.array(map_alt_all))
    np.save(f'./npy/measured_mit_v1_part1_map_alt_complete_avg_beats_all.npy', np.array(map_alt_complete_avg_beats_all))

    print('finished')
